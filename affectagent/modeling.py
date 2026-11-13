"""Backbone loading, trainable-parameter selection and checkpoints shared by SFT, MAPPO and evaluation.

Training stages
---------------
* Audiovisual encoders, Q-Formers and projectors stay frozen in every AffectAgent stage, so the
  perceptual token store built from the AffectGPT checkpoint remains valid throughout.
* Role-specific SFT updates the LLM LoRA adapters of AffectGPT (plus RAAF / MB-MoE through the
  Generator loss) and saves an AffectGPT-format checkpoint usable as ``model.ckpt_3``.
* MAPPO merges the SFT adapters into the LLM and attaches a fresh policy adapter.  The frozen
  reference policy π_SFT is the same network with that adapter disabled, so no second copy of
  the LLM is kept in GPU memory.
"""

import json
import os
from typing import Dict, Optional

import torch

from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat


LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ENCODER_PREFIXES = ("visual_encoder.", "acoustic_encoder.")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_affectgpt(cfg, device: str, llm_dtype: str = "bf16"):
    """Build AffectGPT from the config (``model.ckpt*`` are loaded by ``from_config``)."""

    model_cls = registry.get_model_class(cfg.model_cfg.arch)
    model = model_cls.from_config(cfg.model_cfg)
    cast_llm_weights(model.llama_model, DTYPES[llm_dtype])
    model = model.to(device).eval()
    return model, Chat(model, cfg.model_cfg, device=device)


def cast_llm_weights(llama_model, dtype: torch.dtype) -> None:
    """Cast LLM base weights (Qwen2.5 is a bf16 model); LoRA adapters stay in float32."""

    for name, parameter in llama_model.named_parameters():
        if "lora_" not in name:
            parameter.data = parameter.data.to(dtype)


def is_adapter_parameter(name: str) -> bool:
    return name.startswith("llama_model.") and "lora_" in name


def train_only_llm_adapters(model) -> int:
    """Freeze everything except the LLM LoRA adapters; return the number of trainable values."""

    count = 0
    for name, parameter in model.named_parameters():
        trainable = is_adapter_parameter(name)
        parameter.requires_grad_(trainable)
        count += parameter.numel() if trainable else 0
    return count


def enable_gradient_checkpointing(llama_model) -> None:
    # Non-reentrant checkpointing works with inputs_embeds whose leaves do not require grad.
    # It is only active in train mode, so rollout generation keeps its KV cache.
    llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def attach_policy_adapter(model, lora_r: int = 16, lora_alpha: int = 32) -> None:
    """Merge the SFT adapters into the LLM and add a fresh RL adapter initialised to zero."""

    from peft import LoraConfig, TaskType, get_peft_model

    merged = model.llama_model.merge_and_unload()
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=LORA_TARGET_MODULES,
    )
    model.llama_model = get_peft_model(merged, config)
    train_only_llm_adapters(model)


# ══════════════════════════════════════════════════════
# Checkpoints
# ══════════════════════════════════════════════════════

def affectgpt_checkpoint_state(model) -> Dict[str, torch.Tensor]:
    """Weights AffectGPT keeps in its own checkpoints: LLM adapters, Q-Formers and projectors."""

    state = {}
    for name, tensor in model.state_dict().items():
        if name.startswith(ENCODER_PREFIXES):
            continue
        if name.startswith("llama_model.") and "lora_" not in name:
            continue
        state[name] = tensor.detach().cpu()
    return state


def policy_adapter_state(model) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if is_adapter_parameter(name)
    }


def save_mappo_checkpoint(directory: str, model, raaf, mb_moe, critic, metadata: Dict) -> None:
    os.makedirs(directory, exist_ok=True)
    torch.save(policy_adapter_state(model), os.path.join(directory, "actor_adapter.pth"))
    torch.save(raaf.state_dict(), os.path.join(directory, "raaf.pth"))
    torch.save(mb_moe.state_dict(), os.path.join(directory, "mb_moe.pth"))
    torch.save(critic.state_dict(), os.path.join(directory, "critic.pth"))
    manifest = {
        "format_version": 3,
        "implementation": "AffectAgent",
        "files": {"actor": "actor_adapter.pth", "raaf": "raaf.pth", "mb_moe": "mb_moe.pth", "critic": "critic.pth"},
        **metadata,
    }
    with open(os.path.join(directory, "checkpoint_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)


def read_manifest(directory: str) -> Optional[Dict]:
    path = os.path.join(directory, "checkpoint_manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def load_policy_adapter(model, directory: str) -> None:
    """Attach the RL adapter described by ``checkpoint_manifest.json`` and load its weights."""

    manifest = read_manifest(directory) or {}
    attach_policy_adapter(model, manifest.get("lora_r", 16), manifest.get("lora_alpha", 32))
    path = os.path.join(directory, manifest.get("files", {}).get("actor", "actor_adapter.pth"))
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_adapters = [name for name in missing if is_adapter_parameter(name)]
    if missing_adapters or unexpected:
        raise RuntimeError(
            f"Adapter checkpoint mismatch: {len(missing_adapters)} missing, {len(unexpected)} unexpected keys"
        )
    model.eval()


def load_module_state(module: torch.nn.Module, path: Optional[str], device) -> bool:
    if not path or not os.path.isfile(path):
        return False
    module.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return True
