"""Multimodal prompt embedding shared by rollout, PPO replay and role-specific SFT.

Every agent prompt is tokenised once; ``<FrameHere>/<FaceHere>/<AudioHere>`` placeholders are
then replaced by audiovisual tokens.  Rollout, replay and SFT all go through this class so that
the policy is trained on exactly the inputs it acts on.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import config as affectgpt_config

from .prompts import render_prompt


# Evidence text is shortened step by step when a prompt exceeds the token budget.
EVIDENCE_CHAR_BUDGETS = (400, 200, 100)


def embedding_layer(llama_model):
    # AffectGPT wraps the HF causal LM with PEFT: PeftModel -> CausalLM -> base model.
    return llama_model.model.model.embed_tokens


def model_dtype(llama_model) -> torch.dtype:
    return embedding_layer(llama_model).weight.dtype


class MultimodalPromptEncoder:
    def __init__(self, chat, max_prompt_tokens: int = 2048):
        self.chat = chat
        self.tokenizer = chat.tokenizer
        self.device = chat.device
        self.max_prompt_tokens = max_prompt_tokens
        vocab = self.tokenizer.get_vocab()
        # placeholder id -> (feature key, number of query tokens)
        self.placeholders = {
            vocab[affectgpt_config.DEFAULT_FRAME_PATCH_TOKEN]: ("video", chat.num_video_query_token),
            vocab[affectgpt_config.DEFAULT_FACE_PATCH_TOKEN]: ("video", chat.num_video_query_token),
            vocab[affectgpt_config.DEFAULT_AUDIO_PATCH_TOKEN]: ("audio", chat.num_audio_query_token),
            vocab[affectgpt_config.DEFAULT_MULTI_PATCH_TOKEN]: ("multi", chat.num_multi_query_token),
            vocab[affectgpt_config.DEFAULT_IMAGE_PATCH_TOKEN]: ("image", chat.num_image_query_token),
        }
        self._warned_missing = set()

    # ────────────────────────── features ──────────────────────────

    def extract_features(self, sample_data: dict, face_or_frame: str) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, np.ndarray]]:
        """Encode audio/video with the AffectGPT projectors.

        Returns ``features`` ({"video": [1, T_v, D], "audio": [1, T_a, D]}) and the pooled,
        L2-normalised vectors used by perceptual retrieval.
        """

        _, audio_llms = self.chat.postprocess_audio(sample_data)
        _, frame_llms = self.chat.postprocess_frame(sample_data)
        _, face_llms = self.chat.postprocess_face(sample_data)
        if "face" in face_or_frame:
            video_llms = face_llms if face_llms is not None else frame_llms
        else:
            video_llms = frame_llms if frame_llms is not None else face_llms

        features = {"video": video_llms, "audio": audio_llms}
        vectors = {}
        for key, value in features.items():
            if value is None:
                continue
            vector = value.detach().float().mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(vector)
            vectors[key] = vector / norm if norm > 0 else vector
        return features, vectors

    # ────────────────────────── prompts ──────────────────────────

    def tokenize(self, prompt: str) -> torch.Tensor:
        """``<bos> prompt``, the input layout AffectGPT is trained with (see its ``collater``)."""

        prompt = self.chat.replace_token_for_multimodal(prompt)
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_id is not None:
            ids = torch.cat([torch.tensor([bos_id], dtype=ids.dtype), ids])
        return ids

    def tokenize_target(self, text: str) -> torch.Tensor:
        """SFT target ``" {text}###<eos>"``, the answer layout of AffectGPT training."""

        ids = self.tokenizer(f" {text}###", return_tensors="pt", add_special_tokens=False).input_ids[0]
        return torch.cat([ids, torch.tensor([self.tokenizer.eos_token_id], dtype=ids.dtype)])

    def fits(self, prompt_ids: torch.Tensor) -> bool:
        return self.max_prompt_tokens <= 0 or prompt_ids.numel() <= self.max_prompt_tokens

    def prompt_ids(self, build_messages: Callable[[int], List[Dict]]) -> torch.Tensor:
        """Tokenise ``build_messages(evidence_char_budget)``, shortening evidence until it fits.

        The prompt is never cut from the right, so the trailing ``###Assistant:`` tag is kept.
        """

        prompt_ids = None
        for budget in EVIDENCE_CHAR_BUDGETS:
            prompt_ids = self.tokenize(render_prompt(build_messages(budget)))
            if self.fits(prompt_ids):
                return prompt_ids
        print(f"[AffectAgent] Prompt has {prompt_ids.numel()} tokens (budget {self.max_prompt_tokens}).")
        return prompt_ids

    def num_query_tokens(self, key: str) -> int:
        return next(count for name, count in self.placeholders.values() if name == key)

    def embed(
        self,
        prompt_ids: torch.Tensor,
        features: Dict[str, Optional[torch.Tensor]],
        llama_model,
    ) -> torch.Tensor:
        """Token embeddings with every placeholder block replaced by its features: [L, D]."""

        input_ids = prompt_ids.to(self.device)
        safe_ids = input_ids.clone()
        for placeholder_id in self.placeholders:
            safe_ids[safe_ids == placeholder_id] = 0
        embeds = embedding_layer(llama_model)(safe_ids)
        dtype = embeds.dtype

        for placeholder_id, (key, num_tokens) in self.placeholders.items():
            positions = torch.where(input_ids == placeholder_id)[0]
            if positions.numel() == 0:
                continue
            start = int(positions[0])
            if positions.numel() != num_tokens or int(positions[-1]) != start + num_tokens - 1:
                raise ValueError(f"Expected {num_tokens} consecutive {key} placeholder tokens in the prompt.")
            value = features.get(key)
            if value is None:
                if key not in self._warned_missing:
                    print(f"[AffectAgent] No {key} features for a prompt placeholder; using zeros.")
                    self._warned_missing.add(key)
                value = embeds.new_zeros(num_tokens, embeds.size(-1))
            value = value[0] if value.dim() == 3 else value
            if value.size(0) != num_tokens:
                raise ValueError(f"{key} features have {value.size(0)} tokens, prompt expects {num_tokens}.")
            embeds = torch.cat([embeds[:start], value.to(self.device, dtype), embeds[start + num_tokens:]], dim=0)
        return embeds

    # ────────────────────────── policy calls ──────────────────────────

    @torch.no_grad()
    def generate(
        self,
        llama_model,
        prompt_embeds: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, str]:
        # Qwen2.5 ships a generation_config with top_k=20 and repetition_penalty=1.05.  Both are
        # overridden so that the sampling distribution is softmax(logits / T) with top-p only,
        # which is what ``response_log_probs_and_states`` evaluates for the PPO ratio.
        sampling = {"temperature": temperature, "top_p": top_p, "top_k": 0} if do_sample else {}
        outputs = llama_model.generate(
            inputs_embeds=prompt_embeds.unsqueeze(0),
            attention_mask=torch.ones(1, prompt_embeds.size(0), dtype=torch.long, device=self.device),
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.chat.stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
            repetition_penalty=1.0,
            **sampling,
        )
        # With inputs_embeds only, generate() returns just the new tokens (stop token included).
        response_ids = outputs[0].detach().cpu()
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_ids, text.split("###")[0].strip()

    def _forward(self, llama_model, prompt_embeds: torch.Tensor, response_ids: torch.Tensor, need_states: bool = True):
        response_ids = response_ids.to(self.device).reshape(-1)
        response_embeds = embedding_layer(llama_model)(response_ids)
        full_embeds = torch.cat([prompt_embeds, response_embeds], dim=0).unsqueeze(0)
        full_embeds = full_embeds.to(model_dtype(llama_model))
        outputs = llama_model(
            inputs_embeds=full_embeds,
            attention_mask=torch.ones(1, full_embeds.size(1), dtype=torch.long, device=self.device),
            output_hidden_states=need_states,
            use_cache=False,
            return_dict=True,
        )
        # Position P-1+t holds state s_t, which emits response token t.
        start = prompt_embeds.size(0) - 1
        end = start + response_ids.numel()
        states = outputs.hidden_states[-1][0, start:end] if need_states else None
        return response_ids, outputs.logits[0, start:end], states

    def response_log_probs_and_states(
        self,
        llama_model,
        prompt_embeds: torch.Tensor,
        response_ids: Optional[torch.Tensor],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Per-token log π(a_t | s_t) under softmax(logits / T) and the states s_t.

        With top_p = 1 (the MAPPO default) this is exactly the rollout distribution.  The same
        formula is used for π_θ, π_θold and π_SFT, so ratios and KL terms are consistent.
        """

        if response_ids is None or response_ids.numel() == 0:
            return torch.empty(0, device=self.device), None
        response_ids, logits, states = self._forward(llama_model, prompt_embeds, response_ids)
        log_probs = torch.log_softmax(logits.float() / max(temperature, 1e-5), dim=-1)
        token_log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs, states

    def lm_loss(self, llama_model, prompt_embeds: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Mean token NLL of ``target_ids`` given the prompt (role-specific SFT objective)."""

        target_ids, logits, _ = self._forward(llama_model, prompt_embeds, target_ids, need_states=False)
        return F.cross_entropy(logits.float(), target_ids)
