"""Command-line plumbing shared by role-specific SFT, MAPPO training and evaluation."""

import argparse
import json
import math
import os
import random

import numpy as np
import torch

import decord

decord.bridge.set_bridge("torch")

import my_affectgpt  # noqa: F401,E402  (registers AffectGPT models, processors and datasets)
from my_affectgpt.common.config import Config  # noqa: E402

from .fusion_modules import MBMoE, RAAF  # noqa: E402
from .orchestrator import AffectAgentPipeline  # noqa: E402
from .retriever_service import DualChannelRetriever  # noqa: E402


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cfg-path", required=True, help="AffectGPT YAML configuration")
    parser.add_argument("--options", nargs="+", default=None,
                        help="Config overrides, e.g. model.ckpt_3=/path/to/checkpoint.pth")
    parser.add_argument("--llm-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="Device for single-process runs")


def add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--semantic-index-dir", default="affectagent/artifacts/semantic_index")
    parser.add_argument("--multimodal-index-dir", default="retrieval/faiss/artifacts/mercaptionplus")
    parser.add_argument("--retrieval-top-k", type=int, default=3, help="K cognitive candidates per query group")
    parser.add_argument("--channel-b-top-k", type=int, default=1, help="Perceptual evidence samples for RAAF")


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens-q", type=int, default=256)
    parser.add_argument("--max-new-tokens-filter", "--max-new-tokens-s", dest="max_new_tokens_f", type=int, default=96)
    parser.add_argument("--max-new-tokens-g", type=int, default=192)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt-lang", choices=["en", "zh"], default="en")


FUSION_KEYS = ("fusion_heads", "moe_experts", "moe_top_k", "moe_hidden_dim")


def add_fusion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fusion-heads", type=int, default=8, help="RAAF cross-attention heads")
    parser.add_argument("--moe-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-hidden-dim", type=int, default=1024, help="Hidden width of every MB-MoE expert")


def apply_saved_fusion_config(args, directory: str) -> None:
    """Use the RAAF / MB-MoE shapes recorded by role-specific SFT (``sft_config.json``)."""

    path = os.path.join(directory, "sft_config.json") if directory else ""
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as stream:
        saved = json.load(stream)
    for key in FUSION_KEYS:
        if key in saved and getattr(args, key) != saved[key]:
            print(f"[AffectAgent] Using {key}={saved[key]} from {path}")
            setattr(args, key, saved[key])


def load_config(args) -> Config:
    return Config(argparse.Namespace(cfg_path=args.cfg_path, options=args.options))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_retriever(args) -> DualChannelRetriever:
    return DualChannelRetriever(
        semantic_index_dir=args.semantic_index_dir,
        multimodal_index_dir=args.multimodal_index_dir,
    )


def build_fusion_modules(args, hidden_size: int, device):
    """RAAF and MB-MoE operate in the LLM embedding space (identity at initialisation)."""

    raaf = RAAF(dim=hidden_size, n_heads=args.fusion_heads).to(device)
    mb_moe = MBMoE(
        dim=hidden_size, n_experts=args.moe_experts, top_k=args.moe_top_k, expert_hidden_dim=args.moe_hidden_dim,
    ).to(device)
    return raaf, mb_moe


def build_pipeline(args, chat, retriever, raaf, mb_moe, **overrides) -> AffectAgentPipeline:
    options = dict(
        max_new_tokens_q=args.max_new_tokens_q,
        max_new_tokens_s=args.max_new_tokens_f,
        max_new_tokens_g=args.max_new_tokens_g,
        retrieval_top_k=args.retrieval_top_k,
        channel_b_top_k=args.channel_b_top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    options.update(overrides)
    return AffectAgentPipeline(chat=chat, retriever=retriever, raaf=raaf, mb_moe=mb_moe, **options)


def warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.1):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
