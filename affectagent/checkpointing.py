"""Canonical AffectAgent checkpoint file names."""

import os
from typing import Dict, Iterable, Optional


CHECKPOINT_FILES: Dict[str, tuple] = {
    "actor": ("actor_adapter.pth",),
    "raaf": ("raaf.pth",),
    "mb_moe": ("mb_moe.pth",),
    "critic": ("critic.pth",),
    "sft": ("sft_checkpoint.pth",),
}


def checkpoint_candidates(component: str) -> Iterable[str]:
    try:
        return CHECKPOINT_FILES[component]
    except KeyError as error:
        raise ValueError(f"Unknown checkpoint component: {component}") from error


def resolve_checkpoint_path(directory: str, component: str) -> Optional[str]:
    for filename in checkpoint_candidates(component):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            return path
    return None
