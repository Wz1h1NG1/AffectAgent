"""Minimal torch.distributed helpers for multi-GPU rollout + synchronous MAPPO updates.

Every rank collects its own rollouts; gradients are all-reduced manually before each optimizer
step, so all ranks keep identical parameters without wrapping the shared Actor in DDP (one PPO
step runs several forward passes per agent, which DDP's reducer does not support well).
"""

import os
from typing import Dict, Iterable

import torch
import torch.distributed as dist


def init_distributed(gpu: int = 0):
    """Return (rank, world_size, device). Uses torchrun's environment when present.

    The current CUDA device is set explicitly so that libraries placing tensors on "cuda"
    (e.g. the E5 retriever) use this process's GPU.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), world_size, f"cuda:{local_rank}"
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return 0, 1, f"cuda:{gpu}"
    return 0, 1, "cpu"


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_sum(values: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def reduce_sums(values: Dict[str, float], device) -> Dict[str, float]:
    """Sum named scalars over ranks (used for metrics and normalisation statistics)."""

    if not values:
        return {}
    keys = sorted(values)
    tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=device)
    tensor = all_reduce_sum(tensor)
    return {key: float(value) for key, value in zip(keys, tensor.tolist())}


def sync_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    """All-reduce (sum) gradients; parameters without a local gradient contribute zeros."""

    if not is_distributed():
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def broadcast_tensors(tensors: Iterable[torch.Tensor]) -> None:
    """Copy rank-0 values of freshly initialised trainable tensors to every rank."""

    if not is_distributed():
        return
    for tensor in tensors:
        dist.broadcast(tensor.data, src=0)


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()
