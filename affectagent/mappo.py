"""Token-level MAPPO update used by the official AffectAgent implementation (equations 8-12)."""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .distributed import reduce_sums, sync_gradients


AGENTS = ("q", "f", "g")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equation (9) for a single terminal-ended token trajectory (V(s_{T+1}) = 0)."""

    rewards = rewards.reshape(-1)
    values = values.reshape(-1)
    if rewards.numel() != values.numel():
        raise ValueError("rewards and values must contain the same number of time steps")
    if rewards.numel() == 0:
        return rewards.clone(), values.clone()
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros((), dtype=values.dtype, device=values.device)
    next_value = torch.zeros((), dtype=values.dtype, device=values.device)
    for step in range(rewards.numel() - 1, -1, -1):
        delta = rewards[step] + gamma * next_value - values[step]
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[step] = next_advantage
        next_value = values[step]
    returns = advantages + values
    return advantages, returns


class AdapterDisabledReference:
    """π_SFT for the KL term: the shared LLM evaluated with its RL adapter disabled.

    MAPPO merges the SFT adapters into the LLM and trains a new adapter (initialised to zero),
    so disabling that adapter recovers the SFT policy without a second copy of the LLM.
    """

    def __init__(self, llama_model):
        self.llama_model = llama_model

    @property
    def model(self):
        # Keeps ``reference.model.model.embed_tokens`` valid for prompt embedding.
        return self.llama_model.model

    def __call__(self, *args, **kwargs):
        with torch.no_grad(), self.llama_model.disable_adapter():
            return self.llama_model(*args, **kwargs)

    def eval(self):
        return self


class SequenceValueHead(nn.Module):
    """Critic V_φ applied to every token state s_t of the shared Actor.

    The states are detached: the Critic has its own parameters φ (paper Sec. 3.5), so value
    regression does not update the Actor θ.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.summary = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        # V_φ starts at zero, so the first advantages are the (KL-regularised) returns.
        nn.init.zeros_(self.summary[-1].weight)
        nn.init.zeros_(self.summary[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)
        hidden_states = hidden_states.detach().to(dtype=self.summary[0].weight.dtype)
        return self.summary(hidden_states).squeeze(-1)


class PolicyGradientUpdater:
    """Joint Actor-Critic PPO with terminal task/KL reward and token-level GAE."""

    def __init__(
        self,
        model,
        reference_model,
        value_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        kl_coef: float = 0.1,
        ppo_epochs: int = 4,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        fusion_modules: Sequence[nn.Module] = (),
        reference_fusion: Optional[Tuple[nn.Module, nn.Module]] = None,
    ):
        """
        Args:
            model: AffectGPT whose ``llama_model`` is the shared Actor π_θ
            reference_model: frozen π_SFT (callable like ``llama_model``)
            fusion_modules: trainable (RAAF, MB-MoE), kept in eval mode during replay
            reference_fusion: frozen SFT copies of (RAAF, MB-MoE).  The Generator's input passes
                through the fusion modules, so π_SFT of the Generator uses these copies.
        """
        self.model = model
        self.reference_model = reference_model
        self.value_head = value_head
        self.optimizer = optimizer
        self.kl_coef = kl_coef
        self.ppo_epochs = ppo_epochs
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.fusion_modules = list(fusion_modules)
        self.reference_fusion = reference_fusion
        self.parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]

    @staticmethod
    def _aligned_length(*tensors: torch.Tensor) -> int:
        return min(tensor.reshape(-1).numel() for tensor in tensors)

    def _set_mode(self) -> None:
        # The RL adapter has no dropout and RAAF/MB-MoE stay in eval mode, so replay evaluates
        # exactly the distribution that sampled the rollout.  ``train()`` on the LLM only turns
        # on gradient checkpointing when it is enabled.
        self.model.eval()
        self.model.llama_model.train()
        self.reference_model.eval()
        self.value_head.train()
        for module in self.fusion_modules:
            module.eval()

    def _build_replay_buffer(self, trajectories, pipeline) -> List[Dict[str, object]]:
        # π_θold, π_SFT and V_φold are evaluated without gradients (and without checkpointing).
        self.model.llama_model.eval()
        replay_buffer = []
        for trajectory in trajectories:
            result = trajectory["result"]
            agent_name = str(trajectory["agent"])
            task_reward = float(trajectory["reward"])
            with torch.no_grad():
                old_log_probs, old_states = pipeline.compute_agent_log_probs_and_state(
                    result, agent_name, self.model.llama_model,
                )
                if old_states is None or old_log_probs.numel() == 0:
                    continue
                ref_log_probs = pipeline.compute_agent_log_probs(
                    result, agent_name, self.reference_model, fusion=self.reference_fusion,
                ).to(old_log_probs.device)
                old_values = self.value_head(old_states).float()

            length = self._aligned_length(old_log_probs, ref_log_probs, old_values)
            old_log_probs = old_log_probs[:length]
            ref_log_probs = ref_log_probs[:length]
            old_values = old_values[:length]

            # Equation (10): task reward and sequence-level KL are assigned at t = T_i only.
            sequence_kl = (old_log_probs - ref_log_probs).sum()
            terminal_reward = task_reward - self.kl_coef * sequence_kl
            rewards = torch.zeros_like(old_values)
            rewards[-1] = terminal_reward
            advantages, returns = compute_gae(rewards, old_values, self.gamma, self.gae_lambda)
            replay_buffer.append({
                "result": result,
                "agent": agent_name,
                "task_reward": task_reward,
                "regularized_reward": float(terminal_reward),
                "sequence_kl": float(sequence_kl),
                "old_log_probs": old_log_probs.detach(),
                "ref_log_probs": ref_log_probs.detach(),
                "old_values": old_values.detach(),
                "returns": returns.detach(),
                "advantages": advantages.detach(),
            })
        self._normalize_advantages(replay_buffer)
        return replay_buffer

    def _normalize_advantages(self, replay_buffer: List[Dict[str, object]]) -> None:
        """Whiten token advantages per agent, with statistics pooled over all ranks."""

        local = {}
        for agent in AGENTS:
            items = [item["advantages"] for item in replay_buffer if item["agent"] == agent]
            flat = torch.cat(items) if items else torch.zeros(0, device=self.device)
            local[f"{agent}_n"] = flat.numel()
            local[f"{agent}_sum"] = float(flat.sum()) if flat.numel() else 0.0
            local[f"{agent}_sq"] = float((flat ** 2).sum()) if flat.numel() else 0.0
        stats = reduce_sums(local, self.device)
        for agent in AGENTS:
            count = stats[f"{agent}_n"]
            if count < 2:
                continue
            mean = stats[f"{agent}_sum"] / count
            std = max(stats[f"{agent}_sq"] / count - mean ** 2, 0.0) ** 0.5
            if std <= 1e-8:
                continue
            for item in replay_buffer:
                if item["agent"] == agent:
                    item["advantages"] = ((item["advantages"] - mean) / std).detach()

    def _item_loss(self, item, pipeline) -> Optional[Tuple[torch.Tensor, Dict[str, float]]]:
        policy_log_probs, states = pipeline.compute_agent_log_probs_and_state(
            item["result"], item["agent"], self.model.llama_model,
        )
        if policy_log_probs.numel() == 0 or states is None:
            return None
        value_predictions = self.value_head(states).float()
        length = self._aligned_length(policy_log_probs, value_predictions, item["old_log_probs"])
        policy_log_probs = policy_log_probs[:length]
        value_predictions = value_predictions[:length]
        old_log_probs = item["old_log_probs"][:length]
        advantages = item["advantages"][:length]

        # Equation (8): clipped surrogate.
        ratio = torch.exp(torch.clamp(policy_log_probs - old_log_probs, -20.0, 20.0))
        surrogate_1 = ratio * advantages
        surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

        # Equations (11)-(12): clipped value regression around V_φold.
        old_values = item["old_values"][:length]
        value_targets = item["returns"][:length]
        clipped_values = old_values + torch.clamp(value_predictions - old_values, -self.clip_range, self.clip_range)
        value_loss = torch.max(
            (value_predictions - value_targets).pow(2),
            (clipped_values - value_targets).pow(2),
        ).mean()

        loss = policy_loss + self.value_coef * value_loss
        with torch.no_grad():
            stats = {
                "policy": float(policy_loss),
                "value": float(value_loss),
                "kl": float((policy_log_probs - item["ref_log_probs"][:length]).mean()),
                "clip_frac": float(((ratio - 1.0).abs() > self.clip_range).float().mean()),
            }
        return loss, stats

    def update_step(self, trajectories, pipeline) -> Dict[str, float]:
        """One MAPPO update. Every rank must call this (possibly with no trajectories),
        because advantage statistics and gradients are reduced across ranks."""

        self._set_mode()
        replay_buffer = self._build_replay_buffer(trajectories or [], pipeline)
        self._set_mode()
        totals = {"policy": 0.0, "value": 0.0, "kl": 0.0, "clip_frac": 0.0, "count": 0}
        update_rounds = 0

        for _ in range(self.ppo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            # Global number of trajectories so the summed gradients equal the global mean.
            global_count = reduce_sums({"n": len(replay_buffer)}, self.device)["n"]
            if global_count == 0:
                break
            for item in replay_buffer:
                output = self._item_loss(item, pipeline)
                if output is None:
                    continue
                loss, stats = output
                (loss / global_count).backward()
                for key, value in stats.items():
                    totals[key] += value
                totals["count"] += 1

            sync_gradients(self.parameters)
            torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
            self.optimizer.step()
            update_rounds += 1

        count = max(totals["count"], 1)
        n_items = max(len(replay_buffer), 1)
        return {
            "policy_loss": totals["policy"] / count,
            "value_loss": totals["value"] / count,
            "loss": (totals["policy"] + self.value_coef * totals["value"]) / count,
            "mean_kl": totals["kl"] / count,
            "clip_frac": totals["clip_frac"] / count,
            "mean_reward": sum(item["task_reward"] for item in replay_buffer) / n_items,
            "mean_regularized_reward": sum(item["regularized_reward"] for item in replay_buffer) / n_items,
            "mean_sequence_kl": sum(item["sequence_kl"] for item in replay_buffer) / n_items,
            "num_trajectories": len(replay_buffer),
            "update_rounds": update_rounds,
        }
