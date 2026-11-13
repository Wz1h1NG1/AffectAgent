"""RAAF (paper equations 5-6) and MB-MoE (equation 7).

Both modules sit between the AffectGPT projectors and the LLM and start as the identity map
(x̂_m = x_m and x̃_m = x̂_m): the attention output projection of RAAF and the last layer of
every MB-MoE expert are zero-initialised, so training starts from the audiovisual tokens the
backbone was trained on.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalAugmentedAdaptiveFusion(nn.Module):
    """
    Retrieval-Augmented Adaptive Fusion (RAAF), equations (5)-(6):

        h_m = Attn(x_m, E_perc^m, E_perc^m)
        x̂_m = x_m + σ(W_m [x_m; h_m]) ⊙ h_m,      m ∈ {v, a}
    """

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        # 交叉注意力：Query 为当前样本，Key/Value 为感知检索得到的支持样本
        self.video_cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.audio_cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

        # 门控网络 σ(W_m[x_m; h_m])：决定吸收多少证据特征
        self.gate_video = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.gate_audio = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # h_m = 0 at initialisation, hence x̂_m = x_m; gradients still reach out_proj.
        for attention in (self.video_cross_attn, self.audio_cross_attn):
            nn.init.zeros_(attention.out_proj.weight)
            nn.init.zeros_(attention.out_proj.bias)

    @staticmethod
    def _fuse(
        attention: nn.MultiheadAttention,
        gate: nn.Module,
        current: Optional[torch.Tensor],
        evidence: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], float]:
        if current is None or evidence is None:
            return current, 0.0
        attended, _ = attention(query=current, key=evidence, value=evidence, need_weights=False)
        weights = gate(torch.cat([current, attended], dim=-1))
        return current + weights * attended, float(weights.detach().mean())

    def forward(self, cur_video, cur_audio, ev_video, ev_audio):
        """
        Args:
            cur_video / cur_audio: [batch, T, dim] current-sample tokens (either may be None)
            ev_video / ev_audio:   [batch, T_e, dim] perceptual evidence E_perc (either may be None)

        Returns:
            fused_video, fused_audio, video_gate_mean, audio_gate_mean
        """
        fused_video, video_gate_mean = self._fuse(self.video_cross_attn, self.gate_video, cur_video, ev_video)
        fused_audio, audio_gate_mean = self._fuse(self.audio_cross_attn, self.gate_audio, cur_audio, ev_audio)
        return fused_video, fused_audio, video_gate_mean, audio_gate_mean


class ResidualExpert(nn.Module):
    """Expert transformation f_j(x) = x + W_2 GELU(W_1 LN(x)); W_2 starts at zero."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class ModalityBalancingMoE(nn.Module):
    """
    Modality-Balancing Mixture of Experts (MB-MoE), equation (7):

        g = [pool(x̂_v); pool(x̂_a)],   K = TopK(router(g)),   α = softmax(router(g)_K)
        x̃_m = Σ_{j∈K} α_j f_j(x̂_m),    m ∈ {v, a}

    The same routing weights α are shared by both modalities.
    """

    def __init__(self, dim: int, n_experts: int = 4, top_k: int = 2, expert_hidden_dim: int = 0):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        hidden_dim = expert_hidden_dim if expert_hidden_dim > 0 else dim
        self.experts = nn.ModuleList([ResidualExpert(dim, hidden_dim) for _ in range(n_experts)])
        # 路由器：输入为 video 与 audio 全局池化特征的拼接
        self.router = nn.Linear(dim * 2, n_experts)

    def _global_state(self, video: Optional[torch.Tensor], audio: Optional[torch.Tensor]) -> torch.Tensor:
        reference = video if video is not None else audio
        zeros = reference.new_zeros(reference.size(0), self.dim)
        v_global = video.mean(dim=1) if video is not None else zeros
        a_global = audio.mean(dim=1) if audio is not None else zeros
        return torch.cat([v_global, a_global], dim=-1)

    def route(self, video, audio) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(self._global_state(video, audio))           # [batch, n_experts]
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        return top_k_indices, F.softmax(top_k_logits, dim=-1)            # α over the selected experts

    def _mix(self, x: Optional[torch.Tensor], dense_weights: torch.Tensor, active) -> Optional[torch.Tensor]:
        if x is None:
            return None
        output = torch.zeros_like(x)
        for expert_index in active:
            weight = dense_weights[:, expert_index].view(-1, 1, 1)
            output = output + weight * self.experts[expert_index](x)
        return output

    def forward(self, fused_video, fused_audio, return_routing: bool = False):
        """
        Args:
            fused_video / fused_audio: [batch, T, dim] RAAF outputs (either may be None)

        Returns:
            balanced_video, balanced_audio[, (top_k_indices, top_k_weights)]
        """
        top_k_indices, top_k_weights = self.route(fused_video, fused_audio)
        dense_weights = torch.zeros(
            top_k_weights.size(0), self.n_experts, dtype=top_k_weights.dtype, device=top_k_weights.device
        ).scatter(1, top_k_indices, top_k_weights)
        # 只计算被选中的专家
        active = sorted(set(top_k_indices.reshape(-1).tolist()))
        balanced_video = self._mix(fused_video, dense_weights, active)
        balanced_audio = self._mix(fused_audio, dense_weights, active)
        if return_routing:
            return balanced_video, balanced_audio, (top_k_indices.detach(), top_k_weights.detach())
        return balanced_video, balanced_audio


# Concise paper names used by the public API.
RAAF = RetrievalAugmentedAdaptiveFusion
MBMoE = ModalityBalancingMoE

# Backward-compatible pre-release implementation names.
SupportFusion = RetrievalAugmentedAdaptiveFusion
ModalityMoE = ModalityBalancingMoE
