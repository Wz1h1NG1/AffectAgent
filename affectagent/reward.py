"""Paper-aligned shared and local affective rewards (equations 1-4)."""

from typing import Dict, Iterable, List, Optional

from .labels import get_emotion_wheel, label_f1
from .schemas import RewardBreakdown, RolloutResult


class AffectiveRewardComputer:
    """Compute F1 task scores and the paper's incremental agent rewards.

    Predictions are open-vocabulary emotion words.  With ``use_emotion_wheel`` the label-set F1
    is computed after MER-UniBench emotion-wheel normalisation, so synonyms of the ground truth
    are not scored as errors; otherwise normalised surface forms are compared.
    """

    def __init__(
        self,
        lambda_planner: float = 1.0,
        lambda_filter: float = 1.0,
        strict_counterfactuals: bool = True,
        use_emotion_wheel: bool = True,
        running_uar: Optional[Dict[str, float]] = None,
    ):
        if lambda_planner < 0 or lambda_filter < 0:
            raise ValueError("Reward coefficients must be non-negative.")
        self.lambda_planner = float(lambda_planner)
        self.lambda_filter = float(lambda_filter)
        self.strict_counterfactuals = strict_counterfactuals
        self.use_emotion_wheel = use_emotion_wheel
        # Accepted only for compatibility with the pre-release constructor.
        self.running_uar = running_uar or {}

    @staticmethod
    def compute_f1_score(prediction: str, ground_truth: str, use_emotion_wheel: bool = False) -> float:
        """Per-sample label-set F1 used as Score_* in equation (1)."""

        return label_f1(prediction, ground_truth, get_emotion_wheel() if use_emotion_wheel else None)

    # Compatibility name used by older logging code.
    compute_r_task = compute_f1_score

    def _score(self, output, result: RolloutResult) -> float:
        prediction = output.prediction if output is not None else ""
        return self.compute_f1_score(prediction, result.sample.ground_truth, self.use_emotion_wheel)

    def compute_pipeline_rewards(self, result: RolloutResult) -> RewardBreakdown:
        """Evaluate Score_full/label/rank and equations (2)-(4)."""

        if result.sample is None or result.generator_output is None:
            raise ValueError("A full rollout and ground-truth sample are required.")

        counterfactuals_complete = (
            result.label_baseline_output is not None
            and result.rank_baseline_output is not None
        )
        if self.strict_counterfactuals and not counterfactuals_complete:
            raise ValueError(
                "Paper-aligned rewards require both Score_label and Score_rank predictions. "
                "Run AffectAgentPipeline with compute_counterfactual_rewards=True."
            )

        score_full = self._score(result.generator_output, result)
        score_label = self._score(result.label_baseline_output, result)
        score_rank = self._score(result.rank_baseline_output, result)

        r_shared = score_full
        r_planner = r_shared + self.lambda_planner * (score_full - score_label)
        r_filter = r_shared + self.lambda_filter * (score_full - score_rank)
        r_generator = r_shared

        return RewardBreakdown(
            score_full=score_full,
            score_label=score_label,
            score_rank=score_rank,
            r_shared=r_shared,
            r_planner=r_planner,
            r_filter=r_filter,
            r_generator=r_generator,
            lambda_planner=self.lambda_planner,
            lambda_filter=self.lambda_filter,
            counterfactuals_complete=counterfactuals_complete,
        )

    def batch_compute_rewards(self, results: Iterable[RolloutResult]) -> List[RewardBreakdown]:
        breakdowns = []
        for result in results:
            breakdown = self.compute_pipeline_rewards(result)
            result.rewards = breakdown
            breakdowns.append(breakdown)
        return breakdowns

    @staticmethod
    def to_reward_tensors(breakdowns: List[RewardBreakdown]) -> Dict[str, object]:
        """Compatibility utility for integrations that expect per-agent tensors."""

        import torch

        return {
            "q": torch.tensor([value.r_planner for value in breakdowns]).unsqueeze(-1),
            "f": torch.tensor([value.r_filter for value in breakdowns]).unsqueeze(-1),
            "g": torch.tensor([value.r_generator for value in breakdowns]).unsqueeze(-1),
        }

    def reset_uar_stats(self) -> None:
        """Deprecated no-op retained for old training scripts."""


EmotionRewardComputer = AffectiveRewardComputer
