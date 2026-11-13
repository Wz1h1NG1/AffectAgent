"""Emotion label vocabulary, emotion-wheel normalisation and label-set F1.

MER-UniBench scores open-vocabulary predictions after mapping every word to an emotion-wheel
cluster (AffectGPT ``my_affectgpt/evaluation/wheel.py``).  The shared reward uses the same
mapping, so semantically equivalent labels ("joyful" / "happy") are rewarded exactly as they
are scored at test time.
"""

import glob
import os
import re
from typing import Iterable, List, Optional, Sequence, Set

import config


# MER-UniBench task families (paper Sec. 4.1).  The Generator always answers with
# open-vocabulary emotion words; the family only selects the metric (HIT / WAF / F_s).
DATASET_TASK_TYPE = {
    "mercaptionplus": "ovlabel",
    "ovmerd": "ovlabel",
    "ovmerdplus": "ovlabel",
    "mer2023": "discrete",
    "mer2024": "discrete",
    "meld": "discrete",
    "iemocap": "discrete",
    "iemocapfour": "discrete",
    "cmumosi": "sentiment",
    "cmumosei": "sentiment",
    "sims": "sentiment",
    "simsv2": "sentiment",
}

WHEEL_METRICS = tuple(f"case3_wheel{index}_level1" for index in range(1, 6))


def get_task_type(dataset_name: str) -> str:
    try:
        return DATASET_TASK_TYPE[dataset_name.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown dataset: {dataset_name}") from error


def normalize_label(label: object) -> str:
    value = str(label or "").strip().lower()
    value = re.sub(r"[\[\]{}\"'`]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,.;:，；、/|")


def split_labels(text: object) -> List[str]:
    """Split a label string or list into unique normalised labels, preserving order."""

    if text is None:
        return []
    if isinstance(text, (list, tuple, set)):
        parts = [str(item) for item in text]
    else:
        parts = re.split(r"[,，;/；、|\n]+", str(text))
    labels = []
    for part in parts:
        label = normalize_label(part)
        if label and label != "nan" and label not in labels:
            labels.append(label)
    return labels


def candidate_label_vocabulary(levels: Sequence[str] = ("level1", "level2")) -> List[str]:
    """Emotion-wheel vocabulary plus ``neutral`` (sorted, identical in training and testing)."""

    import pandas as pd

    vocabulary: Set[str] = {"neutral"}
    paths = sorted(glob.glob(os.path.join(config.EMOTION_WHEEL_ROOT, "wheel*.xlsx")))
    if not paths:
        raise FileNotFoundError(
            f"No emotion wheel found under {config.EMOTION_WHEEL_ROOT}; copy AffectGPT's emotion_wheel/."
        )
    for xlsx_path in paths:
        frame = pd.read_excel(xlsx_path).ffill()
        for level in levels:
            vocabulary.update(label for label in map(normalize_label, frame[level].tolist()) if label)
    return sorted(vocabulary)


def build_candidate_labels(dataset_cls, dataset_name: str, source: str = "wheel") -> List[str]:
    """Candidate label set C observed by the Planner and the Generator.

    ``wheel`` (default): the emotion-wheel vocabulary plus ``neutral``, identical for every dataset
    in training and testing.  MER-UniBench scores zero-shot open-vocabulary predictions (AffectGPT
    asks for "all possible emotional states" and maps the answer to wheel clusters), so a fixed
    vocabulary avoids a train/test mismatch and covers every ground-truth word after synonym
    mapping.
    ``dataset``: the closed label set of a discrete benchmark (MER2023/2024, MELD, IEMOCAP).

    C is a reference vocabulary: predictions are not restricted to it.
    """

    if source == "dataset":
        if get_task_type(dataset_name) != "discrete":
            raise ValueError(f"{dataset_name} has no closed label set; use source='wheel'")
        labels = split_labels(getattr(dataset_cls, "candidate_labels", ""))
        if not labels:
            raise ValueError(f"{dataset_cls.__class__.__name__} does not define candidate_labels")
        return sorted(labels)
    if source != "wheel":
        raise ValueError(f"Unknown candidate label source: {source}")
    return candidate_label_vocabulary()


class EmotionWheel:
    """Thin wrapper around the AffectGPT / MER-UniBench emotion-wheel mapping."""

    def __init__(self):
        from my_affectgpt.evaluation import wheel

        self._wheel = wheel
        self.format_mapping = wheel.format_mapping
        self.raw_mapping = wheel.raw_mapping
        self.wheel_maps = {
            metric: wheel.func_get_wheel_cluster(*metric.split("_")[1:])
            for metric in WHEEL_METRICS
        }

    def map_labels(self, labels: Iterable[str], metric: str) -> Set[str]:
        """Map labels to wheel clusters.  Words outside the wheel are dropped, as in
        MER-UniBench, except ``neutral``: no wheel contains it, but it is a valid answer."""

        labels = list(labels)
        mapped = set(self._wheel.func_map_label_to_synonym(
            labels, self.format_mapping, self.raw_mapping, self.wheel_maps[metric], metric
        ))
        if "neutral" in labels:
            mapped.add("neutral")
        return mapped


_WHEEL: Optional[EmotionWheel] = None


def get_emotion_wheel() -> EmotionWheel:
    """Load the emotion wheel once (requires ``emotion_wheel/`` and ``openpyxl``)."""

    global _WHEEL
    if _WHEEL is None:
        _WHEEL = EmotionWheel()
    return _WHEEL


def _set_f1(predicted: Set[str], reference: Set[str]) -> float:
    if not predicted or not reference:
        return 0.0
    true_positive = len(predicted & reference)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(predicted)
    recall = true_positive / len(reference)
    return 2.0 * precision * recall / (precision + recall)


def label_f1(prediction: object, ground_truth: object, wheel: Optional[EmotionWheel] = None) -> float:
    """Per-sample label-set F1 used as Score_* in equation (1).

    With an emotion wheel the F1 is averaged over the five MER-UniBench wheels (the F_s
    protocol); without one, normalised surface forms are compared.
    """

    predicted = split_labels(prediction)
    reference = split_labels(ground_truth)
    if not predicted or not reference:
        return 0.0
    if wheel is None:
        return _set_f1(set(predicted), set(reference))
    scores = []
    for metric in WHEEL_METRICS:
        mapped_reference = wheel.map_labels(reference, metric)
        if mapped_reference:
            scores.append(_set_f1(wheel.map_labels(predicted, metric), mapped_reference))
        else:
            # Ground truth entirely outside the wheel: compare surface forms instead.
            scores.append(_set_f1(set(predicted), set(reference)))
    return sum(scores) / len(scores)
