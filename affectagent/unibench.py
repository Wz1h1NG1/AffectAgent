"""MER-UniBench metrics (paper Sec. 4.1) following AffectGPT's evaluation protocol.

* HIT (MER2023, MER2024, MELD, IEMOCAP): the ground-truth label and the predicted emotion words
  are mapped to emotion-wheel clusters; a sample is a hit when they share a cluster.
* WAF (CMU-MOSI, CMU-MOSEI, SIMS, SIMS v2): the predicted words are mapped to a sentiment by
  Qwen2.5 (``toolkit.utils.qwen.openset_to_sentiment_qwen``, served with vLLM) and the weighted
  F1 of positive vs. non-positive is computed on samples with non-zero valence.
* F_s (OV-MERD+): precision / recall of the wheel-mapped label sets, F = 2PR / (P + R).

All three are averaged over the five emotion wheels.  AffectGPT first extracts emotion words
from a free-form answer with an LLM; the Emotion Generator already outputs the word list, so
that extraction step is not needed.

Usage (scores every predictions file; sentiment datasets need vLLM):
    python -m affectagent.unibench --predictions output/eval/*/predictions.jsonl
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from affectagent.labels import WHEEL_METRICS, get_task_type  # noqa: E402

PRIMARY_METRIC = {"discrete": "HIT", "sentiment": "WAF", "ovlabel": "F_s"}


def _wheel():
    from my_affectgpt.evaluation import wheel

    return wheel


def hit_rate(name2gt: Dict[str, str], name2pred: Dict[str, List[str]]) -> float:
    wheel = _wheel()
    scores = []
    for metric in WHEEL_METRICS:
        hit, _ = wheel.calculate_openset_onehot_hitrate(
            name2gt=name2gt, name2pred=name2pred, metric=metric,
            format_mapping=wheel.format_mapping, raw_mapping=wheel.raw_mapping, inter_print=False,
        )
        scores.append(hit)
    return float(np.mean(scores))


def fine_grained_scores(name2gt: Dict[str, str], name2pred: Dict[str, List[str]]) -> Dict[str, float]:
    wheel = _wheel()
    scores = []
    for metric in WHEEL_METRICS:
        precision, recall = wheel.calculate_openset_overlap_rate(
            name2gt=name2gt, name2pred=name2pred, metric=metric,
            format_mapping=wheel.format_mapping, raw_mapping=wheel.raw_mapping, inter_print=False,
        )
        fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        scores.append([fscore, precision, recall])
    fscore, precision, recall = np.mean(scores, axis=0).tolist()
    return {"F_s": fscore, "precision": precision, "recall": recall}


def sentiment_scores(valences: List[float], polarities: List[str]) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    labels = np.asarray(valences, dtype=np.float32)
    preds = np.asarray([1 if p == "positive" else -1 if p == "negative" else 0 for p in polarities])
    non_zero = labels != 0
    truth, guess = labels[non_zero] > 0, preds[non_zero] > 0
    return {"WAF": float(f1_score(truth, guess, average="weighted")), "accuracy": float(accuracy_score(truth, guess))}


class QwenSentimentMapper:
    """Official openset -> sentiment mapping (vLLM-served Qwen2.5 with AffectGPT's prompt)."""

    def __init__(self, model_name: str = "Qwen25"):
        import config
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        path = config.PATH_TO_LLM[model_name]
        self.llm = LLM(model=path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    def __call__(self, label_lists: List[List[str]], batch_size: int = 64) -> List[str]:
        from toolkit.utils.qwen import openset_to_sentiment_qwen

        polarities = []
        for start in range(0, len(label_lists), batch_size):
            batch = ["[" + ", ".join(labels) + "]" for labels in label_lists[start:start + batch_size]]
            responses = openset_to_sentiment_qwen(
                llm=self.llm, tokenizer=self.tokenizer, sampling_params=self.sampling_params, batch_reasons=batch,
            )
            polarities.extend(str(response).strip().lower() for response in responses)
        return polarities


def score_records(records: List[Dict], sentiment_mapper=None) -> Dict[str, float]:
    """Score prediction records of one dataset (see ``affectagent.evaluate``)."""

    if not records:
        return {}
    task_type = get_task_type(records[0]["dataset"])
    name2pred = {record["sample_id"]: list(record.get("labels", [])) for record in records}
    if task_type == "discrete":
        return {"HIT": hit_rate({r["sample_id"]: r["ground_truth"] for r in records}, name2pred)}
    if task_type == "ovlabel":
        return fine_grained_scores({r["sample_id"]: r["ground_truth"] for r in records}, name2pred)
    if sentiment_mapper is None:
        return {}
    polarities = sentiment_mapper([name2pred[r["sample_id"]] for r in records])
    return sentiment_scores([r["valence"] for r in records], polarities)


def read_predictions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Score AffectAgent predictions with MER-UniBench metrics")
    parser.add_argument("--predictions", nargs="+", required=True, help="predictions.jsonl files (globs allowed)")
    parser.add_argument("--sentiment-llm", default="Qwen25", help="config.PATH_TO_LLM key for the WAF mapping")
    parser.add_argument("--output", default="", help="Optional JSON file for the summary")
    args = parser.parse_args()

    paths = sorted({path for pattern in args.predictions for path in glob.glob(pattern)})
    files = [(path, read_predictions(path)) for path in paths]
    needs_mapper = any(records and get_task_type(records[0]["dataset"]) == "sentiment" for _, records in files)
    mapper = QwenSentimentMapper(args.sentiment_llm) if needs_mapper else None

    summary = {}
    for path, records in files:
        if not records:
            continue
        scores = score_records(records, mapper)
        dataset = records[0]["dataset"]
        key = f"{dataset}:{records[0].get('variant', '')}:{path}"
        summary[key] = scores
        primary = PRIMARY_METRIC[get_task_type(dataset)]
        print(f"{dataset:>12} {primary}={scores[primary] * 100:.2f}  {path}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)


if __name__ == "__main__":
    main()
