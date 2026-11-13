"""Role-specific SFT data for the Query Planner, Evidence Filter and Emotion Generator.

The data is built from the MER-Caption+ training split, which is also the retrieval library
(paper Sec. 4.1).  Every sample is excluded from its own retrieval results.

* Planner targets: support / confusion / counter queries written by an OpenAI-compatible teacher
  that sees the ground truth and the MER-Caption+ description (the student only sees the input).
* Filter targets: one Yes/No per retrieved candidate, derived from emotion-wheel agreement.
  Support items must share an emotion with the ground truth; confusion and counter items must
  show the confusing / opposite emotion named in the Planner target.
* Generator targets: a short rationale taken from the MER-Caption+ description followed by the
  ground-truth open-vocabulary labels, conditioned on the evidence kept by the Filter target.

Usage:
    python -m affectagent.build_sft_data --output affectagent/artifacts/sft_data/sft.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from affectagent.data import load_dataset_cls, read_split  # noqa: E402
from affectagent.labels import build_candidate_labels, get_emotion_wheel, label_f1, split_labels  # noqa: E402
from affectagent.prompts import (  # noqa: E402
    QUERY_LABEL_KEYS,
    QUERY_OUTPUT_SCHEMA,
    format_filter_target,
    format_generator_target,
    parse_query_planner_output,
)
from affectagent.retriever_service import DualChannelRetriever  # noqa: E402
from affectagent.schemas import GROUPS  # noqa: E402


ROLES = ("planner", "filter", "generator")

TEACHER_SYSTEM_PROMPT = (
    "You write retrieval queries for a multimodal emotion recognition system. You are given a video "
    "clip's subtitle, a description of its audiovisual emotional cues, the ground-truth emotion labels, "
    "and a candidate label vocabulary. Write three queries that will be used to search a library of "
    "annotated clips:\n"
    "- support: describe the observable cues (wording, facial expression, gestures, voice) that "
    "characterise the ground-truth emotion in this clip; target_label is the main ground-truth label.\n"
    "- confusion: name the emotion this clip is most easily confused with (it must differ from the "
    "ground truth) and describe the cues that would indicate that emotion; confusing_label is it.\n"
    "- counter: name an emotion that contradicts the ground truth and describe the cues that would "
    "indicate it; opposite_label is it.\n"
    "Describe cues only. Do not copy the description verbatim and do not mention that a label is known. "
    "Every label must be a single English emotion word. Output JSON only:\n" + QUERY_OUTPUT_SCHEMA
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build role-specific AffectAgent SFT data from MER-Caption+")
    parser.add_argument("--dataset", default="mercaptionplus", help="Training library (MER-Caption+)")
    parser.add_argument("--semantic-index-dir", default="affectagent/artifacts/semantic_index")
    parser.add_argument("--output", default="affectagent/artifacts/sft_data/sft.jsonl")
    parser.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    parser.add_argument("--retrieval-top-k", type=int, default=3, help="Must match the K used in training")
    parser.add_argument("--support-threshold", type=float, default=0.0,
                        help="A support candidate is kept if its wheel F1 with the ground truth exceeds this")
    parser.add_argument("--rationale-sentences", type=int, default=3)
    parser.add_argument("--teacher-model", default=os.environ.get("AFFECTAGENT_TEACHER_MODEL", "gpt-4o"))
    parser.add_argument("--api-key", default="", help="Defaults to OPENAI_API_KEY")
    parser.add_argument("--base-url", default="", help="Defaults to OPENAI_BASE_URL")
    parser.add_argument("--num-workers", type=int, default=8, help="Concurrent teacher requests")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--resume", action="store_true", help="Skip samples already present in --output")
    return parser.parse_args()


# ══════════════════════════════════════════════════════
# Teacher (Planner targets)
# ══════════════════════════════════════════════════════

def make_teacher(args):
    from openai import OpenAI

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("A teacher API key is required: set OPENAI_API_KEY or pass --api-key.")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def teacher_queries(client, model: str, record: Dict, candidate_labels: List[str], retries: int = 5) -> Optional[Dict]:
    user = (
        f'Subtitle: "{record["subtitle"]}"\n'
        f"Audiovisual description: {record['description']}\n"
        f"Ground-truth labels: {record['ground_truth']}\n"
        f"Candidate labels: {', '.join(candidate_labels)}"
    )
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": TEACHER_SYSTEM_PROMPT}, {"role": "user", "content": user}],
                temperature=0.7,
            )
            output = parse_query_planner_output(response.choices[0].message.content or "")
            if output.valid:
                return normalise_plan(output, record["ground_truth"])
        except Exception as error:  # network errors, rate limits, malformed responses
            print(f"[teacher] {record['name']}: {error}")
            time.sleep(min(60, 5 * (attempt + 1)))
    return None


def normalise_plan(output, ground_truth: str) -> Dict:
    """Keep the label fields single words and make the support label agree with the ground truth."""

    plan = {group: dict(getattr(output, group)) for group in GROUPS}
    truth = split_labels(ground_truth)
    for group, key in QUERY_LABEL_KEYS.items():
        labels = split_labels(plan[group].get(key, ""))
        plan[group][key] = labels[0] if labels else ""
    if truth and plan["support"]["target_label"] not in truth:
        plan["support"]["target_label"] = truth[0]
    return plan


# ══════════════════════════════════════════════════════
# Filter and Generator targets
# ══════════════════════════════════════════════════════

def filter_decisions(plan: Dict, candidates: Dict[str, list], ground_truth: str, threshold: float) -> Dict[str, List[bool]]:
    wheel = get_emotion_wheel()
    references = {
        "support": ground_truth,
        "confusion": plan["confusion"].get("confusing_label", ""),
        "counter": plan["counter"].get("opposite_label", ""),
    }
    decisions = {}
    for group in GROUPS:
        if group not in candidates:
            continue
        reference = references[group]
        limit = threshold if group == "support" else 0.0
        decisions[group] = [
            bool(reference) and label_f1(item.get("label_hint", ""), reference, wheel) > limit
            for item in candidates[group]
        ]
    return decisions


def rationale_from_description(description: str, max_sentences: int, max_chars: int = 400) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", str(description or "").strip()) if s.strip()]
    rationale = " ".join(sentences[:max_sentences])
    return rationale[:max_chars].rstrip()


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def load_done(path: str) -> set:
    done = set()
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    done.add(json.loads(line)["sample_id"])
    return done


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    dataset_cls = load_dataset_cls(args.dataset)
    candidate_labels = build_candidate_labels(dataset_cls, args.dataset)
    records = [record for record in read_split(dataset_cls, args.dataset, "train") if record["description"]]
    if args.max_samples > 0:
        records = records[:args.max_samples]
    done = load_done(args.output) if args.resume else set()
    records = [record for record in records if record["name"] not in done]
    print(f"[SFT data] {len(records)} samples to process, {len(candidate_labels)} candidate labels")

    retriever = DualChannelRetriever(semantic_index_dir=args.semantic_index_dir)
    client = make_teacher(args)
    get_emotion_wheel()  # fail early if the wheel files are missing

    counts = {role: 0 for role in ROLES}
    mode = "a" if args.resume else "w"
    with open(args.output, mode, encoding="utf-8") as stream, ThreadPoolExecutor(args.num_workers) as pool:
        futures = {
            pool.submit(teacher_queries, client, args.teacher_model, record, candidate_labels): record
            for record in records
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building SFT records"):
            record = futures[future]
            plan = future.result()
            if plan is None:
                continue
            rows = []
            if "planner" in args.roles:
                rows.append({"role": "planner", "target": json.dumps(plan, ensure_ascii=False)})

            # Retrieval runs in the main thread (the E5 encoder is not shared across threads).
            candidates = retriever.retrieve_channel_A(plan, top_k=args.retrieval_top_k, exclude_sample_id=record["name"])
            decisions = filter_decisions(plan, candidates, record["ground_truth"], args.support_threshold)
            if "filter" in args.roles and any(candidates.values()):
                rows.append({"role": "filter", "candidates": candidates, "target": format_filter_target(decisions)})
            if "generator" in args.roles:
                kept_ids = {
                    group: [str(item["id"]) for item, keep in zip(candidates[group], decisions[group]) if keep]
                    for group in decisions
                }
                target = format_generator_target(
                    rationale_from_description(record["description"], args.rationale_sentences),
                    split_labels(record["ground_truth"]),
                )
                rows.append({"role": "generator", "candidates": candidates, "kept_ids": kept_ids, "target": target})

            for row in rows:
                row["sample_id"] = record["name"]
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["role"]] += 1
            stream.flush()

    print(f"[SFT data] Saved to {args.output}: {counts}")


if __name__ == "__main__":
    main()
