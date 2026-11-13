"""Official AffectAgent evaluation: MER-UniBench, paper ablations and missing-modality settings.

``--ckpt-dir`` may point to a MAPPO checkpoint (``checkpoint_manifest.json`` + ``actor_adapter.pth``)
or to an SFT output directory (``raaf.pth`` / ``mb_moe.pth`` only).  ``model.ckpt_3`` must be the
role-specific SFT checkpoint used to start MAPPO.

HIT (basic emotion) and F_s (fine-grained) are computed here.  WAF (sentiment) needs the
Qwen2.5 openset-to-sentiment mapping served by vLLM; run ``python -m affectagent.unibench`` on
the written ``predictions.jsonl`` after this script has released the GPU.

Usage:
    torchrun --nproc_per_node 2 -m affectagent.evaluate \
        --cfg-path configs/affectagent.yaml \
        --options model.ckpt_3=output/affectagent_sft/final/sft_checkpoint.pth \
        --ckpt-dir output/affectagent_mappo/final \
        --datasets mer2023 mer2024 meld iemocap cmumosi cmumosei sims simsv2 ovmerdplus \
        --variant full --output-dir output/eval/full
"""

import argparse
import glob
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from affectagent import runtime  # noqa: E402  (registers AffectGPT components first)
from affectagent.checkpointing import resolve_checkpoint_path  # noqa: E402
from affectagent.data import AffectAgentDataset, canonical_dataset_name, configure_dataset, load_dataset_cls, resolve_face_or_frame  # noqa: E402
from affectagent.distributed import barrier, cleanup, init_distributed, is_main_process  # noqa: E402
from affectagent.labels import build_candidate_labels, get_task_type  # noqa: E402
from affectagent.modeling import load_affectgpt, load_module_state, load_policy_adapter, read_manifest  # noqa: E402
from affectagent.orchestrator import MODALITIES  # noqa: E402
from affectagent.reward import AffectiveRewardComputer  # noqa: E402
from affectagent.unibench import PRIMARY_METRIC, score_records  # noqa: E402


# Paper tables "Stepwise effectiveness" and "Multi-Agent coordination and Evidence Structures".
# The Planner and Filter ablations are the reward counterfactuals of Sec. 3.2: w/o Planner replaces
# every Planner query by its simple emotion label (Score_label; confusion and counter evidence are
# still retrieved), w/o Filter passes the Top-K items directly (Score_rank).  Removing both is the
# naive retrieval baseline.  Every variant keeps support (primary) evidence.
VARIANTS = {
    "full": {},
    "no_planner": {"planner_mode": "label"},
    "no_filter": {"enable_filter": False},
    "naive_rag": {"planner_mode": "label", "enable_filter": False},
    "input_rag": {"planner_mode": "none", "enable_filter": False},  # retrieval with the raw transcript
    "no_retrieval": {"enable_retrieval": False},                    # SFT-style inference without RAG
    "support_only": {"evidence_groups": ("support",)},              # w/o confusion and counter evidence
    "no_counter": {"evidence_groups": ("support", "confusion")},
    "no_confusion": {"evidence_groups": ("support", "counter")},
    "no_raaf": {"enable_raaf": False},
    "no_mb_moe": {"enable_mb_moe": False},
}
def parse_args():
    parser = argparse.ArgumentParser(description="AffectAgent evaluation")
    runtime.add_model_args(parser)
    runtime.add_retrieval_args(parser)
    runtime.add_generation_args(parser)
    runtime.add_fusion_args(parser)
    parser.add_argument("--ckpt-dir", default="", help="MAPPO checkpoint or SFT output directory")
    parser.add_argument("--sft-dir", default="", help="Fallback directory for raaf.pth / mb_moe.pth")
    parser.add_argument("--datasets", "--dataset", dest="datasets", nargs="+", default=["mer2023"])
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    parser.add_argument("--missing-modalities", nargs="*", choices=MODALITIES, default=[],
                        help="Remove modalities from the input; they are supplemented by retrieval")
    parser.add_argument("--candidate-labels", choices=["wheel", "dataset"], default="wheel",
                        help="Candidate set C: emotion-wheel vocabulary (training default) or the dataset's labels")
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--output-dir", default="output/eval")
    return parser.parse_args()


def first_item(batch):
    return batch[0]


def load_policy(args, cfg, device):
    model, chat = load_affectgpt(cfg, device, args.llm_dtype)
    manifest = read_manifest(args.ckpt_dir) if args.ckpt_dir else None
    if manifest is not None:
        expected = manifest.get("sft_checkpoint", "")
        actual = str(cfg.model_cfg.get("ckpt_3", ""))
        if expected and os.path.abspath(expected) != os.path.abspath(actual):
            print(f"[Eval] WARNING: MAPPO was started from {expected}, but model.ckpt_3={actual}")
        load_policy_adapter(model, args.ckpt_dir)
        for key, value in manifest.get("fusion", {}).items():
            setattr(args, key, value)
        # Settings the policy was trained with; evaluation must use the same ones.
        for key in ("prompt_lang", "retrieval_top_k", "channel_b_top_k"):
            if key in manifest and getattr(args, key) != manifest[key]:
                print(f"[Eval] Using {key}={manifest[key]} from the checkpoint (was {getattr(args, key)})")
                setattr(args, key, manifest[key])
        print(f"[Eval] Loaded MAPPO policy adapter from {args.ckpt_dir}")
    elif args.ckpt_dir:
        print(f"[Eval] No MAPPO manifest in {args.ckpt_dir}; evaluating the SFT policy.")
        runtime.apply_saved_fusion_config(args, args.ckpt_dir)

    hidden_size = int(model.llama_model.config.hidden_size)
    raaf, mb_moe = runtime.build_fusion_modules(args, hidden_size, device)
    for module, component in ((raaf, "raaf"), (mb_moe, "mb_moe")):
        path = None
        for directory in (args.ckpt_dir, args.sft_dir):
            path = path or (resolve_checkpoint_path(directory, component) if directory else None)
        if load_module_state(module, path, device):
            print(f"[Eval] Loaded {component} from {path}")
        else:
            print(f"[Eval] WARNING: no {component} weights found; using the identity initialisation.")
    raaf.eval()
    mb_moe.eval()
    model.eval()
    return model, chat, raaf, mb_moe


def evaluate_dataset(args, cfg, pipeline, dataset_name, rank, world_size, output_dir):
    face_or_frame = resolve_face_or_frame(cfg)
    dataset_cls = load_dataset_cls(dataset_name)
    configure_dataset(dataset_cls, cfg, face_or_frame)
    candidate_labels = build_candidate_labels(dataset_cls, dataset_name, source=args.candidate_labels)
    dataset = AffectAgentDataset(
        dataset_cls, dataset_name, candidate_labels, split="test", lang=args.prompt_lang, max_samples=args.max_samples,
    )
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=args.num_workers, collate_fn=first_item)

    shard_path = os.path.join(output_dir, f"predictions.rank{rank}.jsonl")
    with open(shard_path, "w", encoding="utf-8") as stream, torch.no_grad():
        for position, (sample, sample_data) in enumerate(loader):
            record = dataset.by_name[sample.sample_id]
            result = pipeline.full_rollout(sample, sample_data, face_or_frame)
            output = result.generator_output
            diagnostics = result.fusion_diagnostics
            row = {
                "dataset": dataset.dataset_name,
                "variant": args.variant,
                "missing_modalities": list(args.missing_modalities),
                "sample_id": sample.sample_id,
                "ground_truth": sample.ground_truth,
                "valence": record.get("valence"),
                "labels": list(output.labels) if output else [],
                "prediction": output.prediction if output else "",
                "reasoning": output.reasoning if output else "",
                "media_loaded": sample_data is not None,
                "planner_valid": bool(result.query_output and result.query_output.valid),
                "filter_valid": bool(result.filter_output and result.filter_output.valid),
                "generator_valid": bool(output and output.valid),
                "evidence_retrieved": sum(len(items) for items in result.candidates.values()),
                "evidence_kept": sum(len(items) for items in result.selected_evidence.values()),
                "fusion_partner_id": diagnostics.fusion_partner_id if diagnostics else "",
                "supplemented_modalities": diagnostics.supplemented_modalities if diagnostics else [],
                "video_gate_mean": diagnostics.video_gate_mean if diagnostics else 0.0,
                "audio_gate_mean": diagnostics.audio_gate_mean if diagnostics else 0.0,
                "moe_experts_activated": diagnostics.moe_experts_activated if diagnostics else [],
                "moe_expert_weights": diagnostics.moe_expert_weights if diagnostics else [],
                "generator_raw": output.raw_text if output else "",
            }
            if get_task_type(dataset.dataset_name) != "sentiment":
                row["sample_f1"] = AffectiveRewardComputer.compute_f1_score(row["prediction"], row["ground_truth"], True)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (position + 1) % 50 == 0:
                print(f"[Eval] rank {rank} {dataset.dataset_name}: {position + 1}/{len(indices)}")

    barrier()
    if is_main_process():
        records = []
        for path in sorted(glob.glob(os.path.join(output_dir, "predictions.rank*.jsonl"))):
            with open(path, "r", encoding="utf-8") as shard:
                records.extend(json.loads(line) for line in shard if line.strip())
            os.remove(path)
        records.sort(key=lambda item: item["sample_id"])
        with open(os.path.join(output_dir, "predictions.jsonl"), "w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = score_records(records)
        metrics.update({
            "n": len(records),
            "media_loaded": sum(row["media_loaded"] for row in records) / max(len(records), 1),
            "planner_valid": sum(row["planner_valid"] for row in records) / max(len(records), 1),
            "generator_valid": sum(row["generator_valid"] for row in records) / max(len(records), 1),
            "mean_evidence_kept": sum(row["evidence_kept"] for row in records) / max(len(records), 1),
        })
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2)
        primary = PRIMARY_METRIC[get_task_type(dataset.dataset_name)]
        if primary in metrics:
            print(f"[Eval] {dataset.dataset_name} ({args.variant}): {primary} = {metrics[primary] * 100:.2f}")
        else:
            print(f"[Eval] {dataset.dataset_name}: run `python -m affectagent.unibench --predictions "
                  f"{os.path.join(output_dir, 'predictions.jsonl')}` to compute {primary}.")
    barrier()


def main():
    args = parse_args()
    rank, world_size, device = init_distributed(args.gpu)
    runtime.set_seed(args.seed + rank)
    cfg = runtime.load_config(args)

    model, chat, raaf, mb_moe = load_policy(args, cfg, device)
    retriever = runtime.build_retriever(args)
    pipeline = runtime.build_pipeline(
        args, chat, retriever, raaf, mb_moe,
        do_sample=args.sample,
        compute_counterfactual_rewards=False,
        missing_modalities=args.missing_modalities,
        **VARIANTS[args.variant],
    )

    for dataset_name in args.datasets:
        dataset_name = canonical_dataset_name(dataset_name)
        output_dir = os.path.join(args.output_dir, dataset_name)
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "eval_config.json"), "w", encoding="utf-8") as stream:
                json.dump(vars(args), stream, indent=2, ensure_ascii=False)
        barrier()
        evaluate_dataset(args, cfg, pipeline, dataset_name, rank, world_size, output_dir)
    cleanup()


if __name__ == "__main__":
    main()
