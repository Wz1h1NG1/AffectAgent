"""Role-specific SFT warm start of the shared Actor (paper Sec. 3.5, before MAPPO).

The Query Planner, Evidence Filter and Emotion Generator are trained on the records written by
``affectagent.build_sft_data``.  Prompts and multimodal inputs are produced by the same
``AffectAgentPipeline`` code used during MAPPO and evaluation.  The LLM LoRA adapters of
AffectGPT are updated, and RAAF / MB-MoE are trained through the Generator loss.

Outputs (``--output-dir``):
    sft_checkpoint.pth   AffectGPT-format checkpoint; pass it as ``model.ckpt_3`` to MAPPO/eval
    raaf.pth, mb_moe.pth fusion modules; pass the directory as ``--sft-dir``

Usage:
    torchrun --nproc_per_node 2 -m affectagent.train_sft \
        --cfg-path configs/affectagent.yaml \
        --options model.ckpt_3=/path/to/affectgpt_mercaptionplus.pth \
        --sft-data affectagent/artifacts/sft_data/sft.jsonl \
        --output-dir output/affectagent_sft
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from affectagent import runtime  # noqa: E402  (registers AffectGPT components first)
from affectagent.data import configure_dataset, load_dataset_cls, read_sample_data, read_split, resolve_face_or_frame  # noqa: E402
from affectagent.distributed import (  # noqa: E402
    barrier, broadcast_tensors, cleanup, init_distributed, is_main_process, reduce_sums, sync_gradients,
)
from affectagent.labels import build_candidate_labels, get_task_type  # noqa: E402
from affectagent.modeling import (  # noqa: E402
    affectgpt_checkpoint_state, enable_gradient_checkpointing, load_affectgpt, train_only_llm_adapters,
)
from affectagent.orchestrator import evidence_items  # noqa: E402
from affectagent.schemas import RolloutSample  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="AffectAgent role-specific SFT")
    runtime.add_model_args(parser)
    runtime.add_retrieval_args(parser)
    runtime.add_generation_args(parser)
    runtime.add_fusion_args(parser)
    parser.add_argument("--dataset", default="mercaptionplus")
    parser.add_argument("--sft-data", default="affectagent/artifacts/sft_data/sft.jsonl")
    parser.add_argument("--roles", nargs="+", choices=["planner", "filter", "generator"],
                        default=["planner", "filter", "generator"])
    parser.add_argument("--output-dir", default="output/affectagent_sft")
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4, help="Samples per rank and optimizer step")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--fusion-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4, help="Media decoding workers")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    return parser.parse_args()


class SFTSamples(Dataset):
    """One item per training sample: its record, its SFT rows and its decoded media."""

    def __init__(self, dataset_cls, groups):
        self.dataset_cls = dataset_cls
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        record, rows = self.groups[index]
        return record, rows, read_sample_data(self.dataset_cls, record["name"])

    @staticmethod
    def collate_fn(batch):
        return [item for item in batch if item[2] is not None]


def load_groups(args, dataset_cls, rank: int, world_size: int):
    records = {record["name"]: record for record in read_split(dataset_cls, args.dataset, "train")}
    rows = defaultdict(list)
    with open(args.sft_data, "r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["role"] in args.roles and row["sample_id"] in records:
                    rows[row["sample_id"]].append(row)
    groups = [(records[name], rows[name]) for name in sorted(rows)]
    if 0 < args.max_samples < len(groups):
        groups = groups[:args.max_samples]
    # Equal-size shards so every rank performs the same number of optimizer steps.
    per_rank = len(groups) // world_size
    return groups[rank * per_rank:(rank + 1) * per_rank]


def row_loss(pipeline, row, sample, inputs, face_or_frame):
    encoder = pipeline.encoder
    llama_model = pipeline.model.llama_model
    features = inputs.features
    if row["role"] == "planner":
        prompt_ids = pipeline.planner_prompt_ids(sample, inputs.subtitle, face_or_frame)
    elif row["role"] == "filter":
        prompt_ids = pipeline.filter_prompt_ids(sample, inputs.subtitle, row["candidates"], face_or_frame)
    else:
        evidence = evidence_items(row["candidates"], row["kept_ids"])
        prompt_ids = pipeline.generator_prompt_ids(sample, inputs.subtitle, evidence, face_or_frame)
        features, _ = pipeline.run_fusion(features, inputs.perceptual)  # trains RAAF / MB-MoE
    prompt_embeds = encoder.embed(prompt_ids, features, llama_model)
    return encoder.lm_loss(llama_model, prompt_embeds, encoder.tokenize_target(row["target"]))


def save(output_dir, model, raaf, mb_moe, args, tag):
    directory = os.path.join(output_dir, tag)
    os.makedirs(directory, exist_ok=True)
    torch.save({"model": affectgpt_checkpoint_state(model)}, os.path.join(directory, "sft_checkpoint.pth"))
    torch.save(raaf.state_dict(), os.path.join(directory, "raaf.pth"))
    torch.save(mb_moe.state_dict(), os.path.join(directory, "mb_moe.pth"))
    with open(os.path.join(directory, "sft_config.json"), "w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2, ensure_ascii=False)
    print(f"[SFT] Saved {directory}")


def main():
    args = parse_args()
    rank, world_size, device = init_distributed(args.gpu)
    runtime.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = runtime.load_config(args)
    model, chat = load_affectgpt(cfg, device, args.llm_dtype)
    trainable = train_only_llm_adapters(model)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model.llama_model)
    hidden_size = int(model.llama_model.config.hidden_size)
    raaf, mb_moe = runtime.build_fusion_modules(args, hidden_size, device)
    lora_parameters = [p for p in model.parameters() if p.requires_grad]
    fusion_parameters = list(raaf.parameters()) + list(mb_moe.parameters())
    broadcast_tensors(lora_parameters + fusion_parameters)
    if is_main_process():
        print(f"[SFT] Trainable LoRA values: {trainable:,}; fusion values: {sum(p.numel() for p in fusion_parameters):,}")

    # Only the perceptual channel is needed: cognitive candidates are stored in the SFT records.
    args.semantic_index_dir = None
    retriever = runtime.build_retriever(args)
    pipeline = runtime.build_pipeline(args, chat, retriever, raaf, mb_moe)

    face_or_frame = resolve_face_or_frame(cfg)
    dataset_cls = load_dataset_cls(args.dataset)
    configure_dataset(dataset_cls, cfg, face_or_frame)
    candidate_labels = build_candidate_labels(dataset_cls, args.dataset)
    groups = load_groups(args, dataset_cls, rank, world_size)
    loader = DataLoader(
        SFTSamples(dataset_cls, groups), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=SFTSamples.collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.learning_rate},
            {"params": fusion_parameters, "lr": args.fusion_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(args.num_epochs * len(loader), 1)
    scheduler = runtime.warmup_cosine_schedule(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    parameters = lora_parameters + fusion_parameters

    # Every rank reduces the same keys, whatever roles its current batches contained.
    metric_keys = ["seconds"] + [f"{role}_{kind}" for role in args.roles for kind in ("loss", "n")]

    def empty_window():
        return dict.fromkeys(metric_keys, 0.0)

    step = 0
    for epoch in range(args.num_epochs):
        window = empty_window()
        for batch in loader:
            step += 1
            started = time.time()
            model.eval()
            model.llama_model.train()
            raaf.train()
            mb_moe.train()
            optimizer.zero_grad(set_to_none=True)

            jobs = []
            for record, rows, sample_data in batch:
                sample = RolloutSample(
                    sample_id=record["name"], subtitle=record["subtitle"], ground_truth=record["ground_truth"],
                    candidate_labels=candidate_labels, dataset=args.dataset, lang=args.prompt_lang,
                    task_type=get_task_type(args.dataset),
                )
                inputs = pipeline.prepare_inputs(sample, sample_data, face_or_frame)
                jobs.extend((row, sample, inputs) for row in rows)

            global_rows = reduce_sums({"n": len(jobs)}, device)["n"]
            if global_rows == 0:  # every rank skips together (no media could be read)
                continue
            for row, sample, inputs in jobs:
                loss = row_loss(pipeline, row, sample, inputs, face_or_frame)
                (loss / max(global_rows, 1)).backward()
                window[f"{row['role']}_loss"] += float(loss.detach())
                window[f"{row['role']}_n"] += 1

            sync_gradients(parameters)
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            window["seconds"] += time.time() - started

            if step % args.logging_steps == 0:
                totals = reduce_sums(window, device)
                if is_main_process():
                    log = {"epoch": epoch + 1, "step": step, "lr": scheduler.get_last_lr()[0],
                           "seconds_per_step": round(totals["seconds"] / (args.logging_steps * world_size), 2)}
                    for role in args.roles:
                        if totals[f"{role}_n"]:
                            log[f"{role}_loss"] = round(totals[f"{role}_loss"] / totals[f"{role}_n"], 4)
                    print(json.dumps(log))
                    with open(os.path.join(args.output_dir, "sft_log.jsonl"), "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(log) + "\n")
                window = empty_window()

        barrier()
        if is_main_process():
            save(args.output_dir, model, raaf, mb_moe, args, f"epoch_{epoch + 1}")
        barrier()

    if is_main_process():
        save(args.output_dir, model, raaf, mb_moe, args, "final")
    cleanup()


if __name__ == "__main__":
    main()
