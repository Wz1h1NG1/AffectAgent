"""Official AffectAgent MAPPO training (paper Sec. 3.2 and 3.5).

One iteration:
    rollout  Planner -> Retriever -> Filter -> RAAF/MB-MoE -> Generator, plus the Score_label and
             Score_rank counterfactual predictions
    rewards  equations (1)-(4), then the terminal KL-regularised reward of equation (10)
    update   token-level GAE (9) and several clipped Actor / Critic epochs (8), (11)-(12)

The Actor π_θ is the role-specific SFT model (``model.ckpt_3``) with its SFT adapters merged and
a new LoRA adapter on top; π_SFT is the same network with that adapter disabled.  RAAF and MB-MoE
start from the SFT weights in ``--sft-dir`` and are updated through the Generator's PPO loss;
their frozen SFT copies are used when evaluating π_SFT.

Usage (two GPUs):
    torchrun --nproc_per_node 2 -m affectagent.train_ppo \
        --cfg-path configs/affectagent.yaml \
        --options model.ckpt_3=output/affectagent_sft/final/sft_checkpoint.pth \
        --sft-dir output/affectagent_sft/final \
        --output-dir output/affectagent_mappo --gradient-checkpointing
"""

import argparse
import copy
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from affectagent import runtime  # noqa: E402  (registers AffectGPT components first)
from affectagent.data import AffectAgentDataset, configure_dataset, load_dataset_cls, resolve_face_or_frame  # noqa: E402
from affectagent.distributed import (  # noqa: E402
    barrier, broadcast_tensors, cleanup, init_distributed, is_main_process, reduce_sums,
)
from affectagent.labels import build_candidate_labels  # noqa: E402
from affectagent.mappo import AdapterDisabledReference, PolicyGradientUpdater, SequenceValueHead  # noqa: E402
from affectagent.modeling import (  # noqa: E402
    attach_policy_adapter, enable_gradient_checkpointing, load_affectgpt, load_module_state, save_mappo_checkpoint,
)
from affectagent.reward import AffectiveRewardComputer  # noqa: E402


ROLLOUT_KEYS = (
    "n", "score_full", "score_label", "score_rank", "r_planner", "r_filter",
    "planner_valid", "filter_valid", "generator_valid", "evidence_kept", "evidence_retrieved",
)
UPDATE_KEYS = ("policy_loss", "value_loss", "mean_kl", "clip_frac", "mean_sequence_kl", "mean_regularized_reward")


def parse_args():
    parser = argparse.ArgumentParser(description="AffectAgent MAPPO training")
    runtime.add_model_args(parser)
    runtime.add_retrieval_args(parser)
    runtime.add_generation_args(parser)
    runtime.add_fusion_args(parser)
    parser.add_argument("--dataset", default="mercaptionplus",
                        help="MER-Caption+ training set, which is also the retrieval library")
    parser.add_argument("--sft-dir", default="", help="Directory with the SFT raaf.pth / mb_moe.pth")
    parser.add_argument("--output-dir", default="output/affectagent_mappo")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--rollout-batch-size", "--batch-size", dest="rollout_batch_size", type=int, default=4,
                        help="Samples per rank per update")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Actor (RL adapter) learning rate")
    parser.add_argument("--fusion-learning-rate", type=float, default=2e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=2e-5)
    parser.add_argument("--kl-coef", type=float, default=0.05, help="β in equation (10)")
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2, help="ε in equations (8) and (11)")
    parser.add_argument("--value-coef", type=float, default=0.5, help="α in L(θ, φ) = L_Actor + α L_Critic")
    parser.add_argument("--gamma", type=float, default=1.0, help="γ in equation (9); episodes end at T_i")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="λ in equation (9)")
    parser.add_argument("--lambda-planner", type=float, default=1.0, help="λ_P in equation (3)")
    parser.add_argument("--lambda-filter", type=float, default=1.0, help="λ_F in equation (4)")
    parser.add_argument("--no-emotion-wheel-reward", action="store_true",
                        help="Use surface-form label F1 instead of emotion-wheel-normalised F1")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4, help="Media decoding workers")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    # Rollouts sample from softmax(logits / T) without nucleus truncation, so the PPO ratio and
    # the KL term are computed on exactly the behaviour distribution.
    parser.set_defaults(top_p=1.0)
    return parser.parse_args()


def rollout_stats(results, rewards) -> dict:
    stats = dict.fromkeys(ROLLOUT_KEYS, 0.0)
    for result, reward in zip(results, rewards):
        stats["n"] += 1
        stats["score_full"] += reward.score_full
        stats["score_label"] += reward.score_label
        stats["score_rank"] += reward.score_rank
        stats["r_planner"] += reward.r_planner
        stats["r_filter"] += reward.r_filter
        stats["planner_valid"] += float(bool(result.query_output and result.query_output.valid))
        stats["filter_valid"] += float(bool(result.filter_output and result.filter_output.valid))
        stats["generator_valid"] += float(bool(result.generator_output and result.generator_output.valid))
        stats["evidence_kept"] += sum(len(items) for items in result.selected_evidence.values())
        stats["evidence_retrieved"] += sum(len(items) for items in result.candidates.values())
    return stats


def empty_window() -> dict:
    return dict.fromkeys(ROLLOUT_KEYS + UPDATE_KEYS + ("trajectories", "seconds"), 0.0)


def print_diagnostics(step, results, rewards) -> None:
    for result, reward in list(zip(results, rewards))[:2]:
        query = result.query_output
        print(json.dumps({
            "step": step,
            "sample": result.sample.sample_id,
            "ground_truth": result.sample.ground_truth,
            "support_query": query.support.get("query_text", "") if query else "",
            "filter_decisions": result.filter_output.decisions if result.filter_output else {},
            "prediction": result.generator_output.prediction,
            "generator_raw": result.generator_output.raw_text[:300],
            "scores": [reward.score_full, reward.score_label, reward.score_rank],
        }, ensure_ascii=False))


def main():
    args = parse_args()
    rank, world_size, device = init_distributed(args.gpu)
    runtime.set_seed(args.seed + rank)  # rank-specific sampling; shared weights are broadcast below
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as stream:
            json.dump(vars(args), stream, indent=2, ensure_ascii=False)

    # ── 1. Actor π_θ, reference π_SFT, RAAF / MB-MoE and Critic V_φ ──
    cfg = runtime.load_config(args)
    sft_checkpoint = cfg.model_cfg.get("ckpt_3", "")
    if not sft_checkpoint:
        raise SystemExit("Pass the role-specific SFT checkpoint with --options model.ckpt_3=...")
    model, chat = load_affectgpt(cfg, device, args.llm_dtype)
    attach_policy_adapter(model, args.lora_r, args.lora_alpha)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model.llama_model)
    reference = AdapterDisabledReference(model.llama_model)
    hidden_size = int(model.llama_model.config.hidden_size)

    runtime.apply_saved_fusion_config(args, args.sft_dir)
    raaf, mb_moe = runtime.build_fusion_modules(args, hidden_size, device)
    loaded = [
        load_module_state(module, os.path.join(args.sft_dir, filename), device) if args.sft_dir else False
        for module, filename in ((raaf, "raaf.pth"), (mb_moe, "mb_moe.pth"))
    ]
    if is_main_process() and not all(loaded):
        print("[MAPPO] SFT fusion weights not found; RAAF / MB-MoE start from the identity initialisation.")
    critic = SequenceValueHead(hidden_size).to(device)
    actor_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    fusion_parameters = list(raaf.parameters()) + list(mb_moe.parameters())
    critic_parameters = list(critic.parameters())
    broadcast_tensors(actor_parameters + fusion_parameters + critic_parameters)
    reference_fusion = tuple(copy.deepcopy(module).eval().requires_grad_(False) for module in (raaf, mb_moe))
    if is_main_process():
        print(f"[MAPPO] Actor adapter values: {sum(p.numel() for p in actor_parameters):,}")

    # ── 2. Frozen retriever and the MER-Caption+ training split ──
    retriever = runtime.build_retriever(args)
    face_or_frame = resolve_face_or_frame(cfg)
    dataset_cls = load_dataset_cls(args.dataset)
    configure_dataset(dataset_cls, cfg, face_or_frame)
    candidate_labels = build_candidate_labels(dataset_cls, args.dataset)
    train_set = AffectAgentDataset(
        dataset_cls, args.dataset, candidate_labels, split="train", lang=args.prompt_lang, max_samples=args.max_samples,
    )
    if world_size > 1:
        sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    else:
        sampler = RandomSampler(train_set)
    loader = DataLoader(
        train_set, batch_size=args.rollout_batch_size, sampler=sampler, num_workers=args.num_workers,
        collate_fn=AffectAgentDataset.collate_fn, drop_last=True,
    )

    # ── 3. Pipeline, rewards and the joint optimiser ──
    pipeline = runtime.build_pipeline(args, chat, retriever, raaf, mb_moe, do_sample=True)
    reward_computer = AffectiveRewardComputer(
        lambda_planner=args.lambda_planner,
        lambda_filter=args.lambda_filter,
        use_emotion_wheel=not args.no_emotion_wheel_reward,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": actor_parameters, "lr": args.learning_rate},
            {"params": fusion_parameters, "lr": args.fusion_learning_rate},
            {"params": critic_parameters, "lr": args.critic_learning_rate},
        ],
        weight_decay=0.0,
    )
    total_updates = max(args.num_epochs * len(loader) * args.ppo_epochs, 1)
    scheduler = runtime.warmup_cosine_schedule(optimizer, int(args.warmup_ratio * total_updates), total_updates)
    updater = PolicyGradientUpdater(
        model=model,
        reference_model=reference,
        value_head=critic,
        optimizer=optimizer,
        kl_coef=args.kl_coef,
        ppo_epochs=args.ppo_epochs,
        clip_range=args.ppo_clip_range,
        value_coef=args.value_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        device=device,
        fusion_modules=(raaf, mb_moe),
        reference_fusion=reference_fusion,
    )
    manifest = {
        "sft_checkpoint": str(sft_checkpoint),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "fusion": {key: getattr(args, key) for key in runtime.FUSION_KEYS},
        "prompt_lang": args.prompt_lang,
        "retrieval_top_k": args.retrieval_top_k,
        "channel_b_top_k": args.channel_b_top_k,
    }

    def save(tag: str, step: int) -> None:
        barrier()
        if is_main_process():
            directory = os.path.join(args.output_dir, tag)
            save_mappo_checkpoint(directory, model, raaf, mb_moe, critic, dict(manifest, step=step))
            print(f"[MAPPO] Saved {directory}")
        barrier()

    if is_main_process():
        print(f"[MAPPO] {len(loader)} updates per epoch on {world_size} rank(s), "
              f"{args.rollout_batch_size} samples per rank, {total_updates} optimizer steps in total")

    # ── 4. Rollout -> reward -> MAPPO update ──
    step = 0
    window = empty_window()
    log_path = os.path.join(args.output_dir, "training_log.jsonl")
    for epoch in range(args.num_epochs):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for samples, sample_data in loader:
            step += 1
            started = time.time()
            model.eval()
            raaf.eval()
            mb_moe.eval()

            results = pipeline.batch_rollout(samples, sample_data, face_or_frame)
            rewards = reward_computer.batch_compute_rewards(results)
            trajectories = pipeline.collect_ppo_trajectories(results)
            # Every rank calls update_step, even with an empty batch (collective operations inside).
            update = updater.update_step(trajectories, pipeline)
            for _ in range(update["update_rounds"]):
                scheduler.step()
            if step <= 3 and is_main_process():
                print_diagnostics(step, results, rewards)
            pipeline.release_context(results)

            for key, value in rollout_stats(results, rewards).items():
                window[key] += value
            for key in UPDATE_KEYS:
                window[key] += update[key] * update["num_trajectories"]
            window["trajectories"] += update["num_trajectories"]
            window["seconds"] += time.time() - started

            if step % args.logging_steps == 0:
                totals = reduce_sums(window, device)
                window = empty_window()
                if is_main_process():
                    n, t = max(totals["n"], 1.0), max(totals["trajectories"], 1.0)
                    log = {"epoch": epoch + 1, "step": step, "samples": int(totals["n"]),
                           "lr": scheduler.get_last_lr()[0],
                           "seconds_per_step": round(totals["seconds"] / (args.logging_steps * world_size), 2)}
                    log.update({key: round(totals[key] / n, 4) for key in ROLLOUT_KEYS if key != "n"})
                    log.update({key: round(totals[key] / t, 5) for key in UPDATE_KEYS})
                    print(json.dumps(log))
                    with open(log_path, "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(log) + "\n")

            if step % args.save_steps == 0:
                save(f"checkpoint-{step}", step)

    save("final", step)
    cleanup()


if __name__ == "__main__":
    main()
