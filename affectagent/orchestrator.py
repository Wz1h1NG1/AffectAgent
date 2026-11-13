"""AffectAgent rollout orchestration and paper-defined counterfactuals."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .multimodal import MultimodalPromptEncoder
from .prompts import (
    QUERY_LABEL_KEYS,
    build_emotion_generator_messages,
    build_evidence_filter_messages,
    build_query_planner_messages,
    parse_emotion_generator_output,
    parse_evidence_filter_output,
    parse_query_planner_output,
)
from .schemas import (
    GROUPS,
    EmotionGeneratorOutput,
    EvidenceFilterOutput,
    EvidenceItem,
    FusionDiagnostics,
    QueryPlannerOutput,
    RolloutResult,
    RolloutSample,
)


MODALITIES = ("text", "video", "audio")
# full:  the Planner's support / confusion / counter queries;
# label: every query replaced by the Planner's simple emotion label (the Score_label setting,
#        also the w/o-Planner ablation);
# none:  no Planner, the raw transcript is the only (support) query.
PLANNER_MODES = ("full", "label", "none")


@dataclass
class PreparedInputs:
    """Raw audiovisual tokens, perceptual evidence E_perc and the visible transcript."""

    features: Dict[str, Optional[torch.Tensor]]
    perceptual: Dict[str, Optional[torch.Tensor]]
    partners: List[Dict] = field(default_factory=list)
    subtitle: str = ""
    supplemented: List[str] = field(default_factory=list)


def evidence_items(candidates: Dict[str, list], kept_ids: Dict[str, List[str]]) -> Dict[str, List[EvidenceItem]]:
    """Materialise the kept candidates of every group, in retrieval order."""

    evidence = {}
    for group, items in candidates.items():
        kept = set(kept_ids.get(group, []))
        evidence[group] = [
            EvidenceItem(
                evidence_id=str(item.get("id", "")),
                global_id=item.get("global_id"),
                text=item.get("text", ""),
                role=group,
                subquery_type=group,
                label_hint=item.get("label_hint", ""),
            )
            for item in items
            if str(item.get("id", "")) in kept
        ]
    return evidence


class AffectAgentPipeline:
    """
    Official AffectAgent pipeline. All three agents share one multimodal LLM (the Actor).

    Planner -> dual-channel Retriever -> Evidence Filter -> RAAF -> MB-MoE -> Generator.
    """

    def __init__(
        self,
        chat,
        retriever,
        raaf=None,
        mb_moe=None,
        max_new_tokens_q: int = 256,
        max_new_tokens_s: int = 128,
        max_new_tokens_g: int = 256,
        retrieval_top_k: int = 3,
        channel_b_top_k: int = 1,
        compute_counterfactual_rewards: bool = True,
        counterfactuals: Sequence[str] = ("label", "rank"),
        planner_mode: str = "full",
        enable_filter: bool = True,
        enable_retrieval: bool = True,
        enable_raaf: bool = True,
        enable_mb_moe: bool = True,
        evidence_groups: Sequence[str] = GROUPS,
        missing_modalities: Sequence[str] = (),
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_prompt_tokens: int = 2048,
    ):
        """
        Args:
            chat: AffectGPT ``Chat`` (model + tokenizer + audiovisual encoders)
            retriever: frozen ``DualChannelRetriever`` (part of the environment)
            raaf / mb_moe: fusion modules, jointly optimised with the Actor
            retrieval_top_k: K cognitive candidates per query group
            channel_b_top_k: perceptual evidence samples fed to RAAF
            planner_mode / enable_*: switches used by the paper ablations
            evidence_groups: Planner query groups that are retrieved (evidence-structure ablation)
            missing_modalities: modalities removed from the input (missing-modality setting)
        """
        self.chat = chat
        self.model = chat.model
        self.device = chat.device
        self.retriever = retriever
        self.raaf = raaf
        self.mb_moe = mb_moe
        self.encoder = MultimodalPromptEncoder(chat, max_prompt_tokens=max_prompt_tokens)
        self.max_new_tokens = {"q": max_new_tokens_q, "f": max_new_tokens_s, "g": max_new_tokens_g}
        self.retrieval_top_k = retrieval_top_k
        self.channel_b_top_k = channel_b_top_k
        self.compute_counterfactual_rewards = compute_counterfactual_rewards
        self.counterfactuals = tuple(counterfactuals)
        if planner_mode not in PLANNER_MODES:
            raise ValueError(f"planner_mode must be one of {PLANNER_MODES}")
        self.planner_mode = planner_mode
        self.enable_filter = enable_filter
        self.enable_retrieval = enable_retrieval
        self.enable_raaf = enable_raaf
        self.enable_mb_moe = enable_mb_moe
        self.evidence_groups = tuple(group for group in GROUPS if group in evidence_groups)
        if "support" not in self.evidence_groups:
            raise ValueError("Support evidence is retained in every variant.")
        self.missing_modalities = self._check_modalities(missing_modalities)
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

    @staticmethod
    def _check_modalities(missing: Sequence[str]) -> Tuple[str, ...]:
        unknown = set(missing) - set(MODALITIES)
        if unknown:
            raise ValueError(f"Unknown modalities: {sorted(unknown)}")
        if len(set(missing)) >= len(MODALITIES):
            raise ValueError("At least one modality must remain available.")
        return tuple(missing)

    @property
    def sampling_temperature(self) -> float:
        """Temperature of the behaviour policy; PPO log-probs are evaluated at this temperature."""

        return self.temperature if self.do_sample else 1.0

    # ══════════════════════════════════════════════════════
    # Prompts (shared with role-specific SFT)
    # ══════════════════════════════════════════════════════

    def planner_prompt_ids(self, sample: RolloutSample, subtitle: str, face_or_frame: str) -> torch.Tensor:
        return self.encoder.prompt_ids(lambda _: build_query_planner_messages(
            subtitle, sample.candidate_labels, sample.lang, face_or_frame,
        ))

    def filter_prompt_ids(self, sample, subtitle, candidates, face_or_frame) -> torch.Tensor:
        return self.encoder.prompt_ids(lambda budget: build_evidence_filter_messages(
            subtitle, candidates, sample.lang, face_or_frame, max_evidence_chars=budget,
        ))

    def generator_prompt_ids(self, sample, subtitle, evidence, face_or_frame) -> torch.Tensor:
        return self.encoder.prompt_ids(lambda budget: build_emotion_generator_messages(
            subtitle, evidence, sample.candidate_labels, sample.lang, face_or_frame,
            max_evidence_chars=budget,
        ))

    def _act(self, prompt_ids: torch.Tensor, features: Dict, agent: str) -> Tuple[torch.Tensor, str]:
        llama_model = self.model.llama_model
        with torch.no_grad():
            prompt_embeds = self.encoder.embed(prompt_ids, features, llama_model)
            return self.encoder.generate(
                llama_model,
                prompt_embeds,
                self.max_new_tokens[agent],
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
            )

    # ══════════════════════════════════════════════════════
    # Inputs, perceptual evidence and missing modalities
    # ══════════════════════════════════════════════════════

    def prepare_inputs(
        self,
        sample: RolloutSample,
        sample_data: dict,
        face_or_frame: str,
        missing_modalities: Optional[Sequence[str]] = None,
    ) -> PreparedInputs:
        """Encode x_v / x_a, retrieve E_perc and supplement missing modalities (paper Sec. 4.3)."""

        missing = self.missing_modalities if missing_modalities is None else self._check_modalities(missing_modalities)
        with torch.no_grad():
            features, vectors = self.encoder.extract_features(sample_data, face_or_frame)
        features = {key: (value.detach() if value is not None else None) for key, value in features.items()}
        for modality in ("video", "audio"):
            if modality in missing:
                features[modality] = None
                vectors.pop(modality, None)

        perceptual, partners = self.retrieve_perceptual_evidence(vectors, sample.sample_id)
        prepared = PreparedInputs(
            features=features,
            perceptual=perceptual,
            partners=partners,
            subtitle="" if "text" in missing else sample.subtitle,
        )
        # A missing modality is filled with the corresponding evidence of the closest sample.
        for modality in ("video", "audio"):
            if features.get(modality) is None and perceptual.get(modality) is not None:
                num_tokens = self.encoder.num_query_tokens(modality)
                features[modality] = perceptual[modality][:, :num_tokens].clone()
                prepared.supplemented.append(modality)
        if "text" in missing and partners:
            retrieved = str(partners[0].get("subtitle", "")).strip()
            if retrieved:
                prepared.subtitle = f"[transcript unavailable; closest retrieved transcript] {retrieved}"
                prepared.supplemented.append("text")
        return prepared

    def retrieve_perceptual_evidence(self, vectors, exclude_sample_id: Optional[str]):
        """E_perc: audiovisual tokens of the Top-k perceptually closest library samples."""

        perceptual = {"video": None, "audio": None}
        if not vectors or not self.enable_retrieval or not getattr(self.retriever, "multimodal_ready", False):
            return perceptual, []
        try:
            partners = self.retriever.retrieve_channel_B_candidates(
                vectors, top_k=self.channel_b_top_k, exclude_sample_id=exclude_sample_id,
            )
        except Exception as error:
            print(f"[AffectAgent] Perceptual retrieval failed: {error}")
            return perceptual, []
        tokens = [self.retriever.load_partner_tokens(partner["global_id"]) for partner in partners]
        for modality in ("video", "audio"):
            chunks = [item[modality] for item in tokens if item[modality] is not None]
            if chunks:
                perceptual[modality] = torch.cat(chunks, dim=1).to(self.device)
        partners = [dict(partner, **self.retriever.multimodal_meta.get(partner["global_id"], {})) for partner in partners]
        return perceptual, partners

    # ══════════════════════════════════════════════════════
    # RAAF + MB-MoE
    # ══════════════════════════════════════════════════════

    def run_fusion(self, features: Dict, perceptual: Dict, modules: Optional[Tuple] = None) -> Tuple[Dict, FusionDiagnostics]:
        """Equations (5)-(7): z_fuse = MB-MoE(RAAF(x_v, x_a, E_perc)).  Grad mode follows the caller.

        ``modules`` overrides ``(raaf, mb_moe)``; PPO uses it to evaluate the frozen SFT reference.
        """

        raaf, mb_moe = modules if modules is not None else (self.raaf, self.mb_moe)
        diagnostics = FusionDiagnostics()
        modules = [module for module in (raaf, mb_moe) if module is not None]
        if not modules or (features.get("video") is None and features.get("audio") is None):
            return dict(features), diagnostics
        fusion_dtype = next(modules[0].parameters()).dtype

        def prepare(value):
            if value is None:
                return None
            value = value if value.dim() == 3 else value.unsqueeze(0)
            return value.to(device=self.device, dtype=fusion_dtype)

        video, audio = prepare(features.get("video")), prepare(features.get("audio"))
        evidence_video, evidence_audio = prepare(perceptual.get("video")), prepare(perceptual.get("audio"))

        if self.enable_raaf and raaf is not None and (evidence_video is not None or evidence_audio is not None):
            video, audio, video_gate, audio_gate = raaf(video, audio, evidence_video, evidence_audio)
            diagnostics.video_gate_mean = round(video_gate, 4)
            diagnostics.audio_gate_mean = round(audio_gate, 4)
        if self.enable_mb_moe and mb_moe is not None:
            video, audio, (indices, weights) = mb_moe(video, audio, return_routing=True)
            diagnostics.moe_experts_activated = indices[0].tolist()
            diagnostics.moe_expert_weights = [round(float(w), 4) for w in weights[0].tolist()]

        fused = dict(features)
        fused["video"], fused["audio"] = video, audio
        return fused, diagnostics

    # ══════════════════════════════════════════════════════
    # Agents
    # ══════════════════════════════════════════════════════

    def run_query_planner(self, sample, subtitle, features, face_or_frame):
        prompt_ids = self.planner_prompt_ids(sample, subtitle, face_or_frame)
        response_ids, text = self._act(prompt_ids, features, "q")
        return parse_query_planner_output(text), prompt_ids, response_ids

    def retrieve_cognitive_evidence(self, queries: Dict[str, Dict], exclude_sample_id: Optional[str]) -> Dict[str, list]:
        """E_cog: Top-K library items for every query group."""

        groups = [group for group in self.evidence_groups if group in queries]
        if not groups:
            return {}
        try:
            return self.retriever.retrieve_channel_A(
                queries, top_k=self.retrieval_top_k, exclude_sample_id=exclude_sample_id, groups=groups,
            )
        except Exception as error:
            print(f"[AffectAgent] Cognitive retrieval failed: {error}")
            return {group: [] for group in groups}

    @staticmethod
    def input_queries(subtitle: str) -> Dict[str, Dict]:
        """Retrieval with the raw transcript (planner_mode="none", or a Planner output that cannot
        be parsed).  Without Planner hypotheses there is no confusion or counter query, so only
        the support pool is retrieved."""

        return {"support": {"query_text": subtitle}} if subtitle.strip() else {}

    def planner_queries(self, query_output: QueryPlannerOutput, subtitle: str) -> Dict[str, Dict]:
        if query_output is not None and query_output.valid:
            return {group: getattr(query_output, group) for group in self.evidence_groups}
        return self.input_queries(subtitle)

    def simple_label_queries(self, query_output: QueryPlannerOutput, sample: RolloutSample, subtitle: str) -> Dict[str, Dict]:
        """Score_label counterfactual: every Planner query is replaced by its emotion label only."""

        if query_output is None or not query_output.valid:
            return self.input_queries(subtitle)
        support_label = str(query_output.support.get("target_label", "")).strip()
        if not support_label:
            query_text = str(query_output.support.get("query_text", "")).lower()
            support_label = next(
                (str(label) for label in sample.candidate_labels if str(label).lower() in query_text),
                "neutral",
            )
        queries = {}
        for group in self.evidence_groups:
            label = str(getattr(query_output, group).get(QUERY_LABEL_KEYS[group], "")).strip()
            queries[group] = {"query_text": label or support_label}
        return queries

    def run_evidence_filter(self, sample, subtitle, candidates, features, face_or_frame):
        if not any(candidates.values()):
            return EvidenceFilterOutput(kept_ids={group: [] for group in candidates}), None, None
        prompt_ids = self.filter_prompt_ids(sample, subtitle, candidates, face_or_frame)
        response_ids, text = self._act(prompt_ids, features, "f")
        output = parse_evidence_filter_output(text, candidates)
        if not output.valid:
            output = self._rank_fallback(output, candidates)
        return output, prompt_ids, response_ids

    @staticmethod
    def _rank_fallback(output: EvidenceFilterOutput, candidates: Dict[str, list]) -> EvidenceFilterOutput:
        """Keep the rollout defined when decoding fails: groups the Filter did not answer
        completely keep their Top-1 item; groups it answered keep its decisions."""

        for group, items in candidates.items():
            if group not in output.decisions:
                output.decisions[group] = [index == 0 for index in range(len(items))]
                output.kept_ids[group] = [str(items[0].get("id", ""))] if items else []
        return output

    @staticmethod
    def resolve_evidence(filter_output: EvidenceFilterOutput, candidates: Dict[str, list]) -> Dict[str, List[EvidenceItem]]:
        """E*_cog: every candidate the Filter answered with Yes."""

        return evidence_items(candidates, filter_output.kept_ids)

    @staticmethod
    def resolve_rank_baseline(candidates: Dict[str, list]) -> Dict[str, List[EvidenceItem]]:
        """Score_rank counterfactual: bypass the Filter and pass the ranked Top-K items."""

        return evidence_items(candidates, {g: [str(item.get("id", "")) for item in items] for g, items in candidates.items()})

    def run_emotion_generator(self, sample, subtitle, evidence, fused_features, face_or_frame):
        prompt_ids = self.generator_prompt_ids(sample, subtitle, evidence, face_or_frame)
        response_ids, text = self._act(prompt_ids, fused_features, "g")
        output = parse_emotion_generator_output(text, sample.candidate_labels)
        return output, prompt_ids, response_ids

    # ══════════════════════════════════════════════════════
    # Full rollout
    # ══════════════════════════════════════════════════════

    def full_rollout(self, sample: RolloutSample, sample_data: Optional[dict], face_or_frame: str = "face") -> RolloutResult:
        """Run Planner -> Retriever -> Filter -> RAAF/MB-MoE -> Generator for one sample."""

        result = RolloutResult(sample=sample)
        if sample_data is None:
            result.query_output = QueryPlannerOutput(valid=False)
            result.filter_output = EvidenceFilterOutput(valid=False)
            result.generator_output = EmotionGeneratorOutput(valid=False)
            result.fusion_diagnostics = FusionDiagnostics()
            return result

        # Step 1: x_v, x_a (frozen encoders), E_perc and missing-modality supplementation.
        inputs = self.prepare_inputs(sample, sample_data, face_or_frame)
        subtitle, features = inputs.subtitle, inputs.features
        use_agents = self.enable_retrieval  # Planner/Filter have nothing to act on without retrieval.

        # Step 2: Query Planner observes x_t, x_v, x_a and the candidate set C.
        query_output = QueryPlannerOutput(valid=False)
        if use_agents and self.planner_mode != "none":
            query_output, result.q_query_ids, result.q_response_ids = self.run_query_planner(
                sample, subtitle, features, face_or_frame,
            )
        result.query_output = query_output

        # Step 3: cognitive channel E_cog (raw-input retrieval when the Planner is absent or fails).
        candidates = {}
        if use_agents:
            # Only the visible transcript is used: a removed text modality must not leak into queries.
            text = subtitle
            if self.planner_mode == "full":
                queries = self.planner_queries(query_output, text)
            elif self.planner_mode == "label":
                queries = self.simple_label_queries(query_output, sample, text)
            else:
                queries = self.input_queries(text)
            candidates = self.retrieve_cognitive_evidence(queries, exclude_sample_id=sample.sample_id)
        result.candidates = candidates

        # Step 4: Evidence Filter, one Yes/No decision per candidate.
        if use_agents and self.enable_filter:
            filter_output, result.f_query_ids, result.f_response_ids = self.run_evidence_filter(
                sample, subtitle, candidates, features, face_or_frame,
            )
            evidence = self.resolve_evidence(filter_output, candidates)
        else:
            filter_output = EvidenceFilterOutput(valid=False)
            evidence = self.resolve_rank_baseline(candidates)
        result.filter_output = filter_output
        result.selected_evidence = evidence

        # Step 5: RAAF + MB-MoE produce z_fuse.
        with torch.no_grad():
            fused, diagnostics = self.run_fusion(features, inputs.perceptual)
        if inputs.partners:
            partner = inputs.partners[0]
            diagnostics.fusion_partner_id = str(partner.get("name", ""))
            diagnostics.fusion_partner_label = str(partner.get("ovlabel", ""))
            diagnostics.fusion_partner_discrete_label = str(partner.get("discrete_label", ""))
        diagnostics.supplemented_modalities = list(inputs.supplemented)
        result.fusion_diagnostics = diagnostics

        # Step 6: Emotion Generator conditions on x_t, E*_cog and z_fuse.
        result.generator_output, result.g_query_ids, result.g_response_ids = self.run_emotion_generator(
            sample, subtitle, evidence, fused, face_or_frame,
        )
        result.runtime_context = {
            "features": features,
            "perceptual": inputs.perceptual,
            "subtitle": subtitle,
            "face_or_frame": face_or_frame,
        }

        # Equations (1)-(4): Score_label and Score_rank predictions.
        if self.compute_counterfactual_rewards and use_agents:
            self._run_reward_counterfactuals(result, subtitle, features, fused, face_or_frame)
        return result

    def _run_reward_counterfactuals(self, result, subtitle, features, fused, face_or_frame) -> None:
        sample = result.sample

        # Score_label: Planner output replaced by simple emotion labels, rest of the pipeline unchanged.
        if "label" in self.counterfactuals:
            label_queries = self.simple_label_queries(result.query_output, sample, subtitle)
            label_candidates = self.retrieve_cognitive_evidence(label_queries, exclude_sample_id=sample.sample_id)
            if self.enable_filter:
                label_filter, _, _ = self.run_evidence_filter(sample, subtitle, label_candidates, features, face_or_frame)
                label_evidence = self.resolve_evidence(label_filter, label_candidates)
            else:
                label_evidence = self.resolve_rank_baseline(label_candidates)
            result.label_baseline_output, _, _ = self.run_emotion_generator(
                sample, subtitle, label_evidence, fused, face_or_frame,
            )

        # Score_rank: Filter bypassed, Top-K items of every group passed to the Generator.
        if "rank" in self.counterfactuals:
            rank_evidence = self.resolve_rank_baseline(result.candidates)
            result.rank_baseline_output, _, _ = self.run_emotion_generator(
                sample, subtitle, rank_evidence, fused, face_or_frame,
            )

    def batch_rollout(self, samples: List[RolloutSample], sample_data_list: List[dict], face_or_frame: str = "face") -> List[RolloutResult]:
        return [
            self.full_rollout(sample, sample_data, face_or_frame)
            for sample, sample_data in zip(samples, sample_data_list)
        ]

    # ══════════════════════════════════════════════════════
    # PPO replay
    # ══════════════════════════════════════════════════════

    def _replay_inputs(self, result: RolloutResult, agent_name: str, llama_model, fusion: Optional[Tuple] = None):
        context = result.runtime_context
        features = context.get("features")
        if features is None:
            raise RuntimeError("Missing multimodal context for PPO replay.")
        if agent_name == "q":
            prompt_ids, response_ids = result.q_query_ids, result.q_response_ids
        elif agent_name in {"f", "s"}:
            prompt_ids, response_ids = result.f_query_ids, result.f_response_ids
        elif agent_name == "g":
            prompt_ids, response_ids = result.g_query_ids, result.g_response_ids
            # Re-run RAAF + MB-MoE inside the caller's grad mode so the Generator's PPO loss
            # also updates the fusion modules (they stay in the computation graph).
            features, _ = self.run_fusion(features, context["perceptual"], modules=fusion)
        else:
            raise ValueError(f"Unknown agent name for PPO replay: {agent_name}")
        if prompt_ids is None or response_ids is None:
            return None, None
        return self.encoder.embed(prompt_ids, features, llama_model), response_ids

    def compute_agent_log_probs_and_state(self, result: RolloutResult, agent_name: str, llama_model, fusion=None):
        prompt_embeds, response_ids = self._replay_inputs(result, agent_name, llama_model, fusion)
        if prompt_embeds is None:
            return torch.empty(0, device=self.device), None
        return self.encoder.response_log_probs_and_states(
            llama_model, prompt_embeds, response_ids, temperature=self.sampling_temperature,
        )

    def compute_agent_log_probs(self, result: RolloutResult, agent_name: str, llama_model, fusion=None) -> torch.Tensor:
        log_probs, _ = self.compute_agent_log_probs_and_state(result, agent_name, llama_model, fusion)
        return log_probs

    def collect_ppo_trajectories(self, results: List[RolloutResult]) -> List[Dict[str, object]]:
        """One trajectory per agent action, each carrying that agent's reward R_i."""

        trajectories = []
        for result in results:
            rewards = result.rewards
            if rewards is None:
                continue
            for agent, response_ids, reward in (
                ("q", result.q_response_ids, rewards.r_planner),
                ("f", result.f_response_ids, rewards.r_filter),
                ("g", result.g_response_ids, rewards.r_generator),
            ):
                if response_ids is not None and response_ids.numel() > 0:
                    trajectories.append({"agent": agent, "result": result, "reward": float(reward)})
        return trajectories

    @staticmethod
    def release_context(results: List[RolloutResult]) -> None:
        """Drop tensors kept for replay once the PPO update is finished."""

        for result in results:
            result.runtime_context = {}


# Backward-compatible pre-release class name.
MmoaOrchestrator = AffectAgentPipeline
