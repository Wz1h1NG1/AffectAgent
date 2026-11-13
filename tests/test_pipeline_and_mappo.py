import copy
import unittest

try:
    import torch
except (ImportError, OSError):
    torch = None


@unittest.skipIf(torch is None, "PyTorch runtime is unavailable")
class FusionTests(unittest.TestCase):
    def test_modules_start_as_identity(self):
        from affectagent.fusion_modules import MBMoE, RAAF

        raaf, mb_moe = RAAF(dim=8, n_heads=2), MBMoE(dim=8, n_experts=3, top_k=2, expert_hidden_dim=16)
        video, audio = torch.randn(2, 4, 8), torch.randn(2, 3, 8)
        fused_video, fused_audio, _, _ = raaf(video, audio, torch.randn(2, 5, 8), torch.randn(2, 6, 8))
        self.assertTrue(torch.allclose(fused_video, video))
        balanced_video, balanced_audio = mb_moe(fused_video, fused_audio)
        self.assertTrue(torch.allclose(balanced_video, video, atol=1e-6))
        self.assertTrue(torch.allclose(balanced_audio, audio, atol=1e-6))

    def test_gradients_reach_every_fusion_parameter_group(self):
        from affectagent.fusion_modules import MBMoE, RAAF

        raaf, mb_moe = RAAF(dim=8, n_heads=2), MBMoE(dim=8, n_experts=2, top_k=2, expert_hidden_dim=8)
        video, audio = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
        fused = raaf(video, audio, torch.randn(1, 3, 8), torch.randn(1, 3, 8))
        balanced_video, balanced_audio = mb_moe(fused[0], fused[1])
        (balanced_video.sum() + balanced_audio.sum()).backward()
        self.assertIsNotNone(raaf.video_cross_attn.out_proj.weight.grad)
        self.assertIsNotNone(mb_moe.experts[0].fc2.weight.grad)
        self.assertIsNotNone(mb_moe.router.weight.grad)

    def test_missing_modality_is_passed_through(self):
        from affectagent.fusion_modules import MBMoE, RAAF

        raaf, mb_moe = RAAF(dim=8, n_heads=2), MBMoE(dim=8)
        fused_video, fused_audio, _, _ = raaf(torch.randn(1, 2, 8), None, torch.randn(1, 3, 8), None)
        self.assertIsNone(fused_audio)
        balanced_video, balanced_audio = mb_moe(fused_video, None)
        self.assertEqual(tuple(balanced_video.shape), (1, 2, 8))
        self.assertIsNone(balanced_audio)


@unittest.skipIf(torch is None, "PyTorch runtime is unavailable")
class GaeTests(unittest.TestCase):
    def test_terminal_reward_gae(self):
        from affectagent.mappo import compute_gae

        advantages, returns = compute_gae(torch.tensor([0.0, 0.0, 1.0]), torch.zeros(3), gamma=0.9, gae_lambda=1.0)
        self.assertTrue(torch.allclose(advantages, torch.tensor([0.81, 0.9, 1.0])))
        self.assertTrue(torch.allclose(returns, advantages))


@unittest.skipIf(torch is None, "PyTorch runtime is unavailable")
class PipelineTests(unittest.TestCase):
    def setUp(self):
        from affectagent.fusion_modules import MBMoE, RAAF
        from affectagent.schemas import RolloutSample
        from tests.fakes import HIDDEN, FakeChat, FakeRetriever, ScriptedPipeline, sample_data

        torch.manual_seed(0)
        self.chat = FakeChat()
        self.retriever = FakeRetriever()
        self.raaf = RAAF(dim=HIDDEN, n_heads=2)
        self.mb_moe = MBMoE(dim=HIDDEN, n_experts=2, top_k=1, expert_hidden_dim=HIDDEN)
        self.make_pipeline = lambda **kwargs: ScriptedPipeline(
            self.chat, self.retriever, self.raaf, self.mb_moe, retrieval_top_k=2, **kwargs,
        )
        self.sample = RolloutSample("x", "I passed the exam!", "happy", ["happy", "sad", "neutral"], lang="en")
        self.data = sample_data()

    def test_full_rollout_and_counterfactuals(self):
        from affectagent.reward import AffectiveRewardComputer

        pipeline = self.make_pipeline()
        result = pipeline.full_rollout(self.sample, self.data, "frame")
        self.assertEqual(pipeline.generator_calls, 3)
        self.assertTrue(result.query_output.valid and result.filter_output.valid)
        # Filter decisions select E*_cog: support keeps item 0, confusion item 1, counter nothing.
        kept = {group: [item.evidence_id for item in items] for group, items in result.selected_evidence.items()}
        self.assertEqual(kept, {"support": ["support0"], "confusion": ["confusion1"], "counter": []})
        # Score_label retrieval uses the Planner's simple labels.
        self.assertEqual(self.retriever.calls[1]["support"]["query_text"], "happy")
        self.assertEqual(self.retriever.calls[1]["counter"]["query_text"], "sad")
        rewards = AffectiveRewardComputer(use_emotion_wheel=False).compute_pipeline_rewards(result)
        self.assertEqual((rewards.score_full, rewards.score_label, rewards.score_rank), (1.0, 0.0, 1.0))
        self.assertEqual(result.fusion_diagnostics.fusion_partner_id, "lib7")

    def test_naive_rag_retrieves_with_the_transcript_only(self):
        pipeline = self.make_pipeline(planner_mode="none", enable_filter=False, compute_counterfactual_rewards=False)
        result = pipeline.full_rollout(self.sample, self.data, "frame")
        self.assertEqual(self.retriever.calls[0], {"support": {"query_text": "I passed the exam!"}})
        self.assertIsNone(result.q_response_ids)
        self.assertEqual(len(result.selected_evidence["support"]), 2)

    def test_missing_text_and_audio_are_supplemented(self):
        pipeline = self.make_pipeline(missing_modalities=("text", "audio"), compute_counterfactual_rewards=False)
        result = pipeline.full_rollout(self.sample, self.data, "frame")
        self.assertEqual(sorted(result.fusion_diagnostics.supplemented_modalities), ["audio", "text"])
        self.assertNotIn("I passed the exam!", pipeline.prompts["q"][0])
        self.assertIn("what a great day", pipeline.prompts["q"][0])

    def test_mappo_update_trains_actor_fusion_and_critic(self):
        from affectagent.mappo import AdapterDisabledReference, PolicyGradientUpdater, SequenceValueHead
        from affectagent.reward import AffectiveRewardComputer
        from tests.fakes import HIDDEN

        pipeline = self.make_pipeline()
        llama = self.chat.model.llama_model
        critic = SequenceValueHead(HIDDEN)
        reference_fusion = tuple(copy.deepcopy(module).requires_grad_(False) for module in (self.raaf, self.mb_moe))
        actor = [p for p in llama.parameters() if p.requires_grad]
        fusion = list(self.raaf.parameters()) + list(self.mb_moe.parameters())
        optimizer = torch.optim.AdamW([
            {"params": actor, "lr": 1e-2}, {"params": fusion, "lr": 1e-2}, {"params": list(critic.parameters()), "lr": 1e-2},
        ])
        updater = PolicyGradientUpdater(
            self.chat.model, AdapterDisabledReference(llama), critic, optimizer, kl_coef=0.1, ppo_epochs=2,
            device="cpu", fusion_modules=(self.raaf, self.mb_moe), reference_fusion=reference_fusion,
        )
        before = [p.detach().clone() for p in (actor[0], self.raaf.video_cross_attn.out_proj.weight, critic.summary[-1].weight)]

        results = [pipeline.full_rollout(self.sample, self.data, "frame")]
        AffectiveRewardComputer(use_emotion_wheel=False).batch_compute_rewards(results)
        trajectories = pipeline.collect_ppo_trajectories(results)
        self.assertEqual(sorted(t["agent"] for t in trajectories), ["f", "g", "q"])
        stats = updater.update_step(trajectories, pipeline)

        self.assertEqual(stats["update_rounds"], 2)
        self.assertEqual(stats["num_trajectories"], 3)
        after = (actor[0], self.raaf.video_cross_attn.out_proj.weight, critic.summary[-1].weight)
        for old, new in zip(before, after):
            self.assertFalse(torch.equal(old, new.detach()))
        # At the start π_θ = π_SFT (zero adapter, identical fusion copies), so the KL term is 0.
        self.assertAlmostEqual(stats["mean_sequence_kl"], 0.0, places=5)

    def test_reference_policy_ignores_the_rl_adapter(self):
        from affectagent.mappo import AdapterDisabledReference

        llama = self.chat.model.llama_model
        torch.nn.init.normal_(llama.adapter.weight)
        embeds = torch.randn(1, 3, llama.config.hidden_size)
        with llama.disable_adapter():
            expected = llama(inputs_embeds=embeds).logits
        self.assertTrue(torch.allclose(AdapterDisabledReference(llama)(inputs_embeds=embeds).logits, expected))
        self.assertFalse(torch.allclose(llama(inputs_embeds=embeds).logits, expected))


if __name__ == "__main__":
    unittest.main()
