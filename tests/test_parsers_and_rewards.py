import json
import unittest

from affectagent.labels import label_f1, split_labels
from affectagent.prompts import (
    build_evidence_filter_messages,
    build_query_planner_messages,
    format_filter_target,
    format_generator_target,
    parse_emotion_generator_output,
    parse_evidence_filter_output,
    parse_query_planner_output,
)
from affectagent.reward import AffectiveRewardComputer
from affectagent.schemas import EmotionGeneratorOutput, RolloutResult, RolloutSample


CANDIDATES = {
    "support": [{"id": "s1", "text": "a"}, {"id": "s2", "text": "b"}],
    "confusion": [{"id": "c1", "text": "c"}],
    "counter": [{"id": "n1", "text": "d"}, {"id": "n2", "text": "e"}],
}


class QueryPlannerTests(unittest.TestCase):
    def test_planner_observes_all_modalities_and_labels(self):
        prompt = build_query_planner_messages("hello", ["happy", "sad"], "en", "face")[1]["content"]
        for token in ("hello", "<FaceHere>", "<AudioHere>", "happy, sad"):
            self.assertIn(token, prompt)

    def test_query_output_requires_all_three_queries(self):
        payload = {"support": {"query_text": "a"}, "confusion": {"query_text": "b"}, "counter": {"query_text": "c"}}
        self.assertTrue(parse_query_planner_output(json.dumps(payload)).valid)
        del payload["counter"]
        self.assertFalse(parse_query_planner_output(json.dumps(payload)).valid)


class EvidenceFilterTests(unittest.TestCase):
    def test_one_yes_no_decision_per_candidate(self):
        output = parse_evidence_filter_output(
            '{"support": ["Yes", "Yes"], "confusion": ["No"], "counter": ["No", "Yes"]}', CANDIDATES,
        )
        self.assertTrue(output.valid)
        self.assertEqual(output.kept_ids, {"support": ["s1", "s2"], "confusion": [], "counter": ["n2"]})
        self.assertEqual(output.num_kept, 3)

    def test_missing_decision_marks_group_unreadable(self):
        output = parse_evidence_filter_output('{"support": ["Yes"], "confusion": ["No"], "counter": ["No", "No"]}', CANDIDATES)
        self.assertFalse(output.valid)
        self.assertNotIn("support", output.decisions)
        self.assertEqual(output.kept_ids["counter"], [])

    def test_single_id_format(self):
        output = parse_evidence_filter_output(
            '{"support": {"id": "s2"}, "confusion": {"id": "c1"}, "counter": {"id": "n1"}}', CANDIDATES,
        )
        self.assertTrue(output.valid)
        self.assertEqual(output.kept_ids["support"], ["s2"])
        bad = parse_evidence_filter_output('{"support": {"id": "x"}, "confusion": {"id": "c1"}, "counter": {"id": "n1"}}', CANDIDATES)
        self.assertFalse(bad.valid)

    def test_sft_target_round_trip(self):
        decisions = {"support": [True, False], "confusion": [True], "counter": [False, False]}
        output = parse_evidence_filter_output(format_filter_target(decisions), CANDIDATES)
        self.assertEqual(output.decisions, decisions)

    def test_filter_prompt_lists_every_candidate_with_its_labels(self):
        candidates = {"support": [{"id": "s1", "text": "t", "label_hint": "happy, relaxed"}]}
        prompt = build_evidence_filter_messages("hi", candidates, "en")[1]["content"]
        self.assertIn("labels=happy, relaxed", prompt)
        self.assertIn('"support": ["Yes|No"]', prompt)


class EmotionGeneratorTests(unittest.TestCase):
    def test_open_vocabulary_multi_label_prediction(self):
        output = parse_emotion_generator_output(
            '{"reasoning": "smiles", "prediction": ["Happy", "relaxed", "happy"]}', ["happy", "sad"],
        )
        self.assertEqual(output.labels, ["happy", "relaxed"])
        self.assertEqual(output.prediction, "happy, relaxed")

    def test_phrase_is_snapped_to_vocabulary_but_negation_is_kept(self):
        self.assertEqual(parse_emotion_generator_output('{"prediction": "happy emotion"}', ["happy"]).labels, ["happy"])
        self.assertEqual(parse_emotion_generator_output('{"prediction": "not happy"}', ["happy"]).labels, ["not happy"])

    def test_truncated_json_still_yields_prediction(self):
        output = parse_emotion_generator_output('{"prediction": ["sad", "tired"], "reasoning": "long text that was cu', [])
        self.assertEqual(output.labels, ["sad", "tired"])

    def test_sft_target_round_trip(self):
        target = format_generator_target("cues", ["angry", "frustrated"])
        self.assertEqual(parse_emotion_generator_output(target, []).labels, ["angry", "frustrated"])


class RewardTests(unittest.TestCase):
    def _result(self, full, label, rank, truth="happy"):
        return RolloutResult(
            sample=RolloutSample("x", "", truth, ["happy", "sad"]),
            generator_output=EmotionGeneratorOutput(prediction=full),
            label_baseline_output=EmotionGeneratorOutput(prediction=label),
            rank_baseline_output=EmotionGeneratorOutput(prediction=rank),
        )

    def test_equations_1_to_4(self):
        reward = AffectiveRewardComputer(lambda_planner=0.5, lambda_filter=0.25, use_emotion_wheel=False)
        values = reward.compute_pipeline_rewards(self._result("happy", "sad", "happy"))
        self.assertEqual((values.score_full, values.score_label, values.score_rank), (1.0, 0.0, 1.0))
        self.assertEqual((values.r_shared, values.r_generator), (1.0, 1.0))
        self.assertEqual(values.r_planner, 1.5)
        self.assertEqual(values.r_filter, 1.0)

    def test_reward_requires_counterfactual_predictions(self):
        result = RolloutResult(sample=RolloutSample("x", "", "happy", []), generator_output=EmotionGeneratorOutput(prediction="happy"))
        with self.assertRaises(ValueError):
            AffectiveRewardComputer(use_emotion_wheel=False).compute_pipeline_rewards(result)

    def test_multi_label_f1(self):
        self.assertAlmostEqual(label_f1("happy, relaxed", "happy, excited, relaxed"), 0.8)
        self.assertEqual(split_labels("Happy; relaxed，happy"), ["happy", "relaxed"])


if __name__ == "__main__":
    unittest.main()
