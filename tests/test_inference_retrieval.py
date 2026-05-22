import unittest
from pathlib import Path

from omegaconf import OmegaConf

from core.hypothesis_set import Hypothesis
from core.summary import retrieve_inference_profile
from core.utils import TracerConfig, TracerContext
from data import Conversation, Turn, UserData
from eval.runner import _metrics_cache_matches, evaluate_user_record
from model.base import GenerationConfig
from prompt import prism_prompts


class FakeSummaryModel:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return {"output": "retrieved profile"}


class FakeHypothesisSet:
    def __init__(self, items):
        self.items = items
        self.hypotheses = {item[0].id: item[0] for item in items}
        self.global_prior = {item[0].id: item[1] for item in items}
        self.last_query = None
        self.last_top_k = None

    def retrieve_hypotheses_with_scores(self, query, top_k=5, exclude_ids=None):
        self.last_query = query
        self.last_top_k = top_k
        selected = self.items[:top_k]
        return (
            [item[0] for item in selected],
            [item[1] for item in selected],
            [item[2] for item in selected],
        )

    def get_similarity(self, key1, key2):
        return 0.0


class FakeEvalModel:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return {"output": {"ranking": ["C1", "C2"], "justification": "test"}}


def make_hypothesis(idx: int, content: str) -> Hypothesis:
    return Hypothesis(id=f"h{idx}", category="test", content=content)


def make_context(items, cfg: TracerConfig | None = None) -> TracerContext:
    return TracerContext(
        model=FakeSummaryModel(),
        hypothesis_set=FakeHypothesisSet(items),
        tracer_config=cfg or TracerConfig(
            inference_profile_source="retrieved",
            inference_retrieve_top_k=3,
            inference_retrieve_pool_k=4,
            inference_min_prior=0.1,
        ),
        generation_config=GenerationConfig(max_retries=1),
        prompts=prism_prompts(),
    )


class InferenceRetrievalTests(unittest.TestCase):
    def test_default_config_uses_working_profile(self):
        self.assertEqual(TracerConfig().inference_profile_source, "working")

    def test_personamem_tracer_config_uses_retrieved_profile(self):
        cfg = OmegaConf.to_container(
            OmegaConf.load(Path("config/tracer/personamem_v2.yaml")),
            resolve=True,
        )
        self.assertEqual(cfg["inference_profile_source"], "retrieved")
        self.assertEqual(cfg["inference_retrieve_top_k"], 12)

    def test_inference_retrieval_query_excludes_candidates_and_choice(self):
        items = [
            (make_hypothesis(1, "[background_fact] user likes hiking"), 0.2, 0.9),
            (make_hypothesis(2, "[domain_preference] user prefers low-impact routes"), 0.2, 0.8),
            (make_hypothesis(3, "[adaptation_rule] use knee-friendly trail advice"), 0.2, 0.7),
        ]
        context = make_context(items)
        history = [
            Turn(user_message="Earlier hiking question", candidates=["OLD_CANDIDATE_SECRET"], chosen="OLD_CANDIDATE_SECRET"),
            Turn(user_message="Plan a weekend walk", candidates=["CANDIDATE_SECRET"], chosen="CANDIDATE_SECRET"),
        ]

        profile, diagnostics = retrieve_inference_profile(history, context)

        self.assertEqual(profile, "retrieved profile")
        self.assertEqual(diagnostics["source"], "retrieved")
        self.assertNotIn("CANDIDATE_SECRET", context.hypothesis_set.last_query)
        self.assertNotIn("chosen", context.hypothesis_set.last_query.lower())
        self.assertIn("[background_fact]", context.model.prompts[0])

    def test_low_prior_filter_falls_back_when_too_few_pass(self):
        items = [
            (make_hypothesis(1, "one"), 0.05, 0.9),
            (make_hypothesis(2, "two"), 0.04, 0.8),
            (make_hypothesis(3, "three"), 0.03, 0.7),
            (make_hypothesis(4, "four"), 0.02, 0.6),
        ]
        context = make_context(items)
        _, diagnostics = retrieve_inference_profile(
            [Turn(user_message="current", candidates=["A"], chosen="A")],
            context,
        )

        self.assertFalse(diagnostics["used_prior_filter"])
        self.assertEqual(len(diagnostics["retrieved"]), 3)

    def test_prior_filter_applies_when_enough_pass(self):
        items = [
            (make_hypothesis(1, "one"), 0.2, 0.9),
            (make_hypothesis(2, "two"), 0.15, 0.8),
            (make_hypothesis(3, "three"), 0.12, 0.7),
            (make_hypothesis(4, "four"), 0.02, 0.6),
        ]
        context = make_context(items)
        _, diagnostics = retrieve_inference_profile(
            [Turn(user_message="current", candidates=["A"], chosen="A")],
            context,
        )

        self.assertTrue(diagnostics["used_prior_filter"])
        self.assertEqual(len(diagnostics["retrieved"]), 3)
        self.assertTrue(all(item["prior"] >= 0.1 for item in diagnostics["retrieved"]))

    def test_eval_prefers_inference_profile_and_falls_back_to_summary(self):
        user = UserData(
            user_id="u1",
            conversations=[
                Conversation(
                    conversation_id="c1",
                    turns=[
                        Turn(
                            turn_id="t1",
                            user_message="Pick one",
                            candidates=["A", "B"],
                            chosen="A",
                            chosen_idx=0,
                        ),
                        Turn(
                            turn_id="t2",
                            user_message="Pick again",
                            candidates=["A", "B"],
                            chosen="A",
                            chosen_idx=0,
                        ),
                    ],
                )
            ],
        )
        model = FakeEvalModel()
        evaluate_user_record(
            user_data=user,
            record={
                "user": "u1",
                "turns": [
                    {
                        "adapted": {"success": False},
                        "inference_profile": "USE_RETRIEVED_PROFILE",
                        "summary": "DO_NOT_USE_SUMMARY",
                    },
                    {"adapted": {"success": False}},
                ],
            },
            eval_model=model,
            eval_cfg=GenerationConfig(max_retries=1),
            embed_cfg=None,
            prompts=prism_prompts(),
        )
        self.assertIn("USE_RETRIEVED_PROFILE", model.prompts[0])
        self.assertNotIn("DO_NOT_USE_SUMMARY", model.prompts[0])

        fallback_model = FakeEvalModel()
        evaluate_user_record(
            user_data=user,
            record={
                "user": "u1",
                "turns": [
                    {
                        "adapted": {"success": False},
                        "summary": "USE_SUMMARY_FALLBACK",
                    },
                    {"adapted": {"success": False}},
                ],
            },
            eval_model=fallback_model,
            eval_cfg=GenerationConfig(max_retries=1),
            embed_cfg=None,
            prompts=prism_prompts(),
        )
        self.assertIn("USE_SUMMARY_FALLBACK", fallback_model.prompts[1])

    def test_prediction_model_is_separate_from_eval_model(self):
        user = UserData(
            user_id="u1",
            conversations=[
                Conversation(
                    conversation_id="c1",
                    turns=[
                        Turn(
                            turn_id="t1",
                            user_message="Pick one",
                            candidates=["A", "B"],
                            chosen="A",
                            chosen_idx=0,
                        )
                    ],
                )
            ],
        )
        eval_model = FakeEvalModel()
        prediction_model = FakeEvalModel()

        metrics = evaluate_user_record(
            user_data=user,
            record={"user": "u1", "turns": [{"adapted": {"success": False}}]},
            eval_model=eval_model,
            eval_cfg=GenerationConfig(max_retries=1),
            embed_cfg=None,
            prompts=prism_prompts(),
            prediction_model=prediction_model,
            prediction_cfg=GenerationConfig(max_retries=1),
            eval_model_name="eval-model",
            prediction_model_name="prediction-model",
        )

        self.assertEqual(len(eval_model.prompts), 0)
        self.assertEqual(len(prediction_model.prompts), 1)
        self.assertEqual(metrics["eval_model"], "eval-model")
        self.assertEqual(metrics["prediction_model"], "prediction-model")

    def test_prediction_only_skips_response_and_profile_eval(self):
        user = UserData(
            user_id="u1",
            conversations=[
                Conversation(
                    conversation_id="c1",
                    turns=[
                        Turn(
                            turn_id="t1",
                            user_message="Pick one",
                            candidates=["A", "B"],
                            chosen="A",
                            chosen_idx=0,
                        )
                    ],
                )
            ],
            gt_profile="ground truth",
        )
        eval_model = FakeEvalModel()
        prediction_model = FakeEvalModel()

        metrics = evaluate_user_record(
            user_data=user,
            record={
                "user": "u1",
                "turns": [
                    {
                        "adapted": {"success": True, "response": "A"},
                    }
                ],
                "final_profile": "profile",
            },
            eval_model=eval_model,
            eval_cfg=GenerationConfig(max_retries=1),
            embed_cfg=None,
            prompts=prism_prompts(),
            prediction_model=prediction_model,
            prediction_cfg=GenerationConfig(max_retries=1),
            prediction_model_name="prediction-model",
            prediction_only=True,
        )

        self.assertEqual(metrics["eval_scope"], "prediction_only")
        self.assertEqual(len(prediction_model.prompts), 1)
        self.assertEqual(len(eval_model.prompts), 0)
        self.assertNotIn("adaptation", metrics["turns"][0])
        self.assertEqual(metrics["profile_alignment"], {"skipped": True, "reason": "prediction_only"})

    def test_metrics_cache_checks_prediction_model(self):
        self.assertTrue(
            _metrics_cache_matches(
                {"eval_model": "gemini", "prediction_model": "gpt-5"},
                eval_model_name="gemini",
                prediction_model_name="gpt-5",
                eval_scope=None,
            )
        )
        self.assertFalse(
            _metrics_cache_matches(
                {"eval_model": "gemini", "prediction_model": "gemini", "eval_scope": "prediction_only"},
                eval_model_name="gemini",
                prediction_model_name="gpt-5",
                eval_scope="prediction_only",
            )
        )
        self.assertFalse(
            _metrics_cache_matches(
                {"prediction_model": "gpt-5", "eval_scope": "full"},
                prediction_model_name="gpt-5",
                eval_scope="prediction_only",
            )
        )


if __name__ == "__main__":
    unittest.main()
