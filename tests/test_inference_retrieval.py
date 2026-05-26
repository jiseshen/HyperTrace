import unittest
from pathlib import Path

from omegaconf import OmegaConf

from core.hypothesis_set import Hypothesis
from core.summary import retrieve_inference_profile, retrieve_legacy_prediction_profile
from core.utils import OverrideConfig, TaskOverride, TracerConfig, TracerContext
from data import Conversation, Turn, UserData
from eval.runner import _metrics_cache_matches, evaluate_user_record
from model.base import GenerationConfig
from prompt import load_prompt_adapter, prism_prompts


class FakeSummaryModel:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return {"output": "retrieved profile"}


class FakeFilteringModel:
    def __init__(self, content_ids=None, style_ids=None, boundary_ids=None):
        self.content_ids = content_ids or []
        self.style_ids = style_ids or []
        self.boundary_ids = boundary_ids or []
        self.prompts = []
        self.schemas = []
        self.kwargs = []

    def generate(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        if kwargs.get("schema") is not None:
            self.schemas.append(kwargs["schema"])
            return {
                "output": {
                    "content_ids": self.content_ids,
                    "style_ids": self.style_ids,
                    "boundary_ids": self.boundary_ids,
                    "reason": "selected by test",
                }
            }
        return {"output": "retrieved profile"}


class FakeHypothesisSet:
    def __init__(self, items):
        self.items = items
        self.hypotheses = {item[0].id: item[0] for item in items}
        self.global_prior = {item[0].id: item[1] for item in items}
        self.last_query = None
        self.last_top_k = None
        self.queries = []

    def retrieve_hypotheses_with_scores(self, query, top_k=5, exclude_ids=None):
        self.last_query = query
        self.last_top_k = top_k
        self.queries.append(query)
        selected = self.items[:top_k]
        return (
            [item[0] for item in selected],
            [item[1] for item in selected],
            [item[2] for item in selected],
        )

    def get_similarity(self, key1, key2):
        return 0.0


class QueryAwareFakeHypothesisSet(FakeHypothesisSet):
    def __init__(self, current_items, context_items):
        super().__init__(current_items + context_items)
        self.current_items = current_items
        self.context_items = context_items

    def retrieve_hypotheses_with_scores(self, query, top_k=5, exclude_ids=None):
        self.last_query = query
        self.last_top_k = top_k
        self.queries.append(query)
        items = self.current_items if query.startswith("[Current user message]") else self.context_items
        selected = items[:top_k]
        return (
            [item[0] for item in selected],
            [item[1] for item in selected],
            [item[2] for item in selected],
        )


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
        self.assertEqual(cfg["inference_retrieval_query_mode"], "current_and_context")
        self.assertEqual(cfg["inference_retrieve_top_k"], 14)
        self.assertEqual(cfg["inference_retrieve_pool_k"], 40)
        self.assertEqual(cfg["inference_filter_pool_k"], 20)
        self.assertEqual(cfg["inference_content_max_k"], 6)
        self.assertEqual(cfg["inference_style_max_k"], 5)
        self.assertFalse(cfg["inference_model_filter"])
        self.assertFalse(cfg["inference_empty_profile_on_no_relevance"])
        self.assertEqual(cfg["prediction_profile_source"], "legacy_retrieved")
        self.assertEqual(cfg["prediction_retrieve_top_k"], 12)
        self.assertEqual(cfg["prediction_retrieve_pool_k"], 30)
        self.assertEqual(cfg["prediction_min_prior"], 0.1)

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
        self.assertIn("[Current inference task]", context.model.prompts[0])

    def test_current_message_retrieval_uses_current_only_query(self):
        current_items = [
            (make_hypothesis(1, "[domain_preference] user likes familiar dinner dishes"), 0.2, 0.9),
            (make_hypothesis(2, "[adaptation_rule] use dinner party planning cues"), 0.2, 0.8),
            (make_hypothesis(3, "[constraint_boundary] avoid overfitting dinner to unrelated history"), 0.2, 0.7),
        ]
        context_items = [
            (make_hypothesis(4, "[domain_preference] user wants grant paperwork checklists"), 0.2, 0.95),
            (make_hypothesis(5, "[constraint_boundary] redact identity documents"), 0.2, 0.9),
            (make_hypothesis(6, "[adaptation_rule] verify official portals"), 0.2, 0.8),
        ]
        cfg = TracerConfig(
            inference_profile_source="retrieved",
            inference_retrieve_top_k=5,
            inference_retrieve_pool_k=5,
            inference_min_prior=0.1,
        )
        context = TracerContext(
            model=FakeSummaryModel(),
            hypothesis_set=QueryAwareFakeHypothesisSet(current_items, context_items),
            tracer_config=cfg,
            generation_config=GenerationConfig(max_retries=1),
            prompts=prism_prompts(),
        )

        _, diagnostics = retrieve_inference_profile(
            [
                Turn(user_message="Earlier grant paperwork question", candidates=["A"], chosen="A"),
                Turn(user_message="Plan a dinner party", candidates=["B"], chosen="B"),
            ],
            context,
        )

        self.assertEqual(len(context.hypothesis_set.queries), 1)
        self.assertNotIn("Earlier grant paperwork question", context.hypothesis_set.queries[0])
        self.assertEqual([item["id"] for item in diagnostics["retrieved"]], ["h1", "h2", "h3"])
        self.assertNotIn("grant paperwork", context.model.prompts[0])
        self.assertIn("Plan a dinner party", context.model.prompts[0])

    def test_legacy_prediction_profile_uses_contextual_query_and_legacy_thresholds(self):
        current_items = [
            (make_hypothesis(1, "[domain_preference] current-only dinner cue"), 0.2, 0.9),
        ]
        context_items = [
            (make_hypothesis(2, "[domain_preference] contextual paperwork cue"), 0.2, 0.95),
            (make_hypothesis(3, "[adaptation_rule] contextual official portal cue"), 0.2, 0.9),
            (make_hypothesis(4, "[constraint_boundary] contextual redaction cue"), 0.2, 0.85),
            (make_hypothesis(5, "[domain_preference] low-prior stale cue"), 0.09, 0.8),
        ]
        cfg = TracerConfig(
            prediction_retrieve_top_k=12,
            prediction_retrieve_pool_k=30,
            prediction_min_prior=0.1,
        )
        context = TracerContext(
            model=FakeSummaryModel(),
            hypothesis_set=QueryAwareFakeHypothesisSet(current_items, context_items),
            tracer_config=cfg,
            generation_config=GenerationConfig(max_retries=1),
            prompts=prism_prompts(),
        )

        profile, diagnostics = retrieve_legacy_prediction_profile(
            [
                Turn(user_message="Earlier grant paperwork question", candidates=["A"], chosen="A"),
                Turn(user_message="Plan a dinner party", candidates=["B"], chosen="B"),
            ],
            context,
        )

        self.assertEqual(profile, "retrieved profile")
        self.assertEqual(diagnostics["source"], "legacy_retrieved")
        self.assertEqual(context.hypothesis_set.last_top_k, 30)
        self.assertIn("Earlier grant paperwork question", diagnostics["query"])
        self.assertEqual([item["id"] for item in diagnostics["retrieved"]], ["h2", "h3", "h4"])
        self.assertTrue(diagnostics["used_prior_filter"])

    def test_context_retrieval_uses_model_filter_to_drop_unrelated_items(self):
        current_items = [
            (make_hypothesis(1, "[domain_preference] user likes familiar dinner dishes"), 0.2, 0.9),
            (make_hypothesis(2, "[adaptation_rule] use dinner party planning cues"), 0.2, 0.8),
            (make_hypothesis(3, "[constraint_boundary] avoid overfitting dinner to unrelated history"), 0.2, 0.7),
        ]
        context_items = [
            (make_hypothesis(4, "[domain_preference] user wants grant paperwork checklists"), 0.2, 0.95),
            (make_hypothesis(5, "[adaptation_rule] verify official portals"), 0.2, 0.8),
        ]
        cfg = TracerConfig(
            inference_profile_source="retrieved",
            inference_retrieval_query_mode="current_and_context",
            inference_retrieve_top_k=5,
            inference_retrieve_pool_k=5,
            inference_min_prior=0.1,
            inference_model_filter=True,
            override=OverrideConfig(
                filter_override=TaskOverride(model="main-filter-model"),
                summary_override=TaskOverride(model="summary-model"),
            ),
        )
        context = TracerContext(
            model=FakeFilteringModel(content_ids=["h1"], style_ids=["h2"]),
            hypothesis_set=QueryAwareFakeHypothesisSet(current_items, context_items),
            tracer_config=cfg,
            generation_config=GenerationConfig(max_retries=1),
            prompts=prism_prompts(),
        )

        profile, diagnostics = retrieve_inference_profile(
            [
                Turn(user_message="Earlier grant paperwork question", candidates=["A"], chosen="A"),
                Turn(user_message="Plan a dinner party", candidates=["B"], chosen="B"),
            ],
            context,
        )

        self.assertEqual(profile, "retrieved profile")
        self.assertEqual(len(context.hypothesis_set.queries), 2)
        self.assertNotIn("Earlier grant paperwork question", context.hypothesis_set.queries[0])
        self.assertIn("Earlier grant paperwork question", context.hypothesis_set.queries[1])
        self.assertEqual(diagnostics["model_filter"]["selected_ids"], ["h1", "h2"])
        self.assertEqual(diagnostics["model_filter"]["content_ids"], ["h1"])
        self.assertEqual(diagnostics["model_filter"]["style_ids"], ["h2"])
        self.assertEqual([item["id"] for item in diagnostics["retrieved"]], ["h1", "h2"])
        self.assertEqual(diagnostics["retrieved"][0]["relevance_roles"], ["content"])
        self.assertEqual(diagnostics["retrieved"][1]["relevance_roles"], ["style"])
        required = context.model.schemas[0].model_json_schema()["required"]
        self.assertIn("content_ids", required)
        self.assertIn("style_ids", required)
        self.assertIn("boundary_ids", required)
        self.assertEqual(context.model.kwargs[0]["model"], "main-filter-model")
        self.assertEqual(context.model.kwargs[1]["model"], "summary-model")
        self.assertIn("not the relevance role", context.model.prompts[0])
        self.assertIn("grant paperwork", context.model.prompts[0])
        self.assertNotIn("grant paperwork", context.model.prompts[1])
        self.assertIn("style cues may affect only tone", context.model.prompts[1])
        self.assertIn("familiar dinner dishes", diagnostics["raw_prediction_profile"])
        self.assertIn("dinner party planning cues", diagnostics["raw_prediction_profile"])
        self.assertIn("Relevance roles: style", diagnostics["raw_prediction_profile"])

    def test_personamem_model_filter_keeps_context_items_when_no_ids_selected(self):
        items = [
            (make_hypothesis(1, "[domain_preference] user likes familiar dinner dishes"), 0.2, 0.9),
            (make_hypothesis(2, "[adaptation_rule] use dinner party planning cues"), 0.2, 0.8),
            (make_hypothesis(3, "[ownership_boundary] do not assume pottery belongs to the user"), 0.2, 0.7),
        ]
        cfg = TracerConfig(
            inference_profile_source="retrieved",
            inference_retrieve_top_k=3,
            inference_retrieve_pool_k=3,
            inference_min_prior=0.1,
            inference_model_filter=True,
            inference_empty_profile_on_no_relevance=False,
        )
        context = TracerContext(
            model=FakeFilteringModel(),
            hypothesis_set=FakeHypothesisSet(items),
            tracer_config=cfg,
            generation_config=GenerationConfig(max_retries=1),
            prompts=load_prompt_adapter("personamem_v2"),
        )

        profile, diagnostics = retrieve_inference_profile(
            [Turn(user_message="Plan a dinner party", candidates=["B"], chosen="B")],
            context,
            fallback_profile="FALLBACK",
        )

        self.assertEqual(profile, "retrieved profile")
        self.assertEqual(diagnostics["source"], "retrieved")
        self.assertEqual(diagnostics["model_filter"]["content_ids"], [])
        self.assertEqual(diagnostics["model_filter"]["style_ids"], [])
        self.assertEqual(diagnostics["model_filter"]["boundary_ids"], [])
        self.assertEqual(diagnostics["model_filter"]["context_added_ids"], ["h1", "h2", "h3"])
        self.assertEqual([item["relevance_roles"] for item in diagnostics["retrieved"]], [["context"], ["context"], ["context"]])
        self.assertIn("ownership", diagnostics["raw_prediction_profile"])

    def test_model_filter_soft_caps_are_prompt_guidance_only(self):
        items = [
            (make_hypothesis(i, f"[domain_preference] cue {i}"), 0.2, 1.0 - i * 0.01)
            for i in range(1, 9)
        ]
        cfg = TracerConfig(
            inference_profile_source="retrieved",
            inference_retrieve_top_k=8,
            inference_retrieve_pool_k=8,
            inference_min_prior=0.1,
            inference_model_filter=True,
            inference_content_max_k=4,
            inference_style_max_k=3,
        )
        context = TracerContext(
            model=FakeFilteringModel(
                content_ids=["h1", "h2", "h3", "h4", "h5"],
                style_ids=["h6", "h7", "h8"],
            ),
            hypothesis_set=FakeHypothesisSet(items),
            tracer_config=cfg,
            generation_config=GenerationConfig(max_retries=1),
            prompts=prism_prompts(),
        )

        _, diagnostics = retrieve_inference_profile(
            [Turn(user_message="current", candidates=["A"], chosen="A")],
            context,
        )

        self.assertEqual(diagnostics["model_filter"]["content_ids"], ["h1", "h2", "h3", "h4", "h5"])
        self.assertEqual(diagnostics["model_filter"]["style_ids"], ["h6", "h7", "h8"])
        self.assertIn("at most 4 content_ids and 3 style_ids", context.model.prompts[0])

    def test_model_filter_can_return_empty_profile_when_no_relevant_items(self):
        items = [
            (make_hypothesis(1, "[domain_preference] user wants grant paperwork checklists"), 0.2, 0.9),
            (make_hypothesis(2, "[adaptation_rule] verify official portals"), 0.2, 0.8),
        ]
        cfg = TracerConfig(
            inference_profile_source="retrieved",
            inference_retrieve_top_k=5,
            inference_retrieve_pool_k=5,
            inference_min_prior=0.1,
            inference_model_filter=True,
            inference_empty_profile_on_no_relevance=True,
        )
        context = TracerContext(
            model=FakeFilteringModel(),
            hypothesis_set=FakeHypothesisSet(items),
            tracer_config=cfg,
            generation_config=GenerationConfig(max_retries=1),
            prompts=prism_prompts(),
        )

        profile, diagnostics = retrieve_inference_profile(
            [Turn(user_message="Plan a dinner party", candidates=["B"], chosen="B")],
            context,
            fallback_profile="SHOULD_NOT_USE",
        )

        self.assertEqual(profile, "")
        self.assertEqual(diagnostics["source"], "retrieved_no_relevant_hypotheses")
        self.assertEqual(diagnostics["model_filter"]["selected_ids"], [])
        self.assertEqual(diagnostics["model_filter"]["content_ids"], [])
        self.assertEqual(diagnostics["model_filter"]["style_ids"], [])
        self.assertEqual(diagnostics["model_filter"]["boundary_ids"], [])
        self.assertEqual(len(context.model.prompts), 1)

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

    def test_eval_prefers_prediction_profile_then_inference_profile_and_summary(self):
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
                        "prediction_profile": "USE_PREDICTION_PROFILE",
                        "inference_profile": "DO_NOT_USE_RESPONSE_PROFILE",
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
        self.assertIn("USE_PREDICTION_PROFILE", model.prompts[0])
        self.assertNotIn("DO_NOT_USE_RESPONSE_PROFILE", model.prompts[0])
        self.assertNotIn("DO_NOT_USE_SUMMARY", model.prompts[0])

        inference_model = FakeEvalModel()
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
            eval_model=inference_model,
            eval_cfg=GenerationConfig(max_retries=1),
            embed_cfg=None,
            prompts=prism_prompts(),
        )
        self.assertIn("USE_RETRIEVED_PROFILE", inference_model.prompts[0])
        self.assertNotIn("DO_NOT_USE_SUMMARY", inference_model.prompts[0])

        empty_profile_model = FakeEvalModel()
        evaluate_user_record(
            user_data=user,
            record={
                "user": "u1",
                "turns": [
                    {
                        "adapted": {"success": False},
                        "prediction_profile": "",
                        "inference_profile": "DO_NOT_USE_RESPONSE_PROFILE",
                        "summary": "DO_NOT_USE_SUMMARY",
                    },
                    {"adapted": {"success": False}},
                ],
            },
            eval_model=empty_profile_model,
            eval_cfg=GenerationConfig(max_retries=1),
            embed_cfg=None,
            prompts=prism_prompts(),
        )
        self.assertNotIn("DO_NOT_USE_RESPONSE_PROFILE", empty_profile_model.prompts[0])
        self.assertNotIn("DO_NOT_USE_SUMMARY", empty_profile_model.prompts[0])

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
