import unittest

from core.preprocess import preprocess_candidates
from core.utils import TracerConfig, TracerContext
from data.base import Turn
from model.base import GenerationConfig
from prompt import prism_prompts


class FakePreprocessModel:
    def __init__(self, fail_skip: bool = False):
        self.fail_skip = fail_skip
        self.async_calls = 0
        self.generate_calls = 0

    async def async_generate(self, *args, **kwargs):
        self.async_calls += 1
        if self.fail_skip:
            raise RuntimeError("skip failed")
        return [{"output": {"reason": "has no signal", "skip": True}} for _ in args[0]]

    def generate(self, *args, **kwargs):
        self.generate_calls += 1
        return {
            "output": {
                "reason": "candidate contrast is preference-relevant",
                "dimensions": ["tone"],
                "summarized_candidates": [
                    {"i": 0, "summary": "direct answer"},
                    {"i": 1, "summary": "verbose answer"},
                ],
            }
        }


def make_context(model, allow_skip: bool) -> TracerContext:
    return TracerContext(
        model=model,
        hypothesis_set=None,
        tracer_config=TracerConfig(allow_skip=allow_skip, n_hypotheses=3),
        generation_config=GenerationConfig(max_retries=1),
        prompts=prism_prompts(),
    )


class PreprocessSkipTests(unittest.TestCase):
    def test_allow_skip_false_bypasses_skip_vote(self):
        model = FakePreprocessModel()
        turn = Turn(
            user_message="Help me rewrite this note.",
            candidates=["Use a direct tone.", "Use a verbose tone."],
            chosen="Use a direct tone.",
        )

        structured, status = preprocess_candidates([turn], make_context(model, allow_skip=False))

        self.assertEqual(model.async_calls, 0)
        self.assertEqual(model.generate_calls, 1)
        self.assertFalse(status["skip"])
        self.assertEqual(status["total_votes"], 0)
        self.assertEqual([item["choice"] for item in structured], ["chosen", "rejected"])

    def test_skip_failure_continues_to_preprocess(self):
        model = FakePreprocessModel(fail_skip=True)
        turn = Turn(
            user_message="Help me rewrite this note.",
            candidates=["Use a direct tone.", "Use a verbose tone."],
            chosen="Use a direct tone.",
        )

        structured, status = preprocess_candidates([turn], make_context(model, allow_skip=True))

        self.assertEqual(model.async_calls, 1)
        self.assertEqual(model.generate_calls, 1)
        self.assertFalse(status["skip"])
        self.assertEqual(status["invalid"], 3)
        self.assertEqual([item["choice"] for item in structured], ["chosen", "rejected"])


if __name__ == "__main__":
    unittest.main()
