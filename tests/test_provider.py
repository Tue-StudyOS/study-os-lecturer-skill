import sys
import types
import unittest

from lecturer_feedback.models import AnalysisConfig, Task, TaskRubric
from lecturer_feedback.providers import OpenAICompatibleEvaluator, _bound_transcript


class Usage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class Response:
    usage = Usage()

    def __init__(self, content):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


class ProviderTests(unittest.TestCase):
    def test_long_transcript_keeps_start_and_end_within_limit(self):
        transcript = [
            {"role": "user", "content": "A" * 3000},
            {"role": "assistant", "content": "middle" * 1000},
            {"role": "user", "content": "Z" * 3000},
        ]
        result = _bound_transcript(transcript, 2000)
        self.assertTrue(result[0]["content"].startswith("A"))
        self.assertTrue(result[-1]["content"].endswith("Z"))
        self.assertEqual(result[1]["role"], "omission_notice")
        self.assertLessEqual(sum(len(item["content"]) for item in result), 2000)

    def test_openrouter_sets_privacy_and_structured_output(self):
        captured = {}
        rubric = TaskRubric(expected_reasoning="Reasoning", concepts=[], common_misconceptions=[])

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return Response(rubric.model_dump_json())

        class Client:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.chat = types.SimpleNamespace(completions=Completions())

        previous = sys.modules.get("openai")
        sys.modules["openai"] = types.SimpleNamespace(OpenAI=Client)
        try:
            evaluator = OpenAICompatibleEvaluator(
                AnalysisConfig(provider="openrouter", api_key="not-a-real-key", max_retries=0)
            )
            evaluator.build_rubric(Task(id="t", title="T", question="Q"))
        finally:
            if previous is None:
                del sys.modules["openai"]
            else:
                sys.modules["openai"] = previous

        self.assertEqual(captured["client"]["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(captured["response_format"]["type"], "json_schema")
        self.assertTrue(captured["response_format"]["json_schema"]["strict"])
        self.assertEqual(
            captured["extra_body"]["provider"],
            {"require_parameters": True, "data_collection": "deny", "zdr": True},
        )
        self.assertFalse(captured["store"])
        self.assertEqual(evaluator.usage.total_tokens, 15)


if __name__ == "__main__":
    unittest.main()
