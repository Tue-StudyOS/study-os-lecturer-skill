import unittest

from lecturer_feedback.analysis import analyze
from lecturer_feedback.models import AnalysisConfig

from helpers import FakeEvaluator, make_dataset


class AnalysisTests(unittest.TestCase):
    def test_aggregate_metrics_and_no_private_output(self):
        report = analyze(make_dataset(), AnalysisConfig(), evaluator=FakeEvaluator())
        task = report.tasks[0]
        self.assertEqual(task.conversation_count, 4)
        self.assertEqual(task.candidate_conversation_count, 3)
        self.assertEqual(task.assessable_count, 2)
        self.assertEqual(task.insufficient_evidence_count, 1)
        self.assertEqual(task.understanding_distribution["secure"].count, 1)
        self.assertEqual(task.understanding_distribution["misconception"].count, 1)
        self.assertEqual(task.misconceptions[0].count, 1)
        self.assertTrue(task.low_evidence)
        serialized = report.model_dump_json()
        for private_value in (
            "private-user-one",
            "private-conversation-one",
            "SECRET_ALPHA",
            "SECRET_BETA",
            "SECRET_GAMMA",
        ):
            self.assertNotIn(private_value, serialized)

    def test_failure_is_sanitized_and_report_is_partial(self):
        report = analyze(make_dataset(), AnalysisConfig(), evaluator=FakeEvaluator(fail=True))
        self.assertFalse(report.run.complete)
        self.assertEqual(len(report.errors), 3)
        serialized = report.model_dump_json()
        self.assertNotIn("SECRET_ALPHA", serialized)
        self.assertNotIn("raw private", serialized)

    def test_system_only_dataset_needs_no_evaluator(self):
        value = make_dataset()
        value.conversations = [value.conversations[-1]]
        report = analyze(value, AnalysisConfig())
        self.assertEqual(report.run.api_usage.calls, 0)
        self.assertEqual(report.cohort.candidate_conversation_count, 0)


if __name__ == "__main__":
    unittest.main()
