import csv
import json
import tempfile
import unittest
from pathlib import Path

from lecturer_feedback.analysis import analyze
from lecturer_feedback.models import AnalysisConfig
from lecturer_feedback.reporting import build_html, write_report

from helpers import FakeEvaluator, make_dataset


class ReportingTests(unittest.TestCase):
    def test_all_artifacts_are_aggregate_and_html_is_offline(self):
        report = analyze(make_dataset(), AnalysisConfig(), evaluator=FakeEvaluator())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = write_report(report, output)
            self.assertEqual(
                {path.name for path in paths},
                {"report.json", "task_metrics.csv", "concept_metrics.csv", "lecturer_report.html"},
            )
            parsed = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(parsed["schema_version"], "1.0")
            with (output / "task_metrics.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
            self.assertNotIn("SECRET_ALPHA", combined)
            self.assertNotIn("private-user", combined)
            html = (output / "lecturer_report.html").read_text(encoding="utf-8")
            self.assertIn("<script>", html)
            self.assertNotIn("https://", html)
            self.assertIn("REPORT=", html)

    def test_html_escapes_script_termination(self):
        report = analyze(make_dataset(), AnalysisConfig(), evaluator=FakeEvaluator())
        report.warnings.append("</script><script>alert('x')</script>")
        html = build_html(report)
        self.assertNotIn("</script><script>alert", html)
        self.assertIn("\\u003c/script>", html)


if __name__ == "__main__":
    unittest.main()
