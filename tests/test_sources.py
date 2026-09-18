import gzip
import json
import tempfile
import unittest
from pathlib import Path

from lecturer_feedback.sources import SourceError, load_beta_log_gzip, load_jsonl, load_sqlite


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
SUPPLIED_EXPORT = PROJECT_ROOT / "exports" / "analysis_export.db"


class SourceTests(unittest.TestCase):
    def test_beta_log_gzip_maps_roles_and_groups_exercises(self):
        payload = {
            "result_count": 2,
            "results": [
                {
                    "beta_exercise_id": 19,
                    "beta_exercise_result_id": 101,
                    "exercise_title": "Lecture",
                    "user": "private-a",
                    "conversation": [
                        {"role": "tutor", "content": "Question"},
                        {"role": "student", "content": "Answer"},
                    ],
                    "trace_history": [
                        {"concept_label": "Bayes", "concept_description": "Conditional probability"}
                    ],
                },
                {
                    "beta_exercise_id": 19,
                    "beta_exercise_result_id": 102,
                    "exercise_title": "Lecture",
                    "user": "private-b",
                    "conversation": [{"role": "student", "content": "Another answer"}],
                    "trace_history": [],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "beta.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)
            dataset = load_beta_log_gzip(path)
        self.assertEqual(len(dataset.tasks), 1)
        self.assertEqual(len(dataset.conversations), 2)
        self.assertEqual(dataset.conversations[0].messages[0].role, "assistant")
        self.assertEqual(dataset.conversations[0].messages[1].role, "user")
        self.assertIn("Bayes", dataset.tasks["beta-exercise-19"].instructor_context)

    @unittest.skipUnless(SUPPLIED_EXPORT.is_file(), "repository export fixture is not included")
    def test_supplied_export_shape_and_canonical_task_context(self):
        dataset = load_sqlite(SUPPLIED_EXPORT)
        self.assertEqual(len(dataset.tasks), 16)
        self.assertEqual(len(dataset.conversations), 21)
        self.assertEqual(
            sum(
                any(message.role == "user" for message in conversation.messages)
                for conversation in dataset.conversations
            ),
            1,
        )
        task = dataset.tasks["task15"]
        self.assertIn("Robert H. Goddard", task.question)
        stored_system = next(
            message.content
            for conversation in dataset.conversations
            if conversation.task_id == "task15"
            for message in conversation.messages
            if message.role == "system"
        )
        self.assertIn("Segelboot", stored_system)
        self.assertNotEqual(stored_system, task.question)

    def test_jsonl_rejects_task_mismatch(self):
        value = {
            "task": {"id": "one", "title": "One", "question": "Q"},
            "conversation": {"id": "c", "task_id": "two", "messages": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SourceError, "does not match"):
                load_jsonl(path)

    def test_jsonl_demo_loads(self):
        dataset = load_jsonl(PACKAGE_ROOT / "examples" / "conversations.jsonl")
        self.assertEqual(len(dataset.tasks), 1)
        self.assertEqual(len(dataset.conversations), 3)
        self.assertEqual(len(dataset.task_ratings), 1)


if __name__ == "__main__":
    unittest.main()
