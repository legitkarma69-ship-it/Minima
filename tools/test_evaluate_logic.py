import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("evaluate_logic.py")
SPEC = importlib.util.spec_from_file_location("evaluate_logic", MODULE_PATH)
evaluate_logic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_logic)

class EvaluationHarnessTests(unittest.TestCase):
    def test_dataset_has_unique_ids_and_expected_count(self):
        dataset = MODULE_PATH.parents[1] / "data" / "evaluation" / "logic.jsonl"
        cases = evaluate_logic.load_cases(dataset)
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({case["id"] for case in cases}), 24)

    def test_chat_input_contains_prompts_but_not_reference_material(self):
        cases = [{
            "id": "T001", "prompt": "Does A imply B?",
            "reference_answer": "SECRET ANSWER", "scoring_notes": "SECRET NOTES",
        }]
        text = evaluate_logic.build_chat_input(cases, 32)
        self.assertIn("Human: Does A imply B?\nAssistant:", text)
        self.assertIn("/test 1.0 1 1.0 1.0 32", text)
        self.assertNotIn("SECRET ANSWER", text)
        self.assertNotIn("SECRET NOTES", text)

    def test_response_parser_preserves_multiline_answers(self):
        output = (
            "CHAT_READY\nTEST: The answer is no.\n"
            "The two conditions do not overlap.\nCHAT_READY\n"
            "TEST: Cannot be determined.\nCHAT_READY\nCHAT_CLOSED\n"
        )
        self.assertEqual(evaluate_logic.parse_test_responses(output), [
            "The answer is no.\nThe two conditions do not overlap.",
            "Cannot be determined.",
        ])

    def test_response_parser_rejects_incomplete_response(self):
        with self.assertRaises(ValueError):
            evaluate_logic.parse_test_responses("TEST: unfinished")

if __name__ == "__main__":
    unittest.main()
