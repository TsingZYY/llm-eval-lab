import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from judge import combine_bidirectional_pairwise, validate_pairwise, validate_score  # noqa: E402


class JudgeValidationTests(unittest.TestCase):
    def test_validate_score_accepts_valid_score(self):
        score = {
            "accuracy": 4,
            "completeness": 3,
            "conciseness": 2,
            "reasoning": "ok",
        }
        self.assertEqual(validate_score(score), score)

    def test_validate_score_rejects_out_of_range_score(self):
        with self.assertRaises(ValueError):
            validate_score(
                {
                    "accuracy": 10,
                    "completeness": 3,
                    "conciseness": 2,
                    "reasoning": "bad",
                }
            )

    def test_validate_score_rejects_non_integer_score(self):
        with self.assertRaises(ValueError):
            validate_score(
                {
                    "accuracy": "5",
                    "completeness": 3,
                    "conciseness": 2,
                    "reasoning": "bad",
                }
            )

    def test_validate_pairwise_rejects_tie_with_margin(self):
        with self.assertRaises(ValueError):
            validate_pairwise({"winner": "tie", "margin": 1, "reasoning": "bad"})

    def test_bidirectional_pairwise_requires_consistent_winner(self):
        forward = {"winner": "v1", "margin": 2, "reasoning": "v1 better"}
        reverse = {"winner": "v2", "margin": 1, "reasoning": "v2 better"}
        result = combine_bidirectional_pairwise(forward, reverse)
        self.assertEqual(result["winner"], "tie")
        self.assertEqual(result["margin"], 0)

    def test_bidirectional_pairwise_keeps_consistent_winner(self):
        forward = {"winner": "v1", "margin": 2, "reasoning": "v1 better"}
        reverse = {"winner": "v1", "margin": 1, "reasoning": "v1 still better"}
        result = combine_bidirectional_pairwise(forward, reverse)
        self.assertEqual(result["winner"], "v1")
        self.assertEqual(result["margin"], 1)


if __name__ == "__main__":
    unittest.main()
