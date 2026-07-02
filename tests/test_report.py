import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from report import (  # noqa: E402
    aggregate_pairwise_by_case,
    case_pairwise_prompt_stats,
    format_mean_std,
    pairwise_prompt_stats,
    significance_label,
    sign_test_p_value,
    unique_values,
)


class ReportTests(unittest.TestCase):
    def test_sign_test_detects_clear_win(self):
        self.assertAlmostEqual(sign_test_p_value(10, 0), 0.001953125)

    def test_sign_test_returns_none_without_non_tie_samples(self):
        self.assertIsNone(sign_test_p_value(0, 0))

    def test_significance_label_does_not_announce_non_significant_winner(self):
        stats = {
            "prompt_a": "v1",
            "prompt_b": "v2",
            "a_wins": 6,
            "b_wins": 4,
            "ties": 30,
            "p_value": sign_test_p_value(6, 4),
        }
        self.assertIn("无显著差异", significance_label(stats))

    def test_pairwise_prompt_stats_counts_ties_separately(self):
        stats = pairwise_prompt_stats(
            [
                {"model": "m", "prompt_a": "v1", "prompt_b": "v2", "winner": "v1"},
                {"model": "m", "prompt_a": "v1", "prompt_b": "v2", "winner": "tie"},
                {"model": "m", "prompt_a": "v1", "prompt_b": "v2", "winner": "v2"},
            ]
        )
        item = stats[("m", "v1", "v2")]
        self.assertEqual(item["a_wins"], 1)
        self.assertEqual(item["b_wins"], 1)
        self.assertEqual(item["ties"], 1)

    def test_aggregate_pairwise_by_case_uses_majority_vote(self):
        cases = aggregate_pairwise_by_case(
            [
                {
                    "case_id": "case_001",
                    "category": "numeric_dense",
                    "model": "m",
                    "prompt_a": "v1",
                    "prompt_b": "v2",
                    "winner": "v2",
                },
                {
                    "case_id": "case_001",
                    "category": "numeric_dense",
                    "model": "m",
                    "prompt_a": "v1",
                    "prompt_b": "v2",
                    "winner": "v1",
                },
                {
                    "case_id": "case_001",
                    "category": "numeric_dense",
                    "model": "m",
                    "prompt_a": "v1",
                    "prompt_b": "v2",
                    "winner": "v2",
                },
            ]
        )
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["winner"], "v2")
        self.assertEqual(cases[0]["run_a_wins"], 1)
        self.assertEqual(cases[0]["run_b_wins"], 2)

    def test_case_pairwise_stats_avoid_pseudoreplication(self):
        stats = case_pairwise_prompt_stats(
            [
                {
                    "case_id": "case_001",
                    "category": "numeric_dense",
                    "model": "m",
                    "prompt_a": "v1",
                    "prompt_b": "v2",
                    "winner": "v2",
                },
                {
                    "case_id": "case_001",
                    "category": "numeric_dense",
                    "model": "m",
                    "prompt_a": "v1",
                    "prompt_b": "v2",
                    "winner": "v2",
                },
                {
                    "case_id": "case_001",
                    "category": "numeric_dense",
                    "model": "m",
                    "prompt_a": "v1",
                    "prompt_b": "v2",
                    "winner": "v2",
                },
            ]
        )
        item = stats[("m", "v1", "v2")]
        self.assertEqual(item["a_wins"], 0)
        self.assertEqual(item["b_wins"], 1)
        self.assertEqual(item["ties"], 0)

    def test_format_mean_std(self):
        self.assertEqual(format_mean_std([1, 2, 3]), "2.00±1.00")

    def test_unique_values_summarizes_modes(self):
        self.assertEqual(
            unique_values(
                [
                    {"generation_run_mode": "mock"},
                    {"generation_run_mode": "real"},
                    {"generation_run_mode": "mock"},
                ],
                "generation_run_mode",
            ),
            "mock, real",
        )


if __name__ == "__main__":
    unittest.main()
