import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agreement import (  # noqa: E402
    cohen_kappa,
    dedupe_latest,
    match_labels_to_judge,
    self_consistency,
    summarize,
)


def make_label(case_id, chosen, annotator="a1", annotated_at="2026-07-02T10:00:00"):
    return {
        "case_id": case_id,
        "model": "glm-5.2",
        "run_id": "run_001",
        "category": "ordinary_news",
        "prompt_a": "v1",
        "prompt_b": "v2",
        "chosen_version": chosen,
        "annotator": annotator,
        "annotated_at": annotated_at,
    }


def make_judge(case_id, winner):
    return {
        "case_id": case_id,
        "model": "glm-5.2",
        "run_id": "run_001",
        "prompt_a": "v1",
        "prompt_b": "v2",
        "winner": winner,
    }


class KappaTests(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        pairs = [("v1", "v1"), ("v2", "v2"), ("tie", "tie")]
        self.assertEqual(cohen_kappa(pairs), 1.0)

    def test_chance_level_agreement_is_zero(self):
        pairs = [("v1", "v1"), ("v1", "v2"), ("v2", "v1"), ("v2", "v2")]
        self.assertAlmostEqual(cohen_kappa(pairs), 0.0)

    def test_empty_returns_none(self):
        self.assertIsNone(cohen_kappa([]))


class MatchTests(unittest.TestCase):
    def test_match_joins_on_key_and_counts_unmatched(self):
        labels = [make_label("c1", "v2"), make_label("c_missing", "v1")]
        judges = [make_judge("c1", "v2")]
        matched, unmatched = match_labels_to_judge(labels, judges)
        self.assertEqual(len(matched), 1)
        self.assertEqual(unmatched, 1)
        self.assertEqual(matched[0]["human"], "v2")
        self.assertEqual(matched[0]["judge"], "v2")

    def test_summarize_rates(self):
        matched = [
            {"category": "x", "annotator": "a1", "case_id": "c1", "human": "v2", "judge": "v2"},
            {"category": "x", "annotator": "a1", "case_id": "c2", "human": "v1", "judge": "v2"},
            {"category": "x", "annotator": "a1", "case_id": "c3", "human": "tie", "judge": "tie"},
        ]
        s = summarize(matched)
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["agree"], 2)
        self.assertAlmostEqual(s["agree_rate"], 2 / 3)
        self.assertEqual(s["non_tie_n"], 2)
        self.assertEqual(s["non_tie_agree"], 1)


class DedupeTests(unittest.TestCase):
    def test_latest_label_wins_and_duplicates_reported(self):
        labels = [
            make_label("c1", "v1", annotated_at="2026-07-01T10:00:00"),
            make_label("c1", "v2", annotated_at="2026-07-02T10:00:00"),
            make_label("c2", "v1"),
        ]
        latest, duplicates = dedupe_latest(labels)
        chosen = {r["case_id"]: r["chosen_version"] for r in latest}
        self.assertEqual(chosen["c1"], "v2")
        self.assertEqual(len(duplicates), 1)

        consistent, total = self_consistency(duplicates)
        self.assertEqual((consistent, total), (0, 1))


if __name__ == "__main__":
    unittest.main()
