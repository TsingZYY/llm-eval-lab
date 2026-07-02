import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from annotate import build_pairs, show_a_first, stratified_sample  # noqa: E402


def make_gen(case_id, version, category="ordinary_news", run_id="run_001"):
    return {
        "case_id": case_id,
        "model": "glm-5.2",
        "prompt_version": version,
        "run_id": run_id,
        "category": category,
        "input_text": "原文",
        "reference": "参考",
        "output": f"{case_id}-{version}",
    }


class AnnotateTests(unittest.TestCase):
    def test_build_pairs_filters_run_and_pairs_versions(self):
        generations = [
            make_gen("c1", "v1"),
            make_gen("c1", "v2"),
            make_gen("c1", "v1", run_id="run_002"),
        ]
        pairs = build_pairs(generations, "run_001")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["prompt_a"], "v1")
        self.assertEqual(pairs[0]["prompt_b"], "v2")
        self.assertEqual(pairs[0]["output_a"], "c1-v1")

    def test_stratified_sample_is_deterministic_and_covers_categories(self):
        pairs = []
        for category in ["a_cat", "b_cat"]:
            for i in range(10):
                pairs.append(
                    {
                        "case_id": f"{category}_{i}",
                        "category": category,
                        "prompt_a": "v1",
                        "prompt_b": "v2",
                    }
                )
        first = stratified_sample(pairs, 6, seed=42)
        second = stratified_sample(pairs, 6, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        categories = [p["category"] for p in first]
        self.assertEqual(categories.count("a_cat"), 3)
        self.assertEqual(categories.count("b_cat"), 3)

    def test_stratified_sample_caps_at_pool_size(self):
        pairs = [{"case_id": "c1", "category": "a", "prompt_a": "v1", "prompt_b": "v2"}]
        self.assertEqual(len(stratified_sample(pairs, 10, seed=1)), 1)

    def test_show_a_first_is_deterministic(self):
        self.assertEqual(show_a_first(42, "case_x"), show_a_first(42, "case_x"))


if __name__ == "__main__":
    unittest.main()
