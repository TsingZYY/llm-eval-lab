import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from io_utils import ResumableJsonlWriter, load_jsonl  # noqa: E402


class ResumableJsonlWriterTests(unittest.TestCase):
    def test_commit_publishes_inprogress_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            writer = ResumableJsonlWriter(path, lambda r: (r["id"],))
            writer.append({"id": "a", "value": 1})
            writer.commit()

            self.assertEqual(load_jsonl(path), [{"id": "a", "value": 1}])
            self.assertFalse(path.with_name("out.jsonl.inprogress").exists())

    def test_retain_keys_drops_stale_inprogress_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            work_path = path.with_name("out.jsonl.inprogress")
            work_path.write_text('{"id":"stale"}\n{"id":"keep"}\n', encoding="utf-8")

            writer = ResumableJsonlWriter(path, lambda r: (r["id"],))
            writer.retain_keys({("keep",)})
            writer.commit()

            self.assertEqual(load_jsonl(path), [{"id": "keep"}])


if __name__ == "__main__":
    unittest.main()
