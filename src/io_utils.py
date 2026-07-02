"""Small JSONL helpers for resumable experiment outputs."""

import json
from pathlib import Path
from typing import Callable, Iterable


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] skip invalid JSON line {path.name}:{line_no}")
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    tmp_path = path.with_name(path.name + ".rewrite")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


class ResumableJsonlWriter:
    """Write JSONL through an in-progress file, then atomically publish it."""

    def __init__(self, output_path: Path, key_fn: Callable[[dict], tuple]):
        self.output_path = output_path
        self.work_path = output_path.with_name(output_path.name + ".inprogress")
        self.key_fn = key_fn
        self.output_path.parent.mkdir(exist_ok=True)

        resume_path = self.work_path if self.work_path.exists() else self.output_path
        self.records = load_jsonl(resume_path)
        if self.records:
            write_jsonl(self.work_path, self.records)
            print(f"[resume] loaded {len(self.records)} records from {resume_path.name}")

        self.keys = {key_fn(record) for record in self.records}
        self._file = open(self.work_path, "a", encoding="utf-8")

    def has_key(self, key: tuple) -> bool:
        return key in self.keys

    def retain_keys(self, valid_keys: set[tuple]) -> None:
        self._file.close()
        self.records = [record for record in self.records if self.key_fn(record) in valid_keys]
        write_jsonl(self.work_path, self.records)
        self.keys = {self.key_fn(record) for record in self.records}
        self._file = open(self.work_path, "a", encoding="utf-8")

    def append(self, record: dict) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        self.keys.add(self.key_fn(record))

    def commit(self) -> None:
        self._file.close()
        self.work_path.replace(self.output_path)
