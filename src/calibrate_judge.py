"""Run adversarial judge calibration cases.

This is not the main benchmark. It checks whether the judge catches obvious
bad outputs such as copied source text, wrong numbers, and missing caveats.
"""

from datetime import datetime
from pathlib import Path

import yaml

from io_utils import load_jsonl, write_jsonl
from judge import JUDGE_PROMPT, SCORE_TOOL, call_structured_with_retries, validate_score
from llm import LLMClient

ROOT = Path(__file__).resolve().parent.parent


def main():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    judge = LLMClient(
        cfg["judge_model"],
        max_retries=cfg.get("max_retries", 3),
        base_url=cfg.get("openai_base_url"),
    )
    if judge.mock:
        raise RuntimeError("judge calibration 需要真实裁判模型；请先设置对应 API key。")

    failures = []
    results = []
    cases = load_jsonl(ROOT / "data" / "judge_calibration.jsonl")
    for case in cases:
        input_text = case["input_text"]
        output = case["output"]
        prompt = JUDGE_PROMPT.format(
            input_text=input_text,
            input_len=len(input_text),
            reference=case["reference"],
            output=output,
            output_len=len(output),
            length_ratio=len(output) / max(len(input_text), 1),
        )
        score = call_structured_with_retries(
            judge,
            prompt,
            SCORE_TOOL,
            validate_score,
            attempts=cfg.get("structured_retries", 3),
        )
        print(f"{case['id']}: {score}")
        results.append(
            {
                "case_id": case["id"],
                "judge_model": cfg["judge_model"],
                "judge_run_mode": "real",
                "checked_at": datetime.now().isoformat(timespec="seconds"),
                "expected_max": case["expected_max"],
                **score,
            }
        )
        for field, max_allowed in case["expected_max"].items():
            if score[field] > max_allowed:
                failures.append(f"{case['id']} expected {field}<={max_allowed}, got {score[field]}")

    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(exist_ok=True)
    write_jsonl(out_dir / "judge_calibration.jsonl", results)
    if failures:
        raise RuntimeError("judge calibration failed:\n" + "\n".join(failures))
    print(f"judge calibration passed; results written to {out_dir / 'judge_calibration.jsonl'}")


if __name__ == "__main__":
    main()
