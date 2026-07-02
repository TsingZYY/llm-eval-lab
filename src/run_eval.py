"""步骤 1：批量运行实验。

对每个测试用例 × 每个模型 × 每个 prompt 版本 生成一条输出，
写入 results/generations.jsonl。

这就是"实验平台 + 数据管道"的最小形态：
配置驱动、批量执行、结果落盘为结构化数据。
"""

import hashlib
import json
import time
from pathlib import Path

import yaml

from io_utils import ResumableJsonlWriter
from llm import LLMClient

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(path: str) -> list[dict]:
    cases = []
    with open(ROOT / path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def main():
    cfg = load_config()
    cases = load_dataset(cfg["dataset"])
    prompts = {
        version: (ROOT / path).read_text(encoding="utf-8")
        for version, path in cfg["prompts"].items()
    }
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "generations.jsonl"

    num_runs = cfg.get("num_runs", 1)
    total = len(cases) * len(cfg["models"]) * len(prompts) * num_runs
    done = 0
    clients = {}
    run_modes = {}
    valid_keys = set()
    for model_name in cfg["models"]:
        client = LLMClient(
            model_name,
            max_retries=cfg.get("max_retries", 3),
            base_url=cfg.get("openai_base_url"),
        )
        clients[model_name] = client
        run_modes[model_name] = "mock" if client.mock else "real"
        for version, template in prompts.items():
            prompt_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
            for run_index in range(1, num_runs + 1):
                run_id = f"run_{run_index:03d}"
                for case in cases:
                    valid_keys.add((case["id"], model_name, version, prompt_hash, run_modes[model_name], run_id))

    writer = ResumableJsonlWriter(
        out_path,
        lambda r: (
            r.get("case_id"),
            r.get("model"),
            r.get("prompt_version"),
            r.get("prompt_hash"),
            r.get("run_mode"),
            r.get("run_id"),
        ),
    )
    writer.retain_keys(valid_keys)
    for model_name in cfg["models"]:
        client = clients[model_name]
        if client.mock:
            if client.provider == "zhipu":
                key_name = "ZHIPU_API_KEY"
            elif client.provider == "openai_compatible":
                key_name = "CLIRELAY_API_KEY/OPENAI_API_KEY"
            else:
                key_name = "ANTHROPIC_API_KEY"
            print(f"[warn] 未检测到 {key_name}，模型 {model_name} 使用 mock 模式")
        run_mode = run_modes[model_name]
        for version, template in prompts.items():
            prompt_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
            for run_index in range(1, num_runs + 1):
                run_id = f"run_{run_index:03d}"
                for case in cases:
                    prompt = template.format(text=case["text"])
                    record_key = (case["id"], model_name, version, prompt_hash, run_mode, run_id)
                    if writer.has_key(record_key):
                        done += 1
                        print(f"[skip {done}/{total}] {case['id']} | {model_name} | prompt {version} | {run_id}")
                        continue
                    t0 = time.time()
                    output = client.complete(prompt, cfg.get("max_tokens", 512))
                    record = {
                        "case_id": case["id"],
                        "category": case.get("category", "uncategorized"),
                        "model": model_name,
                        "prompt_version": version,
                        "prompt_hash": prompt_hash,
                        "run_mode": run_mode,
                        "run_id": run_id,
                        "input_text": case["text"],
                        "reference": case.get("reference"),
                        "output": output,
                        "latency_s": round(time.time() - t0, 2),
                    }
                    writer.append(record)
                    done += 1
                    print(f"[{done}/{total}] {case['id']} | {model_name} | prompt {version} | {run_id}")

    writer.commit()
    print(f"\n完成，结果写入 {out_path}")


if __name__ == "__main__":
    main()
