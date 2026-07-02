"""人工盲评标注 CLI。

对 generations.jsonl 中同一 case 的两个 prompt 版本输出做人工 pairwise 标注，
用于之后计算"人工 vs LLM 裁判"一致率（见 agreement.py）。

设计要点：
- 分层抽样：按 category 轮转抽取，固定 seed，同样参数下抽样结果可复现
- 盲评：不显示 prompt 版本号；两个摘要的展示顺序按 case 确定性随机翻转，
  防止位置偏差（与 judge.py 的双向 pairwise 是同一思路）
- 标注时不读取、不显示裁判分数，保证独立判断
- 每条标注立即落盘 data/human_labels.jsonl，随时可以 Ctrl+C 中断，下次续标

用法：
    python src/annotate.py --target 40 --seed 42 --run run_001 --annotator yiyang
"""

import argparse
import itertools
import json
import random
from datetime import datetime
from pathlib import Path

import yaml

from io_utils import load_jsonl

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data" / "human_labels.jsonl"


def build_pairs(generations: list[dict], run_id: str) -> list[dict]:
    """同一 (case, model, run) 下的 prompt 版本两两配对，与 judge.py 的配对方式一致。"""
    groups = {}
    for gen in generations:
        if gen.get("run_id") != run_id:
            continue
        groups.setdefault((gen["case_id"], gen["model"]), []).append(gen)

    pairs = []
    for items in groups.values():
        ordered = sorted(items, key=lambda x: x["prompt_version"])
        for left, right in itertools.combinations(ordered, 2):
            pairs.append(
                {
                    "case_id": left["case_id"],
                    "model": left["model"],
                    "run_id": run_id,
                    "category": left.get("category", "uncategorized"),
                    "input_text": left["input_text"],
                    "reference": left.get("reference"),
                    "prompt_a": left["prompt_version"],
                    "prompt_b": right["prompt_version"],
                    "output_a": left["output"],
                    "output_b": right["output"],
                }
            )
    return pairs


def stratified_sample(pairs: list[dict], target: int, seed: int) -> list[dict]:
    """按 category 轮转抽样：每个类别内部先用固定 seed 洗牌，再轮流取，直到凑够 target。"""
    by_category: dict[str, list[dict]] = {}
    for pair in sorted(pairs, key=lambda p: (p["category"], p["case_id"], p["prompt_a"], p["prompt_b"])):
        by_category.setdefault(pair["category"], []).append(pair)

    for category, items in by_category.items():
        random.Random(f"{seed}:{category}").shuffle(items)

    sampled = []
    categories = sorted(by_category)
    index = 0
    while len(sampled) < target and any(by_category[c] for c in categories):
        category = categories[index % len(categories)]
        if by_category[category]:
            sampled.append(by_category[category].pop())
        index += 1
    return sampled


def show_a_first(seed: int, case_id: str) -> bool:
    """展示顺序的确定性随机翻转：同一 seed + case 永远得到同一顺序。"""
    return random.Random(f"{seed}:{case_id}:order").random() < 0.5


def label_key(record: dict) -> tuple:
    return (
        record.get("case_id"),
        record.get("model"),
        record.get("run_id"),
        record.get("prompt_a"),
        record.get("prompt_b"),
        record.get("annotator"),
    )


def ask_choice() -> str:
    while True:
        raw = input("哪个更好？ [1]=摘要一 [2]=摘要二 [t]=难分高下 [s]=跳过 [q]=退出: ").strip().lower()
        if raw in {"1", "2", "t", "s", "q"}:
            return raw
        print("无效输入，请输入 1 / 2 / t / s / q")


def main():
    parser = argparse.ArgumentParser(description="人工盲评标注")
    parser.add_argument("--target", type=int, default=40, help="计划标注的 pair 数量")
    parser.add_argument("--seed", type=int, default=42, help="抽样与展示顺序的随机种子")
    parser.add_argument("--run", default="run_001", help="标注哪一轮的生成结果")
    parser.add_argument("--annotator", default="annotator_01", help="标注者标识")
    args = parser.parse_args()

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    generations = load_jsonl(ROOT / cfg["output_dir"] / "generations.jsonl")
    if not generations:
        raise RuntimeError("generations.jsonl 为空或不存在，请先运行 src/run_eval.py")

    pairs = build_pairs(generations, args.run)
    if not pairs:
        available = sorted({g.get("run_id") for g in generations})
        raise RuntimeError(f"run_id={args.run} 没有可配对的生成结果；现有 run_id: {available}")

    sampled = stratified_sample(pairs, args.target, args.seed)
    done_keys = {label_key(record) for record in load_jsonl(LABELS_PATH)}

    todo = []
    for pair in sampled:
        key = (pair["case_id"], pair["model"], args.run, pair["prompt_a"], pair["prompt_b"], args.annotator)
        if key not in done_keys:
            todo.append(pair)

    print(f"抽样 {len(sampled)} 对（seed={args.seed}），已标 {len(sampled) - len(todo)}，待标 {len(todo)}")
    if not todo:
        print("没有待标注样本。")
        return

    LABELS_PATH.parent.mkdir(exist_ok=True)
    labeled = 0
    with open(LABELS_PATH, "a", encoding="utf-8") as out:
        for i, pair in enumerate(todo, 1):
            a_first = show_a_first(args.seed, pair["case_id"])
            first_version = pair["prompt_a"] if a_first else pair["prompt_b"]
            second_version = pair["prompt_b"] if a_first else pair["prompt_a"]
            first_output = pair["output_a"] if a_first else pair["output_b"]
            second_output = pair["output_b"] if a_first else pair["output_a"]

            print("\n" + "=" * 72)
            print(f"[{i}/{len(todo)}] {pair['category']} | {pair['case_id']}")
            print("-" * 72)
            print(f"【原文】\n{pair['input_text']}")
            print(f"\n【参考摘要】\n{pair['reference'] or '（无）'}")
            print(f"\n【摘要一】\n{first_output}")
            print(f"\n【摘要二】\n{second_output}")
            print("-" * 72)

            choice = ask_choice()
            if choice == "q":
                break
            if choice == "s":
                continue
            chosen_version = {"1": first_version, "2": second_version, "t": "tie"}[choice]
            record = {
                "case_id": pair["case_id"],
                "model": pair["model"],
                "run_id": pair["run_id"],
                "category": pair["category"],
                "prompt_a": pair["prompt_a"],
                "prompt_b": pair["prompt_b"],
                "first_shown": first_version,
                "choice_raw": choice,
                "chosen_version": chosen_version,
                "annotator": args.annotator,
                "seed": args.seed,
                "annotated_at": datetime.now().isoformat(timespec="seconds"),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            labeled += 1

    print(f"\n本次标注 {labeled} 条，累计写入 {LABELS_PATH}")
    print("下一步：python src/agreement.py 计算与裁判的一致率")


if __name__ == "__main__":
    main()
