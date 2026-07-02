"""计算人工标注与 LLM 裁判的 pairwise 一致率。

读取 data/human_labels.jsonl 和 results/pairwise.jsonl，按
(case_id, model, run_id, prompt_a, prompt_b) 对齐（同一份输出、同一对比），输出：

- 总体一致率 + Cohen's kappa（v1/v2/tie 三分类，修正随机一致后的指标）
- 排除双方均判 tie 后的一致率（更严格的口径）
- 按 category 分层一致率（哪一层裁判不可信，一目了然）
- 同一标注者重复标注的自我一致率（作为裁判一致率的天花板参照）

结果写入 results/agreement.md。经验参考：好的 LLM judge 与人工的
pairwise 一致率通常在 70-85%，kappa 0.4-0.6 中等、0.6+ 较好。
"""

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from io_utils import load_jsonl

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data" / "human_labels.jsonl"


def label_key(record: dict) -> tuple:
    return (
        record.get("case_id"),
        record.get("model"),
        record.get("run_id"),
        record.get("prompt_a"),
        record.get("prompt_b"),
    )


def dedupe_latest(labels: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """同一标注者对同一样本的多次标注：主指标取最新一条，重复组单独返回用于自我一致率。"""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in labels:
        groups[label_key(record) + (record.get("annotator"),)].append(record)

    latest, duplicates = [], []
    for records in groups.values():
        ordered = sorted(records, key=lambda r: r.get("annotated_at") or "")
        latest.append(ordered[-1])
        if len(ordered) > 1:
            duplicates.append(ordered)
    return latest, duplicates


def self_consistency(duplicates: list[list[dict]]) -> tuple[int, int]:
    """重复标注组中，首末两次选择一致的组数 / 总组数。"""
    consistent = sum(
        1 for records in duplicates if records[0]["chosen_version"] == records[-1]["chosen_version"]
    )
    return consistent, len(duplicates)


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """pairs: [(human_choice, judge_choice), ...]，三分类（v1/v2/tie）Cohen's kappa。"""
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for h, j in pairs if h == j) / n
    human_counts = Counter(h for h, _ in pairs)
    judge_counts = Counter(j for _, j in pairs)
    classes = set(human_counts) | set(judge_counts)
    pe = sum((human_counts[c] / n) * (judge_counts[c] / n) for c in classes)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def match_labels_to_judge(labels: list[dict], pairwise: list[dict]) -> tuple[list[dict], int]:
    """按 key 对齐人工标注与裁判判断，返回 (匹配列表, 未匹配数)。"""
    judge_by_key = {label_key(item): item for item in pairwise}
    matched, unmatched = [], 0
    for label in labels:
        judge = judge_by_key.get(label_key(label))
        if judge is None:
            unmatched += 1
            continue
        matched.append(
            {
                "category": label.get("category", "uncategorized"),
                "annotator": label.get("annotator", "unknown"),
                "case_id": label["case_id"],
                "human": label["chosen_version"],
                "judge": judge["winner"],
            }
        )
    return matched, unmatched


def summarize(matched: list[dict]) -> dict:
    pairs = [(m["human"], m["judge"]) for m in matched]
    n = len(pairs)
    agree = sum(1 for h, j in pairs if h == j)
    non_tie = [(h, j) for h, j in pairs if h != "tie" and j != "tie"]
    non_tie_agree = sum(1 for h, j in non_tie if h == j)
    return {
        "n": n,
        "agree": agree,
        "agree_rate": agree / n if n else None,
        "kappa": cohen_kappa(pairs),
        "human_ties": sum(1 for h, _ in pairs if h == "tie"),
        "judge_ties": sum(1 for _, j in pairs if j == "tie"),
        "non_tie_n": len(non_tie),
        "non_tie_agree": non_tie_agree,
        "non_tie_rate": non_tie_agree / len(non_tie) if non_tie else None,
    }


def fmt_rate(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


def fmt_kappa(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def main():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = ROOT / cfg["output_dir"]

    labels = load_jsonl(LABELS_PATH)
    pairwise = load_jsonl(out_dir / "pairwise.jsonl")
    if not labels:
        raise RuntimeError(f"{LABELS_PATH} 为空，请先运行 src/annotate.py")
    if not pairwise:
        raise RuntimeError("pairwise.jsonl 为空，请先运行 src/judge.py")

    latest, duplicates = dedupe_latest(labels)
    matched, unmatched = match_labels_to_judge(latest, pairwise)
    if not matched:
        raise RuntimeError(
            "没有任何标注能与裁判判断对齐；请确认标注时的 --run 与 judge 跑的是同一批 generations"
        )

    overall = summarize(matched)

    lines = [
        "# 人工标注 vs LLM 裁判一致率",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M} ｜ 对齐样本：{overall['n']}"
        f" ｜ 未对齐（无对应裁判记录）：{unmatched}",
        "",
        "## 总体",
        "",
        "| 标注者 | N | 一致率 | Cohen's kappa | 排除双方 tie 后一致率 | 人工 tie | 裁判 tie |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    by_annotator = defaultdict(list)
    for m in matched:
        by_annotator[m["annotator"]].append(m)
    for annotator, items in sorted(by_annotator.items()):
        s = summarize(items)
        lines.append(
            f"| {annotator} | {s['n']} | {fmt_rate(s['agree_rate'])} | {fmt_kappa(s['kappa'])} "
            f"| {fmt_rate(s['non_tie_rate'])} ({s['non_tie_agree']}/{s['non_tie_n']}) "
            f"| {s['human_ties']} | {s['judge_ties']} |"
        )
    if len(by_annotator) > 1:
        s = overall
        lines.append(
            f"| （合计） | {s['n']} | {fmt_rate(s['agree_rate'])} | {fmt_kappa(s['kappa'])} "
            f"| {fmt_rate(s['non_tie_rate'])} ({s['non_tie_agree']}/{s['non_tie_n']}) "
            f"| {s['human_ties']} | {s['judge_ties']} |"
        )

    lines += [
        "",
        "## 按类别",
        "",
        "| 类别 | N | 一致率 | Cohen's kappa |",
        "|---|---:|---:|---:|",
    ]
    by_category = defaultdict(list)
    for m in matched:
        by_category[m["category"]].append(m)
    for category, items in sorted(by_category.items()):
        s = summarize(items)
        lines.append(f"| {category} | {s['n']} | {fmt_rate(s['agree_rate'])} | {fmt_kappa(s['kappa'])} |")

    consistent, total_dupes = self_consistency(duplicates)
    if total_dupes:
        lines += [
            "",
            f"## 自我一致率\n",
            f"重复标注 {total_dupes} 组，首末选择一致 {consistent} 组"
            f"（{consistent / total_dupes:.1%}）。这是对裁判一致率的天花板参照：",
            "裁判与人工的一致率不应被期望超过人工与自己的一致率。",
        ]

    disagreements = [m for m in matched if m["human"] != m["judge"]]
    if disagreements:
        lines += ["", "## 分歧明细（优先复核）", ""]
        for m in disagreements:
            lines.append(f"- {m['category']} / {m['case_id']}：人工选 {m['human']}，裁判选 {m['judge']}")

    lines += [
        "",
        "> 参考口径：好的 LLM judge 与人工 pairwise 一致率通常 70-85%；",
        "> kappa 0.4-0.6 中等、0.6+ 较好。某类别一致率明显偏低时，该层的裁判结论应降级处理。",
        "",
    ]

    report_path = out_dir / "agreement.md"
    tmp_path = report_path.with_name(report_path.name + ".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    tmp_path.replace(report_path)

    print(f"对齐 {overall['n']} 条 ｜ 一致率 {fmt_rate(overall['agree_rate'])} "
          f"｜ kappa {fmt_kappa(overall['kappa'])} ｜ 未对齐 {unmatched}")
    print(f"报告已生成：{report_path}")


if __name__ == "__main__":
    main()
