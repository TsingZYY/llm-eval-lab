"""步骤 3：生成评估报告。

聚合 scores.jsonl 和 pairwise.jsonl，输出 results/report.md：
- 总体和分层 category 的 mean±std
- case 级 pairwise 胜负及 sign test p 值
- 分差不显著时不宣布赢家
"""

from collections import defaultdict
from datetime import datetime
from math import comb
from pathlib import Path
from statistics import mean, stdev

import yaml

from io_utils import load_jsonl

ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = ["accuracy", "completeness", "conciseness"]
ALPHA = 0.05


def overall_score(record: dict) -> float:
    return mean(record[d] for d in DIMENSIONS)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def format_mean_std(values: list[float]) -> str:
    avg, sd = mean_std(values)
    return f"{avg:.2f}±{sd:.2f}"


def unique_values(records: list[dict], field: str) -> str:
    values = sorted({str(record.get(field, "unknown")) for record in records})
    return ", ".join(values) if values else "unknown"


def sign_test_p_value(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def pairwise_prompt_stats(pairwise: list[dict]) -> dict[tuple, dict]:
    stats = {}
    for item in pairwise:
        key = (item["model"], item["prompt_a"], item["prompt_b"])
        if key not in stats:
            stats[key] = {
                "a_wins": 0,
                "b_wins": 0,
                "ties": 0,
                "prompt_a": item["prompt_a"],
                "prompt_b": item["prompt_b"],
                "model": item["model"],
            }
        if item["winner"] == item["prompt_a"]:
            stats[key]["a_wins"] += 1
        elif item["winner"] == item["prompt_b"]:
            stats[key]["b_wins"] += 1
        else:
            stats[key]["ties"] += 1
    for item in stats.values():
        item["p_value"] = sign_test_p_value(item["a_wins"], item["b_wins"])
    return stats


def pairwise_by_category(pairwise: list[dict]) -> dict[tuple, dict]:
    stats = {}
    for item in pairwise:
        key = (item.get("category", "uncategorized"), item["model"], item["prompt_a"], item["prompt_b"])
        if key not in stats:
            stats[key] = {
                "category": item.get("category", "uncategorized"),
                "model": item["model"],
                "prompt_a": item["prompt_a"],
                "prompt_b": item["prompt_b"],
                "a_wins": 0,
                "b_wins": 0,
                "ties": 0,
            }
        if item["winner"] == item["prompt_a"]:
            stats[key]["a_wins"] += 1
        elif item["winner"] == item["prompt_b"]:
            stats[key]["b_wins"] += 1
        else:
            stats[key]["ties"] += 1
    for item in stats.values():
        item["p_value"] = sign_test_p_value(item["a_wins"], item["b_wins"])
    return stats


def aggregate_pairwise_by_case(pairwise: list[dict]) -> list[dict]:
    """Collapse repeated runs for one case into a single majority-vote result."""

    groups = defaultdict(list)
    for item in pairwise:
        key = (
            item["model"],
            item["prompt_a"],
            item["prompt_b"],
            item.get("category", "uncategorized"),
            item["case_id"],
        )
        groups[key].append(item)

    cases = []
    for (model, prompt_a, prompt_b, category, case_id), items in sorted(groups.items()):
        a_wins = sum(1 for item in items if item["winner"] == prompt_a)
        b_wins = sum(1 for item in items if item["winner"] == prompt_b)
        ties = len(items) - a_wins - b_wins
        counts = {prompt_a: a_wins, prompt_b: b_wins, "tie": ties}
        max_votes = max(counts.values())
        winners = [winner for winner, votes in counts.items() if votes == max_votes]
        winner = winners[0] if len(winners) == 1 else "tie"
        cases.append(
            {
                "case_id": case_id,
                "category": category,
                "model": model,
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "winner": winner,
                "run_count": len(items),
                "run_a_wins": a_wins,
                "run_b_wins": b_wins,
                "run_ties": ties,
            }
        )
    return cases


def case_pairwise_prompt_stats(pairwise: list[dict]) -> dict[tuple, dict]:
    return pairwise_prompt_stats(aggregate_pairwise_by_case(pairwise))


def case_pairwise_by_category(pairwise: list[dict]) -> dict[tuple, dict]:
    return pairwise_by_category(aggregate_pairwise_by_case(pairwise))


def significance_label(stats: dict) -> str:
    p_value = stats.get("p_value")
    if p_value is None:
        return "无有效胜负样本，不宣布赢家"
    if p_value >= ALPHA:
        return f"无显著差异 (p={p_value:.3f})，不宣布赢家"
    winner = stats["prompt_a"] if stats["a_wins"] > stats["b_wins"] else stats["prompt_b"]
    return f"{winner} 显著胜出 (p={p_value:.3f})"


def grouped_scores(scores: list[dict], fields: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for score in scores:
        groups[tuple(score.get(field, "unknown") for field in fields)].append(score)
    return groups


def render_score_table(lines: list[str], groups: dict[tuple, list[dict]], include_category: bool = False) -> list[tuple]:
    if include_category:
        lines += [
            "| 类别 | 模型 | Prompt | 生成模式 | 裁判 | N | 准确性 | 完整性 | 简洁性 | 综合 |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    else:
        lines += [
            "| 模型 | Prompt | 生成模式 | 裁判 | N | 准确性 | 完整性 | 简洁性 | 综合 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]

    summary = []
    for key, items in sorted(groups.items()):
        if include_category:
            category, model, version, generation_mode, judge_model, judge_run_mode = key
        else:
            model, version, generation_mode, judge_model, judge_run_mode = key
            category = None
        judge_label = f"{judge_model} ({judge_run_mode})"
        overall_values = [overall_score(item) for item in items]
        summary.append(
            {
                "category": category,
                "model": model,
                "prompt_version": version,
                "generation_mode": generation_mode,
                "judge_model": judge_model,
                "judge_run_mode": judge_run_mode,
                "overall_mean": mean(overall_values),
                "overall_std": mean_std(overall_values)[1],
                "items": items,
            }
        )
        row = [
            model,
            version,
            generation_mode,
            judge_label,
            str(len(items)),
            format_mean_std([item["accuracy"] for item in items]),
            format_mean_std([item["completeness"] for item in items]),
            format_mean_std([item["conciseness"] for item in items]),
            format_mean_std(overall_values),
        ]
        if include_category:
            row.insert(0, category)
        lines.append("| " + " | ".join(row) + " |")
    return summary


def render_category_deltas(lines: list[str], category_summary: list[dict]) -> None:
    by_category = defaultdict(list)
    for row in category_summary:
        by_category[(row["category"], row["model"], row["generation_mode"], row["judge_model"], row["judge_run_mode"])].append(row)

    lines += [
        "",
        "### 类别差异",
        "",
        "| 类别 | 对比 | 综合分差 | 说明 |",
        "|---|---|---:|---|",
    ]
    for (category, _model, _mode, _judge, _judge_mode), rows in sorted(by_category.items()):
        if len(rows) < 2:
            continue
        ordered = sorted(rows, key=lambda r: r["overall_mean"], reverse=True)
        best, second = ordered[0], ordered[1]
        diff = best["overall_mean"] - second["overall_mean"]
        lines.append(
            f"| {category} | {best['prompt_version']} vs {second['prompt_version']} | "
            f"{diff:.2f} | {best['prompt_version']} 在该层均分更高；是否显著需看 case 级 pairwise sign test |"
        )


def render_significance_conclusion(pairwise: list[dict]) -> list[str]:
    lines = ["## 显著性判定", ""]
    if not pairwise:
        return lines + ["无 pairwise 数据，无法进行 sign test。", ""]

    prompt_stats = case_pairwise_prompt_stats(pairwise)
    lines += [
        "> 主结论按 case 聚合：同一 case 的多轮 pairwise 先做多数决；无唯一多数时记为平局，避免把重复运行当作独立样本。",
        "",
        "| 模型 | 对比 | Case A 胜 | Case B 胜 | Case 平 | sign test p | 判定 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for stats in sorted(prompt_stats.values(), key=lambda x: (x["model"], x["prompt_a"], x["prompt_b"])):
        p_text = "NA" if stats["p_value"] is None else f"{stats['p_value']:.3f}"
        lines.append(
            f"| {stats['model']} | {stats['prompt_a']} vs {stats['prompt_b']} | "
            f"{stats['a_wins']} | {stats['b_wins']} | {stats['ties']} | {p_text} | {significance_label(stats)} |"
        )
    lines.append("")
    if not any(stats.get("p_value") is not None and stats["p_value"] < ALPHA for stats in prompt_stats.values()):
        lines += ["> 统计结论：当前 case 级 pairwise 胜负没有达到显著性阈值，不宣布 prompt 赢家。", ""]
    return lines


def main():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = ROOT / cfg["output_dir"]

    scores = load_jsonl(out_dir / "scores.jsonl")
    pairwise = load_jsonl(out_dir / "pairwise.jsonl")
    if not scores:
        raise RuntimeError("scores.jsonl 为空，无法生成报告")

    lines = [
        "# LLM 评估报告",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M} ｜ 用例数：{len({s['case_id'] for s in scores})} ｜ 总评分条数：{len(scores)} ｜ 轮次：{len({s.get('run_id') for s in scores})}",
        f"生成模式：{unique_values(scores, 'generation_run_mode')} ｜ 裁判模型：{unique_values(scores, 'judge_model')} ｜ 裁判模式：{unique_values(scores, 'judge_run_mode')}",
        "",
        "## 总体均分",
        "",
    ]

    overall_groups = grouped_scores(
        scores,
        ("model", "prompt_version", "generation_run_mode", "judge_model", "judge_run_mode"),
    )
    render_score_table(lines, overall_groups)

    tested_models = set(cfg.get("models", []))
    if cfg.get("judge_model") in tested_models:
        lines += [
            "",
            "> 评估风险：当前裁判模型也在被评估模型列表中，可能存在自我偏好。正式评测建议换成不同且更强的裁判模型，或至少以 pairwise 结果为主。",
        ]
    if "mock" in {s.get("generation_run_mode") for s in scores}:
        lines += [
            "",
            "> 运行提示：本报告包含 mock 生成结果。它只能验证评测链路和裁判严格度，不能代表真实 prompt 质量。",
        ]
    if "mock" in {s.get("judge_run_mode") for s in scores}:
        lines += [
            "",
            "> 运行提示：本报告包含 mock 裁判结果。分数不可用于比较 prompt。",
        ]

    lines += ["", "## 分层得分", ""]
    category_groups = grouped_scores(
        scores,
        ("category", "model", "prompt_version", "generation_run_mode", "judge_model", "judge_run_mode"),
    )
    category_summary = render_score_table(lines, category_groups, include_category=True)
    render_category_deltas(lines, category_summary)

    lines += [""] + render_significance_conclusion(pairwise)

    if pairwise:
        lines += [
            "## Pairwise A/B 对比",
            "",
            "### Case 级分层结果",
            "",
            "| 类别 | 模型 | 对比 | Case A 胜 | Case B 胜 | Case 平 | sign test p | 判定 |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        for stats in sorted(case_pairwise_by_category(pairwise).values(), key=lambda x: (x["category"], x["model"])):
            p_text = "NA" if stats["p_value"] is None else f"{stats['p_value']:.3f}"
            lines.append(
                f"| {stats['category']} | {stats['model']} | {stats['prompt_a']} vs {stats['prompt_b']} | "
                f"{stats['a_wins']} | {stats['b_wins']} | {stats['ties']} | {p_text} | {significance_label(stats)} |"
            )
        lines += [
            "",
            "### 逐轮辅助统计（不作为主显著性结论）",
            "",
            "| 类别 | 模型 | 对比 | Run A 胜 | Run B 胜 | Run 平 | sign test p | 判定 |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        for stats in sorted(pairwise_by_category(pairwise).values(), key=lambda x: (x["category"], x["model"])):
            p_text = "NA" if stats["p_value"] is None else f"{stats['p_value']:.3f}"
            lines.append(
                f"| {stats['category']} | {stats['model']} | {stats['prompt_a']} vs {stats['prompt_b']} | "
                f"{stats['a_wins']} | {stats['b_wins']} | {stats['ties']} | {p_text} | {significance_label(stats)} |"
            )
        lines += ["", "### Pairwise 明细", ""]
        for item in pairwise[:20]:
            lines.append(
                f"- {item.get('category', 'uncategorized')} / {item['case_id']} / {item.get('run_id', 'run_unknown')} ｜ "
                f"{item['model']} ｜ {item['prompt_a']} vs {item['prompt_b']} -> {item['winner']} "
                f"（margin {item['margin']}）：{item['reasoning']}"
            )
        if len(pairwise) > 20:
            lines.append(f"- 其余 {len(pairwise) - 20} 条明细略。")
        lines.append("")

    lines += [
        "## 最差用例 Top 5（优先排查）",
        "",
    ]

    worst = sorted(scores, key=lambda s: sum(s[d] for d in DIMENSIONS))[:5]
    for s in worst:
        total = sum(s[d] for d in DIMENSIONS)
        lines += [
            f"### {s['category']} ｜ {s['case_id']} ｜ {s.get('run_id', 'run_unknown')} ｜ {s['model']} ｜ prompt {s['prompt_version']} ｜ 总分 {total}/15",
            "",
            f"- 准确 {s['accuracy']} / 完整 {s['completeness']} / 简洁 {s['conciseness']}",
            f"- 裁判理由：{s['reasoning']}",
            "",
        ]

    report_path = out_dir / "report.md"
    tmp_path = report_path.with_name(report_path.name + ".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    tmp_path.replace(report_path)
    print(f"报告已生成：{report_path}")


if __name__ == "__main__":
    main()
