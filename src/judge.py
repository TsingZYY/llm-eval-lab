"""步骤 2：LLM-as-judge 自动评分。

读取 generations.jsonl，让裁判模型对每条输出做两类判断：
- 严格绝对打分：accuracy / completeness / conciseness，均为 1-5 分
- 同一 case 下 prompt 版本两两 pairwise 对比：更能暴露 A/B 差异

所有结构化输出都经过本地校验，避免 0、10、字符串分数等污染报告。
"""

import itertools
import time
from collections import defaultdict
from pathlib import Path

import yaml

from io_utils import ResumableJsonlWriter, load_jsonl
from llm import LLMClient

ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = ["accuracy", "completeness", "conciseness"]

SCORE_TOOL = {
    "name": "submit_score",
    "description": "提交对摘要质量的严格评分",
    "input_schema": {
        "type": "object",
        "properties": {
            "accuracy": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": (
                    "事实准确性：摘要是否忠于原文且无捏造。"
                    "1=严重失实, 3=有明显事实风险, 5=完全准确。"
                ),
            },
            "completeness": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": (
                    "完整性：是否覆盖关键事实、数字、限制条件和结论。"
                    "1=遗漏大量要点, 3=遗漏一个关键要点, 5=要点齐全。"
                ),
            },
            "conciseness": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": (
                    "简洁性：是否真正压缩信息。原样复制或接近复述原文最高 2 分；"
                    "同时输出多个版本/多段模板化内容最高 3 分；5=接近参考摘要的精炼程度。"
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "一两句话解释扣分点，必须指出最主要的风险或缺陷。",
            },
        },
        "required": ["accuracy", "completeness", "conciseness", "reasoning"],
    },
}

PAIRWISE_TOOL = {
    "name": "submit_pairwise_judgment",
    "description": "提交两个摘要版本的相对质量判断",
    "input_schema": {
        "type": "object",
        "properties": {
            "winner": {
                "type": "string",
                "enum": ["a", "b", "tie"],
                "description": "a=版本 A 更好, b=版本 B 更好, tie=质量几乎无差异",
            },
            "margin": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
                "description": "胜出幅度：0=平局, 1=轻微, 2=明显, 3=压倒性",
            },
            "reasoning": {
                "type": "string",
                "description": "一两句话说明选择依据，优先谈事实错误、遗漏和冗长。",
            },
        },
        "required": ["winner", "margin", "reasoning"],
    },
}

JUDGE_PROMPT = """你是一名非常严格的摘要评估员。请按下面规则评估摘要，不要因为语气流畅就给高分。

评分规则：
- accuracy 只评事实忠实度。任何原文没有的信息、错误数字、错误因果，accuracy 最高 2 分。
- completeness 评是否覆盖关键事实、数字、限制条件和结论。遗漏一个关键限制/数字，最高 3 分。
- conciseness 评压缩质量。复制原文或接近逐句复述，最高 2 分；输出"一句话+要点"等多个版本，最高 3 分。
- 不要默认给 5 分。5 分表示几乎达到参考摘要质量。

【原文】（{input_len} 字）
{input_text}

【参考摘要】（人工理想答案，仅供参照）
{reference}

【待评估摘要】（{output_len} 字，长度约为原文的 {length_ratio:.0%}）
{output}

请按 schema 返回 accuracy、completeness、conciseness 和 reasoning。"""

PAIRWISE_PROMPT = """你是一名严格的 prompt A/B 评估员。请选择哪个摘要更适合作为生产系统输出。

判断优先级：
1. 事实准确，无捏造
2. 关键信息覆盖完整，尤其数字、限制条件、时间和结论
3. 真正简洁，不复制原文，不输出多个冗余版本

只有两者质量几乎不可区分时才选 tie。

【原文】
{input_text}

【参考摘要】
{reference}

【版本 A：prompt {version_a}】
{output_a}

【版本 B：prompt {version_b}】
{output_b}

请按 schema 返回 winner、margin 和 reasoning。"""


def validate_int_range(value, field: str, min_value: int, max_value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} 必须是整数，实际为 {value!r}")
    if not min_value <= value <= max_value:
        raise ValueError(f"{field} 必须在 {min_value}-{max_value}，实际为 {value}")
    return value


def validate_score(score: dict) -> dict:
    for field in DIMENSIONS:
        validate_int_range(score.get(field), field, 1, 5)
    if not isinstance(score.get("reasoning"), str) or not score["reasoning"].strip():
        raise ValueError("reasoning 必须是非空字符串")
    return {
        "accuracy": score["accuracy"],
        "completeness": score["completeness"],
        "conciseness": score["conciseness"],
        "reasoning": score["reasoning"].strip(),
    }


def validate_pairwise(result: dict) -> dict:
    winner = result.get("winner")
    if winner not in {"a", "b", "tie"}:
        raise ValueError(f"winner 必须是 a/b/tie，实际为 {winner!r}")
    margin = validate_int_range(result.get("margin"), "margin", 0, 3)
    if winner == "tie" and margin != 0:
        raise ValueError("winner=tie 时 margin 必须为 0")
    if winner != "tie" and margin == 0:
        raise ValueError("winner 不是 tie 时 margin 必须大于 0")
    if not isinstance(result.get("reasoning"), str) or not result["reasoning"].strip():
        raise ValueError("reasoning 必须是非空字符串")
    return {"winner": winner, "margin": margin, "reasoning": result["reasoning"].strip()}


def call_structured_with_retries(client, prompt, tool, validator, attempts: int = 3) -> dict:
    last_error = None
    retry_prompt = prompt
    for attempt in range(1, attempts + 1):
        try:
            return validator(client.complete_with_tool(retry_prompt, tool))
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            retry_prompt = (
                prompt
                + "\n\n上一次返回不符合 schema 或本地校验失败。"
                + f"错误：{exc}。请严格返回合法 JSON 对象。"
            )
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"结构化裁判调用失败：{last_error}") from last_error


def generation_key(record: dict) -> tuple:
    return (record["case_id"], record["model"], record["prompt_version"], record.get("run_id"))


def load_latest_generations(path: Path) -> list[dict]:
    latest = {}
    for record in load_jsonl(path):
        latest[generation_key(record)] = record
    return list(latest.values())


def score_key(record: dict) -> tuple:
    return (
        record.get("case_id"),
        record.get("run_id"),
        record.get("model"),
        record.get("prompt_version"),
        record.get("generation_run_mode"),
        record.get("judge_model"),
        record.get("judge_run_mode"),
    )


def pairwise_key(record: dict) -> tuple:
    return (
        record.get("case_id"),
        record.get("run_id"),
        record.get("model"),
        record.get("prompt_a"),
        record.get("prompt_b"),
        record.get("generation_run_mode"),
        record.get("judge_model"),
        record.get("judge_run_mode"),
        record.get("protocol"),
    )


def run_pairwise_once(judge, left: dict, right: dict, cfg: dict) -> dict:
    prompt = PAIRWISE_PROMPT.format(
        input_text=left["input_text"],
        reference=left["reference"] or "（无参考答案）",
        version_a=left["prompt_version"],
        output_a=left["output"],
        version_b=right["prompt_version"],
        output_b=right["output"],
    )
    result = call_structured_with_retries(
        judge,
        prompt,
        PAIRWISE_TOOL,
        validate_pairwise,
        attempts=cfg.get("structured_retries", 3),
    )
    winner_version = {
        "a": left["prompt_version"],
        "b": right["prompt_version"],
        "tie": "tie",
    }[result["winner"]]
    return {
        "winner": winner_version,
        "margin": result["margin"],
        "reasoning": result["reasoning"],
        "a_prompt": left["prompt_version"],
        "b_prompt": right["prompt_version"],
    }


def combine_bidirectional_pairwise(forward: dict, reverse: dict) -> dict:
    if forward["winner"] != "tie" and forward["winner"] == reverse["winner"]:
        return {
            "winner": forward["winner"],
            "margin": min(forward["margin"], reverse["margin"]),
            "reasoning": (
                "双向一致。"
                f"正向：{forward['reasoning']} "
                f"反向：{reverse['reasoning']}"
            ),
        }
    return {
        "winner": "tie",
        "margin": 0,
        "reasoning": (
            "双向对比不一致或至少一次为平局，按位置偏差防护规则记为 tie。"
            f"正向 winner={forward['winner']}：{forward['reasoning']} "
            f"反向 winner={reverse['winner']}：{reverse['reasoning']}"
        ),
    }


def main():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = ROOT / cfg["output_dir"]

    generations = load_latest_generations(out_dir / "generations.jsonl")
    if not generations:
        raise RuntimeError(f"{out_dir / 'generations.jsonl'} 为空或不存在，请先运行 src/run_eval.py")
    judge = LLMClient(
        cfg["judge_model"],
        max_retries=cfg.get("max_retries", 3),
        base_url=cfg.get("openai_base_url"),
    )
    if judge.mock:
        if judge.provider == "zhipu":
            key_name = "ZHIPU_API_KEY"
        elif judge.provider == "openai_compatible":
            key_name = "CLIRELAY_API_KEY/OPENAI_API_KEY"
        else:
            key_name = "ANTHROPIC_API_KEY"
        print(f"[warn] 未检测到 {key_name}，裁判 {cfg['judge_model']} 使用 mock 模式")
    judge_run_mode = "mock" if judge.mock else "real"

    score_writer = ResumableJsonlWriter(out_dir / "scores.jsonl", score_key)
    valid_score_keys = {
        (
            gen["case_id"],
            gen.get("run_id"),
            gen["model"],
            gen["prompt_version"],
            gen.get("run_mode"),
            cfg["judge_model"],
            judge_run_mode,
        )
        for gen in generations
    }
    score_writer.retain_keys(valid_score_keys)
    for i, gen in enumerate(generations, 1):
        record_key = (
            gen["case_id"],
            gen.get("run_id"),
            gen["model"],
            gen["prompt_version"],
            gen.get("run_mode"),
            cfg["judge_model"],
            judge_run_mode,
        )
        if score_writer.has_key(record_key):
            print(f"[skip {i}/{len(generations)}] {gen['case_id']} | {gen['prompt_version']}")
            continue

        input_text = gen["input_text"]
        output = gen["output"]
        length_ratio = len(output) / max(len(input_text), 1)
        prompt = JUDGE_PROMPT.format(
            input_text=input_text,
            input_len=len(input_text),
            reference=gen["reference"] or "（无参考答案）",
            output=output,
            output_len=len(output),
            length_ratio=length_ratio,
        )
        score = call_structured_with_retries(
            judge,
            prompt,
            SCORE_TOOL,
            validate_score,
            attempts=cfg.get("structured_retries", 3),
        )
        record = {
            "case_id": gen["case_id"],
            "category": gen.get("category", "uncategorized"),
            "run_id": gen.get("run_id"),
            "model": gen["model"],
            "prompt_version": gen["prompt_version"],
            "generation_run_mode": gen.get("run_mode"),
            "judge_model": cfg["judge_model"],
            "judge_run_mode": judge_run_mode,
            "latency_s": gen["latency_s"],
            **score,
        }
        score_writer.append(record)
        print(f"[{i}/{len(generations)}] {gen['case_id']} | {gen['prompt_version']} "
              f"-> 准确 {score['accuracy']} / 完整 {score['completeness']} / 简洁 {score['conciseness']}")
    score_writer.commit()

    groups = defaultdict(list)
    for gen in generations:
        groups[(gen["case_id"], gen["model"], gen.get("run_id"))].append(gen)

    pairs = []
    for items in groups.values():
        for left, right in itertools.combinations(sorted(items, key=lambda x: x["prompt_version"]), 2):
            pairs.append((left, right))

    pairwise_protocol = "bidirectional_v1"
    pairwise_writer = ResumableJsonlWriter(out_dir / "pairwise.jsonl", pairwise_key)
    valid_pairwise_keys = {
        (
            left["case_id"],
            left.get("run_id"),
            left["model"],
            left["prompt_version"],
            right["prompt_version"],
            left.get("run_mode"),
            cfg["judge_model"],
            judge_run_mode,
            pairwise_protocol,
        )
        for left, right in pairs
    }
    pairwise_writer.retain_keys(valid_pairwise_keys)
    for i, (left, right) in enumerate(pairs, 1):
        record_key = (
            left["case_id"],
            left.get("run_id"),
            left["model"],
            left["prompt_version"],
            right["prompt_version"],
            left.get("run_mode"),
            cfg["judge_model"],
            judge_run_mode,
            pairwise_protocol,
        )
        if pairwise_writer.has_key(record_key):
            print(f"[skip pairwise {i}/{len(pairs)}] {left['case_id']} | {left['prompt_version']} vs {right['prompt_version']}")
            continue

        forward = run_pairwise_once(judge, left, right, cfg)
        reverse = run_pairwise_once(judge, right, left, cfg)
        result = combine_bidirectional_pairwise(forward, reverse)
        record = {
            "case_id": left["case_id"],
            "category": left.get("category", "uncategorized"),
            "run_id": left.get("run_id"),
            "model": left["model"],
            "prompt_a": left["prompt_version"],
            "prompt_b": right["prompt_version"],
            "generation_run_mode": left.get("run_mode"),
            "winner": result["winner"],
            "margin": result["margin"],
            "reasoning": result["reasoning"],
            "judge_model": cfg["judge_model"],
            "judge_run_mode": judge_run_mode,
            "protocol": pairwise_protocol,
            "forward_winner": forward["winner"],
            "forward_margin": forward["margin"],
            "reverse_winner": reverse["winner"],
            "reverse_margin": reverse["margin"],
        }
        pairwise_writer.append(record)
        print(f"[pairwise {i}/{len(pairs)}] {left['case_id']} | "
              f"{left['prompt_version']} vs {right['prompt_version']} -> {result['winner']}")
    pairwise_writer.commit()

    print(f"\n完成，评分写入 {out_dir / 'scores.jsonl'}")
    print(f"完成，pairwise 判断写入 {out_dir / 'pairwise.jsonl'}")


if __name__ == "__main__":
    main()
