# llm-eval-lab：LLM Prompt 评估平台（入门骨架）

一个端到端的 LLM 评估 pipeline：**分层测试集 → 多轮批量实验 → LLM-as-judge 严格评分 + pairwise A/B 对比 → 统计报告**。

它对应 AI/LLM Platform Engineer JD 里的每个关键词：

| JD 关键词 | 本项目对应的部分 |
|---|---|
| LLM 结构化输出 | `judge.py` 用 tool calling / JSON 模式让裁判返回结构化分数，并做本地范围校验 |
| 实验平台 | `config.yaml` 驱动的 模型 × prompt 版本 A/B 实验 |
| 数据管道 | JSONL 数据集 → 批量生成 → 评分 → 聚合，三步落盘 |
| 回测评估 | 固定分层测试集 + 绝对打分 + case 级 pairwise sign test = prompt 改动的回归测试 |
| 报告生成 | `report.py` 自动产出带 mean±std、分层得分和显著性判定的 Markdown 报告 |

## 快速开始

```bash
pip install -r requirements.txt

# 没有 API key 也能跑通全流程（自动进入 mock 模式）
python src/run_eval.py   # 步骤 1：批量生成
python src/judge.py      # 步骤 2：LLM 严格评分 + pairwise 对比
python src/report.py     # 步骤 3：生成报告

cat results/report.md
```

默认配置使用智谱 GLM-5.2 生成摘要，并通过本地 CliRelay/OpenAI-compatible endpoint 上的 `gpt-5.5` 做裁判。接入真实模型：

```bash
export ZHIPU_API_KEY=your-api-key
export CLIRELAY_API_KEY=your-clirelay-key
python src/run_eval.py
python src/judge.py
python src/report.py
```

Windows PowerShell：

```powershell
$env:ZHIPU_API_KEY="your-api-key"
$env:CLIRELAY_API_KEY="your-clirelay-key"
.\run_pipeline.ps1
```

CliRelay 地址在 `config.yaml` 的 `openai_base_url` 中配置；如果你的 GPT 5.5 实际模型 ID 不是 `gpt-5.5`，改 `judge_model` 即可。如果改回 Claude 模型，则设置 `ANTHROPIC_API_KEY`。
Claude 支持是可选依赖，需要额外安装 `anthropic`。

注意：默认配置里被评模型是 `glm-5.2`，裁判模型是本地 CliRelay 上的 `gpt-5.5`，这样避免同一个模型评估自己的输出。更严格的正式评测建议定期运行校准集，并保留校准结果。

Pairwise A/B 对比会双向运行：一次 `v1=A, v2=B`，一次 `v2=A, v1=B`。只有两次都选中同一个 prompt 才计为胜出，否则记为平局，用来降低 LLM 裁判的位置偏差。

校准裁判是否过松：

```bash
python src/calibrate_judge.py
```

该脚本会使用 `data/judge_calibration.jsonl` 中的坏摘要样例，检查裁判是否把复制原文、错数字、漏限制条件的输出打得过高。

## 项目结构

```
config.yaml          # 实验配置：模型、prompt 版本、数据集、num_runs
prompts/             # 待对比的 prompt 模板（v1 简陋版 vs v2 精细版）
data/testcases.jsonl # 主测试集：40 条，含 category 分层 + 人工参考摘要
data/judge_calibration.jsonl # 裁判校准集：人工构造的坏摘要
src/
  llm.py             # 模型客户端封装（含 mock 模式）
  run_eval.py        # 步骤 1：批量生成
  judge.py           # 步骤 2：LLM-as-judge 结构化评分 + pairwise
  calibrate_judge.py # 裁判校准
  report.py          # 步骤 3：聚合 + 报告
results/             # 输出：generations.jsonl / scores.jsonl / pairwise.jsonl / report.md
```

## 阅读顺序（学习路径）

1. 先跑通 mock 模式，看 `results/` 里三个文件长什么样
2. 读 `run_eval.py`——理解"配置驱动的批量实验"
3. 读 `judge.py`——重点看严格 rubric、`SCORE_TOOL` 的 1-5 范围约束，以及 pairwise 对比
4. 读 `report.py`——聚合与报告
5. 接上真实 API key，对比 v1/v2 两个 prompt 的分层得分和 case 级 sign test 结果

## 测试集分层

`data/testcases.jsonl` 约 40 条，按 `category` 分层：

| 类别 | 数量 | 目的 |
|---|---:|---|
| `ordinary_news` | 15 | 普通新闻摘要能力 |
| `numeric_dense` | 10 | 数字、百分比、金额和表格事实保真 |
| `caveat_causal` | 10 | 限制条件、相关性/因果陷阱、研究外推风险 |
| `edge_case` | 5 | 超长、极短、表格、混合格式等边界输入 |

报告会按类别输出 `mean±std`，并给出类似“某 prompt 在数字密集型上均分更高”的分层差异提示。

## 统计判定

`config.yaml` 的 `num_runs` 默认为 3。报告会：

- 对绝对分输出 `mean±std`
- 对 pairwise 胜负先按 case 多轮多数决聚合，再执行双侧 sign test，避免伪重复
- 当 p 值不显著时明确写出“无显著差异，不宣布赢家”
- 按 category 分别输出 case 级 pairwise p 值，避免总均分掩盖局部能力差异

## 扩展 Roadmap（做完这些就是一个能写进简历的项目）

**阶段 1 — 巩固基础（1-2 周）**
- 加第 3 个 prompt 版本，自己设计并观察分数变化
- 扩充 `data/judge_calibration.jsonl`，持续检查裁判是否过松

**阶段 2 — 工程化（2-3 周）**
- 并发执行（`asyncio` 或 `concurrent.futures`），记录成本（token 用量 × 单价）
- 给 pipeline 写更多单元测试（mock 模式让这变得容易）

**阶段 3 — 平台化（3-4 周）**
- 结果存 SQLite，支持跨实验的历史对比（"这次改动比上周好还是差？"→ 这就是回测）
- 加人工标注对比：抽样人工打分，计算与 LLM judge 的一致率（判断裁判可不可信）
- 用 Streamlit 做一个简单 Web 界面浏览结果

**阶段 4 — 简历级（可选）**
- 接入 MLflow 做实验追踪
- GitHub Actions：每次 prompt 改动自动跑回归评估，分数下降则 CI 失败
- 写一篇博客讲你从中学到的评估方法论

## 面试时怎么讲这个项目

不要说"我做了个评估工具"，要说：
"我搭了一个 prompt 回归测试平台——任何 prompt 改动都会在固定测试集上自动跑分，用 LLM-as-judge 加人工抽检保证评分可信，报告自动生成。这解决的是团队改 prompt 全靠感觉、上线后才发现变差的问题。"
