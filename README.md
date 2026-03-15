# Plan-and-Execute + ReAct 混合 Agent 示例（Python）

这是一个可直接运行的示例工程，展示了如何用 Python 实现 **Plan-and-Execute + ReAct** 混合推理 Agent。

## 1. 环境要求

- Python 版本：3.9+（推荐 3.10+）
- 操作系统：macOS / Linux / Windows 均可

## 2. 安装依赖

```bash
cd plan-and-execute
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt
```

Planner 与 ReAct 执行器默认使用 **通义千问（Qwen）**；也可切换为 **DeepSeek**。

- **Qwen（默认）**：在 `.env` 中配置 `DASHSCOPE_API_KEY=sk-xxx`（阿里云百炼控制台获取）。
- **DeepSeek**：配置 `DEEPSEEK_API_KEY=sk-xxx`，并设置 `LLM_PROVIDER=deepseek` 或在代码中传入 `provider="deepseek"`、`model="deepseek-chat"`。

未配置或调用失败时，会自动回退到内置规则计划，无需 Key 也能运行。

## 3. 目录结构

```text
plan-and-execute/
  ├── requirements.txt
  ├── README.md
  ├── main.py
  └── agent/
      ├── __init__.py
      ├── tools.py
      ├── planner.py
      ├── react_executor.py
      └── plan_and_react_agent.py
```

## 4. 运行示例

在项目根目录执行：

```bash
python main.py
```

然后按照提示输入你的自然语言问题，例如：

```text
请输入你的问题（输入 q 退出）：请对比一下使用向量数据库和直接全文检索在小型项目中的优缺点。
```

程序会输出：

- 自动生成的高层 Plan
- 针对每个 Plan 步骤的 ReAct 推理过程（由 LLM 输出 Thought / Action，执行工具得到 Observation，直至 Final Answer）
- 一个简单的最终总结

## 5. 模型配置（Qwen / DeepSeek）

支持两种提供商，通过 **provider** 或环境变量 **LLM_PROVIDER** 切换：

| 提供商 | 环境变量 | 默认模型 |
|--------|----------|----------|
| **Qwen**（默认） | `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` | `qwen-plus` |
| **DeepSeek** | `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` | 需显式传 `model="deepseek-chat"` |

- Planner / ReActExecutor 构造参数：`api_key`、`base_url`、`provider`（`"qwen"` | `"deepseek"`）、`model`、`use_llm`。
- 使用 DeepSeek：设置 `LLM_PROVIDER=deepseek` 或在代码中 `Planner(provider="deepseek", model="deepseek-chat")`，ReActExecutor 同理。
- `use_llm=False`：禁用 LLM，仅用规则回退（Planner）或简单说明（ReAct）。

**ReAct 执行器**与 Planner 共用同一套 provider / api_key / base_url / model 配置。

## 6. Replan 与增量式重规划（Incremental Replanning）

当某子任务执行失败时，通过 **ReplanPolicy** 决定是否重规划；推荐使用**增量式**重规划，保留已完成成果，仅重塑剩余路径。

- **ReplanPolicy(mode="incremental")**（默认）：保留已成功步骤及结果，仅对「剩余任务」重新生成计划并继续执行；Planner 会收到「原始任务 + 已完成步骤摘要 + 失败步信息」，只输出从失败步起的新步骤。
- **ReplanPolicy(mode="full")**：整份计划重规划，从步骤 0 重新执行全部（不保留此前结果）。
- 使用方式：`PlanAndReActAgent(..., replan_policy=ReplanPolicy(mode="incremental"))`；`main.py` 中已默认启用增量式 Replan。

## 7. 扩展方向

- 在 `tools.py` 中增加真实业务工具，例如：
  - 调用搜索 API、向量数据库；
  - 查询内部服务 / 数据库；
  - 调用计算、翻译等微服务。

## 8. 许可证

你可以自由修改、商用或集成该代码，无需署名。

