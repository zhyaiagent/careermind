# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

CareerMind（"JobSense"）—— 一个基于 LangGraph 的 AI 智能求职助手。它结合了混合 Agent（ReAct + Plan-Execute + Reflection）、混合 RAG 检索管线（向量 + BM25 + RRF + 重排）、MCP 工具桥接、浏览器自动化（Playwright）以及 JWT 认证，后端为 FastAPI，前端为 React（Vite）。

注意命名不一致：仓库/包目录是 `jobsense`，README/产品名是 **CareerMind**，而内部 `SYSTEM_PROMPT` 称之为 **JobSense**。`config.py` 是唯一的配置来源（加载 `.env`）。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt
cd frontend && npm install

# 测试（使用 mock，无需 PostgreSQL 或 API key）
python -m pytest tests/ -v
python -m pytest tests/test_retrieval.py::TestHybridRetriever::test_xxx -v   # 单个测试

# 代码规范 / 格式化（pre-commit: ruff --fix + ruff-format + 空格/yaml/json 检查）
pre-commit install
pre-commit run --all-files

# 启动服务（两个终端）
python -m api.main                # FastAPI，:8001
cd frontend && npm run dev        # React/Vite，:3000（/api 代理到 :8001）

# 数据管道（交互式；也可分步执行）
python scripts/pipeline.py
python scripts/build_salary_db.py   # 向 PostgreSQL 写入 200 条薪资数据
python scripts/rebuild_kb.py        # 将 JD 分块+嵌入写入 Chroma 与 BM25 索引

# PostgreSQL
docker compose up postgres -d
```

测试刻意采用 mock 实现（`tests/` 中的 `FakeLLM`、`FakeVectorStore`、`FakeEmbeddings`），因此无需数据库或 LLM key 即可运行。CI 在 Python 3.11 上执行 `python -m pytest tests/ -v --tb=line`。

## 架构

分层（自上而下）：`frontend/`（React）→ `api/`（FastAPI）→ `agent/`（LangGraph 编排）→ `core/`（RAG + 数据库 + 嵌入）→ `scripts/`（数据 + MCP 服务）→ `tests/`。

### 混合 Agent（`agent/graph.py`）

`build_agent_graph(llm)` 基于 `HybridAgentState`（messages、route、plan、tool_results、final_answer、reflection_count、critique）编译出一个 LangGraph `StateGraph`：

1. **Router** —— 输入 `<30` 字 → `react`；否则由 LLM 判断 `simple`/`complex`。
2. **ReAct 路径** —— 用完整工具列表和 `REACT_PROMPT` 包装 `langchain.agents.create_agent`。
3. **Plan-Execute 路径** —— `planner`（LLM 输出 JSON 步骤计划）→ `executor`（循环，每次执行一步）→ `synthesizer`（汇总结果生成回答）。
4. **Reflection**（两路径共用）—— 短回答启发式检查 + LLM 自评（`REFLECTION_PROMPT`）；当 `pass=false`/score<4 时带着 `critique` 重新路由，最多重试 2 次。

### 工具（`agent/tools/`）

`tools.py` 定义 6 个内置 `@tool`（search_knowledge_base、search_web、query_salary、analyze_jd、match_skills、calendar_tool）。`mcp_bridge.py` 增加 `call_mcp_tool` + `list_mcp_services`，通过官方 `mcp` 包桥接到外部 MCP 服务。`get_tools()` 返回全部 8 个工具。

**关键注入模式：** 内置工具通过模块级全局变量访问 retriever/generator/web-search，这些依赖在启动时由 `inject_dependencies()`（在 `api/main.py` 中调用）注入。不要指望直接实例化工具就能拿到真实依赖 —— 要通过该函数设置。

### MCP（`agent/tools/mcp_bridge.py`）

`MCP_SERVERS` 注册两个外部服务：`ShuidiRisk`（mcpmarket.cn 企业司法风险/大数据查询，28 个工具，streamable_http）和 `playwright`（微软浏览器自动化，24 个工具，stdio/npx）。客户端用**官方 `mcp` 包**（`ClientSession` + `streamable_http_client`/`sse_client`/`stdio_client`）接通外部 MCP server，通过 `call_mcp_tool` / `list_mcp_services` 暴露给 agent；官方 client 是异步的而 LangChain `@tool` 是同步的，桥接用独立线程 + `asyncio.run()` 规避事件循环冲突。配置支持三种 transport（streamable_http / sse / stdio），`headers` 支持 `${ENV_VAR}` 占位（用于 `Authorization: Bearer` 认证的外部服务）。

### RAG 管线（`core/retrieval.py`）

`HybridRetriever.retrieve()` 执行：Chroma 向量检索（top-15）+ BM25/jieba 关键词检索（top-15）→ RRF 融合（`k=60`）→ BGE 重排（`BAAI/bge-reranker-v2-m3`）→ top-k。运行时嵌入是 BGE-M3（`core/embedding.py`）。注意 `scripts/rebuild_kb.py` 用的是**纯 Python TF-IDF** 嵌入（非 BGE），写入 Chroma 和 `data/processed/jds_chunks.jsonl`（即 `api/main.py` 加载的 BM25 语料）。

### API（`api/`）

`api/main.py::create_app()` 连接所有组件、注入工具依赖、注册路由。端点：`/chat` + `/chat/stream`（SSE 流式输出）、`/upload`、`/evaluation`、`/register` + `/login`（JWT —— 注意：用 query 参数，而非 JSON body）、`/history`、`/health`。对话历史持久化在 PostgreSQL（`conversations` 表），而非 LangGraph checkpointer（图编译时 `checkpointer=None`）。

### 数据库与认证（`core/database.py`、`core/auth.py`）

PostgreSQL + SQLAlchemy（`config.py` 中的 `DATABASE_URL`，默认 `postgresql://jobsense:jobsense123@localhost:5432/jobsense`）。表：`salaries`、`conversations`、`user_memory`、`users`。认证是 JWT + bcrypt（`core/auth.py`），属于近期新增功能。

## 注意事项

- **`requirements.txt` 不完整**，与 `Dockerfile` 相比 —— Dockerfile 额外安装了 `playwright`、`beautifulsoup4`、`torch`、`sentence-transformers>=5.0`，以及 `langchain-chroma`/`langchain-text-splitters`/`langchain-huggingface`。`requirements.txt` 仍列出 `streamlit` 和 `ragas`，但 React 应用并未使用它们。
- **`docker-compose.yml` 的 frontend 服务已过时** —— 它运行 `streamlit run web/app.py`，但仓库中没有 `web/` 目录；真正的前端是 React（`frontend/`）。可靠的路径是 `docker compose up postgres -d`。
- **浏览器自动化走外部 playwright MCP**（stdio/npx，`@playwright/mcp`），Agent 通过 `call_mcp_tool(server="playwright", tool_name="browser_navigate", ...)` 调用。
- `.env` 被 gitignore；API key（`DEEPSEEK_API_KEY`、`TAVILY_API_KEY`、`DASHSCOPE_API_KEY`、`OPENAI_API_KEY`）通过 `config.py` 从 `.env` 读取。
- LLM 选择（`LLM_MODEL`：deepseek / qwen / glm）在 `core/generation.py::GenerationManager._init_llm` 中实现，均通过 `langchain_openai.ChatOpenAI` 配合不同的 `base_url`。
