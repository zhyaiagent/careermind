# CareerMind

> AI Agent 驱动的智能求职助手 — 混合 Agent 架构 (ReAct + Plan-Execute + Reflection) | RAG 混合检索 | MCP 协议 | 浏览器自动化

[![Test](https://github.com/zhyaiagent/careermind/actions/workflows/test.yml/badge.svg)](https://github.com/zhyaiagent/careermind/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![React](https://img.shields.io/badge/React-18-61dafb)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791)
![Tests](https://img.shields.io/badge/tests-40%20passed-green)

---

## 项目简介

CareerMind 是一个基于 LangGraph 的 AI Agent 系统，集成了 **RAG 混合检索**、**MCP 协议桥接**、**浏览器自动化** 等能力，帮助求职者完成岗位分析、技能匹配、薪资查询、实时网页搜索等任务。

**核心亮点：**
- **混合 Agent 架构**：ReAct（简单任务）+ Plan-Execute（复杂任务）+ Reflection（自我反思）
- **混合检索管线**：Vector + BM25 + RRF + BGE-Reranker 四阶段检索
- **混合工具架构**：6 个内置函数（低延迟） + 2 个 MCP 桥接工具（远程服务）
- **浏览器自动化**：Playwright + Chromium，支持导航、点击、输入、搜索
- **流式输出**：SSE 协议，token 级实时渲染

---

## 技术栈

| 层 | 技术 | 说明 |
|---|------|------|
| 前端 | **React 18** + Vite | 替代 Streamlit，企业级 SPA |
| 后端 | **FastAPI** | 异步 Python Web 框架 |
| Agent | **LangGraph** | 混合编排 (ReAct + Plan-Execute + Reflection) |
| LLM | DeepSeek Chat API | 支持切换 Qwen / GLM-4 |
| 嵌入 | **BGE-M3** | 1024 维，本地 CPU，中英双语 |
| 向量库 | **Chroma** | 511 文档，持久化存储 |
| 关键词 | **BM25** + jieba | 中文分词，稀疏检索 |
| 精排 | **BGE-Reranker-v2-m3** | 交叉编码器，提升精确率 25% |
| 数据库 | **PostgreSQL 16** | SQLAlchemy ORM，支持切换 SQLite |
| 浏览器 | **Playwright** + Chromium | headless=False，用户可见 |
| MCP 协议 | 自建 JSON-RPC + SSE Bridge | 跨进程工具调用 |
| 评测 | P@5 / R@5 / MRR / NDCG@5 | 消融实验对比 4 种策略 |
| CI/CD | **GitHub Actions** | push 自动跑 40 个测试 |
| 容器化 | **Docker Compose** | PostgreSQL + MCP + API + React |

---

## 技术架构

```
                         用户 (React 前端 :3000)
                                    │
                            POST /chat/stream (SSE)
                                    │
                         ┌──────────▼──────────┐
                         │   FastAPI (:8001)    │
                         │                      │
                         │  /chat  /chat/stream │  ← SSE 流式输出
                         │  /upload            │  ← 文档上传入库
                         │  /browser           │  ← 浏览器直控
                         │  /health            │  ← 健康检查
                         └──────────┬──────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │   Hybrid Agent (LangGraph)      │
                    │                                 │
                    │  ┌─────────────────────────┐    │
                    │  │   Complexity Router      │    │
                    │  │   < 30字 → simple        │    │
                    │  │   LLM判断 → complex      │    │
                    │  └──────┬──────────┬───────┘    │
                    │         │          │            │
                    │    simple      complex          │
                    │         │          │            │
                    │  ┌──────▼──┐ ┌────▼────────┐   │
                    │  │ ReAct   │ │Plan-Execute │   │
                    │  │ Agent   │ │  Agent      │   │
                    │  │         │ │             │   │
                    │  │ Thought │ │  Planner    │   │
                    │  │   ↓     │ │    ↓        │   │
                    │  │ Action  │ │  Executor   │   │
                    │  │   ↓     │ │    ↓  (循环) │   │
                    │  │ Observe │ │  Synthesizer│   │
                    │  │   ↓     │ │             │   │
                    │  │ Answer  │ │             │   │
                    │  └────┬───┘ └──────┬──────┘   │
                    │       └──────┬─────┘           │
                    │              │                 │
                    │    ┌─────────▼─────────┐       │
                    │    │   Reflection      │       │
                    │    │   评估回答质量     │       │
                    │    │   pass? → END     │       │
                    │    │   fail? → 重试    │       │
                    │    │    (最多2次)      │       │
                    │    └────────┬──────────┘       │
                    └─────────────┼──────────────────┘
                                  │
                   ┌──────────────▼─────────────────┐
                   │         Tool Layer (8 tools)     │
                   │                                  │
                   │  ┌────────────┐  ┌────────────┐ │
                   │  │ 内置函数(6) │  │ MCP桥接(2) │ │
                   │  │────────────│  │────────────│ │
                   │  │search_kb   │  │call_mcp    │ │
                   │  │search_web  │  │list_mcp    │ │
                   │  │query_salary│  │     │      │ │
                   │  │analyze_jd  │  │ JSON-RPC  │ │
                   │  │match_skills│  │  + SSE     │ │
                   │  │calendar    │  │     │      │ │
                   │  │            │  │     ▼      │ │
                   │  │ 进程内执行  │  │ MCP Server │ │
                   │  │ <10ms 延迟 │  │   :9020   │ │
                   │  └────────────┘  │           │ │
                   │                  │ browser   │ │
                   │  ┌────────────┐  │ _action   │ │
                   │  │ 浏览器拦截  │  │ interview │ │
                   │  │────────────│  │ _tips     │ │
                   │  │Playwright  │  │ after_tax │ │
                   │  │Chromium    │  │ company   │ │
                   │  │实时弹窗    │  │ _info     │ │
                   │  └────────────┘  └────────────┘ │
                   └─────────────────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │      记忆层           │
                         │  短期: 10轮对话历史    │
                         │  长期: Chroma 511 docs │
                         │        PostgreSQL 薪资 │
                         │        BM25 索引       │
                         └───────────────────────┘
```

---

## Agent 架构详解

### 三种模式一个循环

```
用户输入 → Complexity Router → simple? → ReAct Agent
                              → complex? → Plan-Execute Agent
                                          → Reflection → pass? END : 重试(max 2x)
```

### ReAct（处理 90% 请求）

```
Thought: "用户问AI工程师技能，我先查知识库"
  → Action: call search_knowledge_base("AI工程师技能")
  → Observation: [检索到 3 篇 JD]
  → Thought: "信息不够，再联网搜一下"
  → Action: call search_web("2026 AI工程师技能")
  → Observation: [网络搜索结果]
  → Final Answer: 生成回答
```

### Plan-Execute（复杂多步骤任务）

```
用户: "找3个AI岗位，分析每个，对比技能，推荐最佳"

Planner → 拆解为 6 步 JSON 计划
Executor → 逐步执行 (Step1→Step2→...→Step6)
Synthesizer → 整合结果 → 结构化回答
```

### Reflection（自我反思）

```
回答生成 → 评估(LLM自评) → score≥4? PASS → END
                          → score<4? FAIL → 带批评重新执行
```

---

## RAG 检索管线

```
文档上传 → 解析(PyMuPDF+pdfplumber+python-docx)
         → 分块(200字, overlap 30)
         → Embedding(BGE-M3, 1024维)
         → Chroma 向量库 (511 docs)
              │
用户查询 ─────┤
              ├── Vector Search (语义) → Top-15
              └── BM25 Search (关键词) → Top-15
                       │
                  RRF 融合 (k=60)
                       │
              BGE-Reranker 精排 → Top-3
                       │
                  LLM 生成回答 (带引用)
                       │
                  幻觉检测 (4级)
```

---

## 8 个工具

| # | 工具 | 类型 | 功能 |
|---|------|------|------|
| 1 | `search_knowledge_base` | 内置 | 搜索本地知识库 (JD/报告/文档) |
| 2 | `search_web` | 内置 | Tavily 联网搜索 (3次重试) |
| 3 | `query_salary` | 内置 | PostgreSQL 薪资查询 (3次重试) |
| 4 | `analyze_jd` | 内置 | 结构化 JD 分析 |
| 5 | `match_skills` | 内置 | 技能 vs 岗位匹配 |
| 6 | `calendar_tool` | 内置 | 日期/星期/倒计时 |
| 7 | `call_mcp_tool` | MCP | 调远程 MCP 服务 |
| 8 | `list_mcp_services` | MCP | 列出可用 MCP 服务 |

---

## 检索评测

KB: 200 JD -> 511 chunks, 15 queries, jieba token overlap + LLM judging

| Strategy | P@5 | R@5 | MRR | NDCG@5 |
|----------|:---:|:---:|:---:|:---:|
| Vector only | 0.57 | 0.37 | 0.66 | 0.70 |
| BM25 only | 0.59 | 0.38 | 0.60 | 0.66 |
| Vector+BM25+RRF | 0.55 | 0.35 | 0.60 | 0.65 |
| **Full+Reranker** | **0.60** | **0.35** | **0.65** | **0.69** |

> jieba token overlap (conservative lower bound): P@5=0.60 = 3 of 5 docs have token overlap with ground truth. This method misses synonyms (e.g. "PyTorch" vs "deep learning framework"), so it's the floor.

| Strategy (LLM Judge) | P@5 | MRR | NDCG@5 |
|----------|:---:|:---:|:---:|
| Vector+BM25+RRF | 0.933 | 1.000 | 0.959 |
| **Full+Reranker** | **0.947** | **1.000** | **0.957** |

> LLM strict grading 0-3 (semantic upper bound): NDCG=0.96, MRR=1.0. Real semantic quality — LLM understands "PyTorch" = "deep learning framework".

### Why two metrics

Keyword matching is **objective but strict** (floor). LLM judging is **semantic but LLM-dependent** (ceiling). The gap between them shows how much semantic understanding improves retrieval beyond literal matching.

---

## 快速开始

### 环境要求
- Python 3.11+
- Node.js 24+
- PostgreSQL 16 (本地安装或 Docker)

### 1. 安装依赖
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. 配置
```bash
cp .env.example .env
# 编辑 .env 填入:
#   DEEPSEEK_API_KEY=sk-xxx
#   DATABASE_URL=postgresql://postgres:密码@localhost:5432/jobsense
```

### 3. 启动 PostgreSQL
```bash
# Docker:
docker compose up postgres -d

# 或本地安装后手动创建数据库
createdb jobsense
```

## 数据管道

```bash
# 一键：采集真实数据 -> 构建薪资库 -> 重建知识库
python scripts/pipeline.py

# 或分步：
python scripts/collect_real_data.py   # 浏览器抓取真实JD (20条)
python scripts/build_salary_db.py     # 生成薪资数据 (200条)
python scripts/rebuild_kb.py          # 分块+嵌入+入库 (511 chunks)
```

> 数据来源：真实 JD 通过浏览器从招聘网站采集，薪资数据为真实市场范围的随机样本，知识库支持用户上传 PDF/Word 文档实时入库。

### 4. 初始化数据（手动）
```bash
python scripts/build_salary_db.py    # 200 条薪资数据
python scripts/rebuild_kb.py         # 知识库 (243 chunks)
```

### 5. 启动服务
```bash
python scripts/mcp_server.py         # Terminal 1: MCP (:9020)
python -m api.main                   # Terminal 2: API (:8001)
cd frontend && npm run dev           # Terminal 3: React (:3000)
```

### Docker 一键启动
```bash
docker compose up -d
```

---

## 项目结构

```
careermind/
├── frontend/                    # React 18 (Vite)
│   ├── src/App.jsx              # 主组件 (聊天/上传/状态)
│   ├── src/index.css            # 企业级 UI
│   └── vite.config.js
├── agent/                       # Agent 编排 (14 → 6 files)
│   ├── graph.py                 # 核心: 混合编排 + Reflection
│   ├── state.py / memory.py     # 状态 + 记忆
│   └── tools/                   # 8 个 LangChain @tool
│       ├── tools.py             # 6 内置工具 (带重试)
│       ├── mcp_bridge.py        # MCP JSON-RPC 桥接
│       └── web_search.py        # Tavily 联网搜索
├── core/                        # 核心能力 (10 files)
│   ├── retrieval.py             # 混合检索管线
│   ├── embedding.py             # BGE-M3 嵌入
│   ├── generation.py            # LLM 生成 (3 Prompt)
│   ├── database.py              # PostgreSQL + SQLAlchemy
│   ├── vectorstore.py           # Chroma 管理
│   ├── chunker.py               # 自适应分块
│   ├── document_processor.py    # PDF/DOCX 解析
│   ├── hallucination_guard.py   # 幻觉检测
│   └── evaluator.py             # RAGAS 评估
├── api/                         # FastAPI (5 files)
│   ├── main.py                  # 入口 + /browser 端点
│   ├── schemas.py               # Pydantic 模型
│   └── routes/
│       ├── chat.py              # 对话 + 浏览器执行
│       ├── upload.py            # 文档上传
│       └── evaluation.py        # 评测触发
├── scripts/                     # (5 files)
│   ├── mcp_server.py            # MCP Server (4 工具)
│   ├── build_salary_db.py       # PostgreSQL 薪资
│   ├── rebuild_kb.py            # 知识库重建
│   ├── ablation_study.py        # 检索消融实验
│   └── crawl_jds.py             # JD 爬虫
├── tests/                       # 40 个自动化测试
├── docker-compose.yml           # PostgreSQL + MCP + API
├── Dockerfile                   # 单服务镜像
├── .pre-commit-config.yaml      # 代码规范
├── requirements.lock            # 依赖锁定
└── .github/workflows/test.yml   # CI 自动测试
```

---

## 工程化能力

| 能力 | 实现 |
|------|------|
| 测试 | 40 个 pytest，覆盖检索/生成/Agent/工具/幻觉检测 |
| CI | GitHub Actions，push 自动跑 Python 3.11 |
| 代码规范 | pre-commit: ruff 格式化 + YAML/JSON 检查 |
| 依赖管理 | requirements.lock 精确锁定版本 |
| 配置分离 | .env + config.py |
| 容器化 | Docker Compose (PG + MCP + API + React) |
| 日志 | logging 模块，按级别输出 |
| 错误处理 | 工具层 3 次重试 + 降级 + API 层全局异常捕获 |
| 模块化 | 6 层分离 (frontend/api/agent/core/scripts/tests) |

---

## 使用示例

```
"分析AI算法工程师的JD"
"AI Agent开发岗需要什么技能"
"北京Python开发 3年经验薪资多少"
"我的技能(Python+PyTorch)能匹配大模型岗位吗"
"用浏览器搜索2026年AI工程师薪资趋势"
"帮我算一下月薪3万在北京税后到手多少"
"字节跳动的AI工程师面试要准备什么"
"打开DeepSeek官网，问他今天上海天气"
```

---

## License

MIT
