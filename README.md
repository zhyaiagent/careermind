# CareerMind

> AI Agent 驱动的智能求职助手 — ReAct + Plan-Execute + Reflection | RAG 混合检索 | MCP 协议 | 浏览器自动化

---

## 技术栈

| 层 | 技术 |
|---|------|
| 前端 | **React 18** (Vite) |
| 后端 | **FastAPI** (Python) |
| Agent | **LangGraph** (ReAct + Plan-Execute + Reflection) |
| LLM | DeepSeek Chat API |
| 嵌入 | BGE-M3 (1024维, 本地 CPU) |
| 向量库 | Chroma (243 docs) |
| 数据库 | **PostgreSQL** (SQLAlchemy ORM) |
| 浏览器 | Playwright (Chromium) |
| MCP | 自建 JSON-RPC + SSE Bridge |
| 评测 | RAGAS (Faithfulness / Relevancy / Precision / Recall) |
| CI | GitHub Actions (40 tests) |
| 容器 | Docker Compose (PostgreSQL + API + MCP + 前端) |

---

## 启动

```bash
# 1. PostgreSQL (本地安装或 Docker)
docker compose up postgres -d     # 用 Docker
# 或本地安装 https://www.postgresql.org/download/windows/

# 2. 初始化数据 (首次)
python scripts/build_salary_db.py

# 3. 启动服务
python scripts/mcp_server.py      # Terminal 1: MCP Server (:9020)
python -m api.main                # Terminal 2: API (:8001)
cd frontend && npm run dev        # Terminal 3: React (:3000)
```

访问 http://localhost:3000

---

## 项目结构

```
careermind/
├── frontend/                    # React (Vite)
│   ├── src/App.jsx              # 主组件
│   └── src/index.css
├── agent/                       # Agent 编排
│   ├── graph.py                 # 核心: ReAct + Plan-Execute + Reflection
│   ├── state.py / memory.py
│   └── tools/                   # 8 个工具
│       ├── tools.py             # 6 内置 @tool
│       ├── mcp_bridge.py        # MCP JSON-RPC 桥接
│       └── web_search.py        # Tavily 搜索
├── core/                        # 核心能力
│   ├── retrieval.py             # 混合检索 (Vector+BM25+RRF+Reranker)
│   ├── embedding.py             # BGE-M3 嵌入
│   ├── generation.py            # LLM 生成 (3 Prompt)
│   ├── database.py              # PostgreSQL (SQLAlchemy)
│   ├── chunker.py / document_processor.py
│   ├── hallucination_guard.py / evaluator.py
│   └── vectorstore.py
├── api/                         # FastAPI
│   ├── main.py                  # 入口 + /browser
│   └── routes/chat.py           # 对话路由 + 浏览器执行
├── scripts/
│   ├── mcp_server.py            # MCP Server (4 工具)
│   ├── build_salary_db.py       # 薪资数据
│   ├── rebuild_kb.py            # 知识库重建
│   └── ablation_study.py        # 检索消融实验
├── tests/                       # 40 个自动化测试
├── docker-compose.yml           # PostgreSQL + MCP + API + React
├── Dockerfile
├── requirements.txt
└── .github/workflows/test.yml   # CI
```

---

## Agent 架构

```
Router (简单/复杂) → ReAct / Plan-Execute → Reflection → 回答

ReAct:      Thought → Action → Observation → Answer
Plan-Execute: Planner → Executor (loop) → Synthesizer
Reflection:  评估质量 → pass? END : 带批评重试 (max 2x)
```

## RAG 检索

```
文档 → 解析 → 分块(200) → BGE-M3(1024d) → Chroma
查询 → Vector(top15) + BM25(top15) → RRF → Reranker → top3 → LLM
```

## 评测

| 策略 | P@5 | R@5 | MRR | NDCG@5 |
|------|:---:|:---:|:---:|:---:|
| Vector only | 0.43 | 0.33 | 0.46 | 0.35 |
| BM25 only | 0.41 | 0.29 | 0.47 | 0.34 |
| + RRF | 0.43 | 0.34 | 0.45 | 0.34 |
| **+ Reranker** | **0.44** | **0.39** | 0.45 | **0.35** |

*243 docs, 15 queries, keyword-based relevance*

## License

MIT
