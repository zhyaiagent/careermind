# CareerMind

> AI Agent 驱动的智能求职助手 — 混合 Agent 架构 (ReAct + Plan-Execute) | RAG 混合检索 | MCP 协议 | 浏览器自动化

---

## 项目简介

CareerMind 是一个基于 LangGraph 的 AI Agent 系统，集成了 **RAG 混合检索**、**MCP 协议桥接**、**浏览器自动化** 等能力，帮助求职者完成岗位分析、技能匹配、薪资查询、实时网页搜索等任务。

**核心亮点：**
- **混合 Agent 架构**：ReAct（简单任务）+ Plan-Execute（复杂多步骤任务）
- **混合检索管线**：Vector + BM25 + RRF + BGE-Reranker 四阶段检索
- **混合工具架构**：6 个内置函数（低延迟） + 2 个 MCP 桥接工具（远程服务）
- **浏览器自动化**：Playwright + Edge，支持导航、点击、输入、搜索、截图
- **流式输出**：SSE 协议，token 级实时渲染

---

## 技术架构

```
                         用户 (Streamlit 前端 :8501)
                                    │
                            POST /chat/stream (SSE)
                                    │
                         ┌──────────▼──────────┐
                         │   FastAPI (:8001)    │
                         │                      │
                         │  /chat  /chat/stream │  ← SSE 流式输出
                         │  /upload            │  ← 文档上传入库
                         │  /browser           │  ← 浏览器直控(测试用)
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
                    │         pass │                  │
                    │              ▼                  │
                    │     ChatResponse (SSE token流)  │
                    └──────────────┼─────────────────┘
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
                    │  │ <10ms 延迟 │  │ ┌────────┐ │ │
                    │  │ 进程内执行 │  │ │MCP Srv │ │ │
                    │  └────────────┘  │ │:9020   │ │ │
                    │                  │ │        │ │ │
                    │  ┌────────────┐  │ │browser │ │ │
                    │  │ 浏览器拦截  │  │ │_action │ │ │
                    │  │────────────│  │ │interview│ │ │
                    │  │Playwright  │  │ │_tips   │ │ │
                    │  │Edge 弹窗   │  │ │after_  │ │ │
                    │  │实时操控    │  │ │tax     │ │ │
                    │  └────────────┘  │ │company │ │ │
                    │                  │ │_info   │ │ │
                    │                  │ └────────┘ │ │
                    └──────────────────┴────────────┘ │
                                                     │
                    ┌─────────────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │    记忆层           │
          │                     │
          │  短期: 对话历史(10轮)│
          │  长期: Chroma向量库  │
          │        SQLite薪资库  │
          │        BM25索引     │
          └─────────────────────┘
```

---

## Agent 当前架构详解

### 整体：混合 Agent (Hybrid Agent)

当前采用 **ReAct + Plan-Execute 混合架构**，通过复杂度路由器自动分流：

```
用户输入 → Complexity Router → simple? → ReAct Agent (灵活、低延迟)
                              → complex? → Plan-Execute Agent (结构化、可审计)
```

### 路径 1：ReAct Agent（处理 90% 的请求）

```
Thought: "用户问AI工程师技能，我先查知识库"
  → Action: call search_knowledge_base("AI工程师技能")
  → Observation: [检索到 3 篇相关 JD]
  → Thought: "信息不够，再联网搜一下最新行情"
  → Action: call search_web("2026年AI工程师技能要求")
  → Observation: [网络搜索结果]
  → Thought: "信息够了"
  → Final Answer: 生成回答返回用户
```

**特点**：LLM 每步自主决策，灵活应对各种问题。适合闲聊、单一查询、快速搜索。

### 路径 2：Plan-Execute Agent（处理复杂多步骤任务）

```
用户: "找3个AI Agent岗位，分析每个要求，对比我技能，推荐最佳，给学习路线"

Planner (拆解任务):
  Step 1: search_knowledge_base("AI Agent开发工程师")     ← 搜岗位
  Step 2: analyze_jd(第1个JD)                              ← 分析
  Step 3: analyze_jd(第2个JD)                              ← 分析
  Step 4: analyze_jd(第3个JD)                              ← 分析
  Step 5: query_salary("AI Agent", "北京")                 ← 查薪资
  Step 6: match_skills("用户技能", "岗位要求")              ← 匹配

Executor (逐步执行):
  Step 1 ✓ → Step 2 ✓ → Step 3 ✓ → Step 4 ✓ → Step 5 ✓ → Step 6 ✓

Synthesizer (综合):
  整合所有步骤结果 → 生成结构化回答
```

**特点**：先规划后执行，步骤可审计。适合多岗位对比、全链路分析等复杂场景。

### 工具执行：内置 vs MCP

```
内置函数(6个):
  进程内直接调用 → <10ms 延迟 → 适合高频操作(检索/查询/分析)

MCP 桥接(2个):
  HTTP → JSON-RPC + SSE → MCP Server (独立进程 :9020)
  → 跨进程/跨机器 → 适合独立维护的外部服务(浏览器操控/面试建议等)
```

### 浏览器自动化

```
用户说"用浏览器..."
  → chat.py 关键词拦截 (不经过LLM)
  → Playwright Edge (独立线程, 绕过asyncio)
  → Chromium 弹窗 (headless=False, 用户可见)
  → 支持: navigate / click / type / press / wait / get_content
  → 结果注入消息 → LLM 基于实际页面内容回答
```

### 反思机制 (Self-Reflection)

```
回答生成后 → Reflection 节点评估

评估维度:
  1. 回答是否完整覆盖用户问题？
  2. 是否有事实性错误或遗漏？
  3. 是否充分利用了可用工具？

结果:
  score ≥ 4 → PASS → 直接返回用户
  score < 4 → FAIL → 生成 critique → 带着批评重新执行 (最多2次)

示例:
  用户: "找3个AI岗位对比"
  第1次: 只搜到1个岗位 → score=2, critique="只找到1个岗位，请用不同关键词再搜"
  第2次: 用不同关键词搜索,找到3个 → score=4 → PASS
```

### 四层架构

| 层 | 职责 | 实现 |
|----|------|------|
| **感知层** | 接收输入、理解意图 | Streamlit → HumanMessage → LLM |
| **规划层** | 制定执行策略 | ReAct (自主决策) / Plan-Execute (结构化分解) |
| **工具层** | 执行具体操作 | 8 个 @tool，LLM 自主选择调用 |
| **记忆层** | 存储和检索信息 | 短期(10轮对话历史) + 长期(Chroma向量库 + SQLite薪资库) |

---

## 快速开始

### 环境要求
- Python 3.10+
- Node.js (Playwright MCP 可选)

### 1. 安装依赖
```bash
pip install -r requirements.txt
python -m playwright install chromium  # 浏览器自动化需要
```

### 2. 配置 API Key
```bash
cp .env.example .env
```

编辑 `.env`：
```env
DEEPSEEK_API_KEY=sk-xxx        # 必填，LLM
TAVILY_API_KEY=tvly-xxx        # 可选，联网搜索
EMBEDDING_MODEL=bge-m3         # bge-m3 / onnx / openai
LLM_MODEL=deepseek             # deepseek / qwen / glm
```

### 3. 构建知识库
```bash
python scripts/build_salary_db.py --num-records 200
python scripts/rebuild_kb.py
```

### 4. 启动服务
```bash
# Terminal 1: MCP Server
python scripts/mcp_server.py

# Terminal 2: API
python -m api.main

# Terminal 3: 前端
streamlit run web/app.py
```

访问 http://localhost:8501

### Docker（可选）
```bash
docker compose up -d
```

---

## 项目结构

```
jobsense/
├── core/                       # 核心模块
│   ├── document_processor.py   # PDF/DOCX/TXT 解析
│   ├── chunker.py              # 自适应分块策略
│   ├── embedding.py            # BGE-M3/ONNX/OpenAI
│   ├── vectorstore.py          # Chroma 向量库管理
│   ├── retrieval.py            # 混合检索 (Vector+BM25+RRF+Reranker)
│   ├── generation.py           # LLM 生成(3种Prompt链)
│   ├── hallucination_guard.py  # 4级幻觉检测
│   └── evaluator.py            # RAGAS 评估
│
├── agent/                      # Agent 层
│   ├── graph.py                # 混合架构编排 (ReAct+Plan-Execute)
│   ├── state.py                # AgentState 定义
│   ├── memory.py               # 对话记忆
│   ├── routers/intent_router.py # 意图路由
│   ├── nodes/                  # 各节点实现
│   └── tools/                  # 8 个 LangChain @tool
│       ├── tools.py            # 6个内置工具
│       ├── mcp_bridge.py       # MCP 协议桥接
│       └── web_search.py       # Tavily 联网搜索
│
├── api/                        # FastAPI
│   ├── main.py                 # 入口 + /browser 端点
│   ├── schemas.py              # Pydantic 模型
│   └── routes/chat.py          # 对话路由 + 浏览器拦截
│
├── web/app.py                  # Streamlit UI
│
├── scripts/
│   ├── mcp_server.py           # MCP Server (4 tools)
│   ├── open_browser.py         # 浏览器子进程
│   ├── rebuild_kb.py           # 知识库重建
│   ├── build_salary_db.py      # 薪资数据库构建
│   └── run_evaluation.py       # RAGAS 评估
│
├── tests/                      # 45 个自动化测试
├── data/                       # 数据目录
│   ├── raw/                    # 原始文档
│   ├── processed/              # 处理后数据 + salary.db
│   ├── evaluation/             # 评估测试集
│   └── chroma_db/              # 向量库持久化
│
├── .github/workflows/test.yml  # CI 流水线
├── .pre-commit-config.yaml     # 代码规范检查
├── Dockerfile                  # 容器化
├── docker-compose.yml          # 一键部署
├── requirements.lock           # 锁定依赖
└── .env.example               # 配置模板
```

---

## 8 个工具

| # | 工具 | 类型 | 功能 |
|---|------|------|------|
| 1 | `search_knowledge_base` | 内置 | 搜索本地知识库(JD/报告/文档) |
| 2 | `search_web` | 内置 | Tavily 联网搜索 |
| 3 | `query_salary` | 内置 | SQLite 薪资查询 |
| 4 | `analyze_jd` | 内置 | 结构化 JD 分析 |
| 5 | `match_skills` | 内置 | 技能 vs 岗位匹配 |
| 6 | `calendar_tool` | 内置 | 日期/星期/倒计时 |
| 7 | `call_mcp_tool` | MCP | 调远程 MCP 服务 |
| 8 | `list_mcp_services` | MCP | 列出可用 MCP 服务 |

### MCP Server 提供的 4 个远程工具

| 工具 | 功能 |
|------|------|
| `browser_action` | 浏览器多步操控 (navigate/click/type/press/wait/get_content) |
| `get_interview_tips` | 面试建议 |
| `calculate_after_tax` | 税后薪资计算 |
| `get_company_info` | 公司信息查询 |

---

## 工程化说明

| 能力 | 实现 |
|------|------|
| 测试 | 45 个 pytest，覆盖检索/生成/Agent/工具/幻觉检测 |
| CI | GitHub Actions，push 自动跑测试 (Python 3.10/3.11) |
| 代码规范 | pre-commit: ruff 格式化 + YAML/JSON 检查 |
| 依赖管理 | requirements.lock 精确锁定版本 |
| 配置分离 | .env + config.py |
| 容器化 | Dockerfile + docker-compose.yml + .dockerignore |
| 日志 | logging 模块，按级别输出 |
| 错误处理 | 工具层 try-except + API 层全局异常捕获 |
| 模块化 | 6 层分离 (core/agent/api/web/scripts/tests) |

---

## 使用示例

```
 分析AI算法工程师的JD
 AI Agent开发岗需要什么技能
 北京Python开发 3年经验薪资多少
 我的技能(Python+PyTorch)能匹配大模型岗位吗
 用浏览器搜索2026年AI工程师薪资趋势
 用浏览器打开BOSS直聘搜索AI Agent岗位
 帮我算一下月薪3万在北京税后到手多少
 字节跳动的AI工程师面试要准备什么
 打开DeepSeek官网，问他今天上海天气
```

---

## License

MIT
