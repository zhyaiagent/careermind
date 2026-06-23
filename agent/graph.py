"""
LangGraph Agent — ReAct + Plan-Execute + Self-Reflection.

Flow:
  Router → ReAct → Reflect ──(fail)→ ReAct (retry)
                 → Reflect ──(pass)→ END
  Router → Plan-Execute → Reflect ──(fail)→ Plan-Execute (retry)
                        → Reflect ──(pass)→ END

Max 2 reflection cycles. Critiques feed back as improvement hints.
"""
import json
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from agent.tools.tools import get_tools
from config import MAX_ITERATIONS


# ═══════════════════════════════════════════════════
#  Shared State
# ═══════════════════════════════════════════════════

class HybridAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    route: str                         # "react" | "plan_execute"
    plan: list[dict]                   # [{step, tool, args, result}]
    tool_results: list[dict]           # collected execution results
    final_answer: str
    reflection_count: int              # reflection循环次数
    critique: str                      # 反思批评，用于改进下一次尝试


# ═══════════════════════════════════════════════════
#  System Prompts
# ═══════════════════════════════════════════════════

TOOLS_PROMPT = """
## 可用工具
- `search_knowledge_base`: 搜索本地知识库（岗位JD、行业报告、用户上传文档）
- `search_web`: 联网搜索最新信息
- `query_salary`: 查询薪资数据库（按岗位/城市/经验）
- `analyze_jd`: 结构化分析岗位JD
- `match_skills`: 对比用户技能与岗位要求
- `calendar_tool`: 日历查询
- `call_mcp_tool`: 调用外部MCP服务，包含以下能力:
  · browser_action: 浏览器多步操控。action类型:
    - navigate: 打开网址 (url)
    - click: 点击按钮 (text或selector)
    - type: 输入文字 (text)
    - press: 按键 (key, 如Enter)
    - get_content: 获取页面文字
    - screenshot: 截图看页面
    - wait: 等待 (seconds, reason如"等待用户扫码")
    - search: Bing搜索 (query)
  · get_interview_tips: 面试准备建议
  · calculate_after_tax: 税后薪资计算
  · get_company_info: 公司信息查询
- `list_mcp_services`: 查看可用的MCP服务
"""

REACT_PROMPT = f"""你是 CareerMind 智能求职助手。{TOOLS_PROMPT}

## 死命令
- 当用户说"打开"/"浏览"/"访问"+网址 → **必须且立即调用**:
  call_mcp_tool(server="JobTools", tool_name="browser_action", arguments='{{"action":"navigate","url":"网址"}}')
- 禁止用文字描述网页内容来假装你打开了浏览器
- 禁止说"我已经打开了"但实际没调工具
- 先调工具，再根据返回内容回答
- 用户问"今日"或"今天"的数据 → 当前日期是{__import__('datetime').datetime.now().strftime('%Y年%m月%d日')}，不要用旧日期

## 以下是上次回答的反思批评（如有），请据此改进:
{{critique}}"""

ROUTER_PROMPT = """你是一个复杂度判断器。判断用户问题属于哪一类：

- simple: 单一问题，1-2步即可回答（闲聊、查一个数据、问一个概念）
- complex: 多步骤任务，需要≥3步或多种工具组合（如"找3个岗位→分析每个→对比技能→推荐最佳→给学习路线"）

仅输出 simple 或 complex。

问题: {query}
类型: """

PLANNER_PROMPT = """你是一个任务规划器。根据用户问题，将任务拆解为执行步骤。

输出JSON格式的计划，每步包含:
- step: 步骤编号(从1开始)
- description: 这一步做什么
- tool: 要调用的工具名称
- args: 工具参数(JSON对象)

工具列表:
- call_mcp_tool: server固定"JobTools"。可用tool_name: browser_action / get_interview_tips / calculate_after_tax / get_company_info
- search_knowledge_base: {{"query": "..."}}
- search_web: {{"query": "..."}}
- query_salary: {{"job_title": "...", "city": "...", "experience": "..."}}
- analyze_jd: {{"jd_text": "..."}}
- match_skills: {{"user_skills": "...", "job_requirements": "..."}}
- calendar_tool: {{"action": "today/...", "date": "...", "days": N}}

输出格式:
```json
[
  {{"step": 1, "description": "...", "tool": "search_knowledge_base", "args": {{"query": "..."}}}},
  {{"step": 2, "description": "...", "tool": "analyze_jd", "args": {{"jd_text": "..."}}}}
]
```

原则: 每步只做一件事，最多5步。独立步骤可并行。

用户问题: {query}

## 以下是上次执行后的反思批评（如有），请据此调整计划:
{critique}

计划: """

SYNTHESIZE_PROMPT = """你是一个综合分析师。根据以下各步骤的执行结果，综合回答用户问题。

{steps_summary}

用户问题: {query}

## 以下是上次回答的反思批评（如有），请特别关注并改进:
{critique}

请给出完整、有条理的回答。"""

REFLECTION_PROMPT = """你是一个严格的质量审核员。评估以下回答是否充分满足用户问题。

用户问题: {query}

当前回答: {answer}

评估标准:
1. 回答是否完整覆盖了用户问题？
2. 是否有事实性错误或遗漏？
3. 是否充分利用了可用工具？

请输出JSON:
```json
{{"pass": true/false, "score": 1-5, "critique": "如果pass=false，具体说明缺什么；如果pass=true，留空"}}
```"""


# ═══════════════════════════════════════════════════
#  Node 1: Complexity Router
# ═══════════════════════════════════════════════════

def router_node(state: HybridAgentState, llm) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"route": "react"}

    query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
    if len(query) < 30:
        return {"route": "react"}

    prompt = ROUTER_PROMPT.format(query=query)
    resp = llm.invoke(prompt)
    decision = resp.content.strip().lower() if hasattr(resp, 'content') else "simple"
    route = "plan_execute" if decision == "complex" else "react"
    return {"route": route}


# ═══════════════════════════════════════════════════
#  Node 2: Planner
# ═══════════════════════════════════════════════════

def planner_node(state: HybridAgentState, llm) -> dict:
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""
    critique = state.get("critique", "")

    prompt = PLANNER_PROMPT.format(query=query, critique=critique)
    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, 'content') else str(resp)

    import re
    match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    if match: raw = match.group(1)

    try: plan = json.loads(raw)
    except: plan = [{"step": 1, "description": "直接回答", "tool": None, "args": {}}]

    for step in plan: step["result"] = None
    return {"plan": plan, "tool_results": []}


# ═══════════════════════════════════════════════════
#  Node 3: Executor
# ═══════════════════════════════════════════════════

def make_executor(llm, tool_map):
    def executor_node(state: HybridAgentState) -> dict:
        plan = state.get("plan", [])
        for step in plan:
            if step.get("result") is None:
                break
        else:
            return {}

        tool_name = step.get("tool")
        args = step.get("args", {})
        if tool_name is None or tool_name not in tool_map:
            step["result"] = "无需工具"
            return {"plan": plan}

        try:
            result = tool_map[tool_name].invoke(args)
            step["result"] = str(result)
        except Exception as e:
            step["result"] = f"执行出错: {e}"

        tool_results = state.get("tool_results", [])
        tool_results.append({"step": step["step"], "description": step["description"], "tool": tool_name, "result": step["result"]})
        return {"plan": plan, "tool_results": tool_results}
    return executor_node


# ═══════════════════════════════════════════════════
#  Node 4: Synthesizer
# ═══════════════════════════════════════════════════

def synthesizer_node(state: HybridAgentState, llm) -> dict:
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""
    critique = state.get("critique", "")

    tool_results = state.get("tool_results", [])
    steps_lines = []
    for r in tool_results:
        steps_lines.append(f"## 步骤{r['step']}: {r['description']}\n工具: {r['tool']}\n结果:\n{r['result'][:500]}")
    steps_summary = "\n\n---\n\n".join(steps_lines) if steps_lines else "无步骤结果"

    prompt = SYNTHESIZE_PROMPT.format(steps_summary=steps_summary, query=query, critique=critique)
    resp = llm.invoke(prompt)
    answer = resp.content if hasattr(resp, 'content') else str(resp)
    return {"final_answer": answer}


# ═══════════════════════════════════════════════════
#  Node 5: ReAct Agent
# ═══════════════════════════════════════════════════

def make_react_node(llm, tools):
    from langchain.agents import create_agent
    react_graph = create_agent(model=llm, tools=tools, system_prompt=REACT_PROMPT)

    def node_fn(state: HybridAgentState) -> dict:
        critique = state.get("critique", "")
        prompt = REACT_PROMPT.replace("{critique}", f"\n## 反思批评（必须改进）:\n{critique}" if critique else "")
        # Rebuild agent with critique-aware prompt
        g = create_agent(model=llm, tools=tools, system_prompt=prompt)
        result = g.invoke({"messages": state["messages"]})
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                return {"final_answer": msg.content}
        return {"final_answer": "处理完成"}
    return node_fn


# ═══════════════════════════════════════════════════
#  Node 6: Reflection (shared by both paths)
# ═══════════════════════════════════════════════════

def reflection_node(state: HybridAgentState, llm) -> dict:
    """Evaluate answer quality and decide if retry is needed."""
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""
    answer = state.get("final_answer", "")
    count = state.get("reflection_count", 0)

    # Max 2 reflections
    if count >= 2:
        return {"reflection_count": 0, "critique": ""}

    # Quick heuristic: short answers to complex queries likely insufficient
    if len(query) > 80 and len(answer) < 100:
        return {
            "reflection_count": count + 1,
            "critique": f"回答太短({len(answer)}字)，用户问题较长({len(query)}字)，请更详细地回答。可以尝试调用更多工具获取补充信息。",
        }

    # LLM Reflection
    prompt = REFLECTION_PROMPT.format(query=query, answer=answer[:2000])
    try:
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, 'content') else str(resp)
        import re
        match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        if match: raw = match.group(1)
        result = json.loads(raw)
    except Exception:
        result = {"pass": True, "score": 4, "critique": ""}

    if result.get("pass", True) or result.get("score", 0) >= 4:
        return {"reflection_count": 0, "critique": ""}

    return {
        "reflection_count": count + 1,
        "critique": result.get("critique", "请改进回答质量和完整性。"),
    }


# ═══════════════════════════════════════════════════
#  Conditional Routing
# ═══════════════════════════════════════════════════

def route_after_router(state: HybridAgentState) -> str:
    return state.get("route", "react")

def route_after_executor(state: HybridAgentState) -> str:
    plan = state.get("plan", [])
    all_done = all(s.get("result") is not None for s in plan)
    return "synthesize" if all_done else "executor"

def route_after_reflection(state: HybridAgentState) -> str:
    critique = state.get("critique", "")
    route = state.get("route", "react")
    if critique:
        # Retry with reflection feedback
        return route
    return "accept"


# ═══════════════════════════════════════════════════
#  Graph Builder
# ═══════════════════════════════════════════════════

def build_agent_graph(llm, checkpointer=None):
    tools = get_tools()
    # Normalize: StructuredTool -> use .name, raw function -> use __name__
    normalized = []
    for t in tools:
        if hasattr(t, 'func'):  # StructuredTool -> unwrap to raw function
            normalized.append(t.func)
        else:
            normalized.append(t)
    tool_map = {getattr(t, 'name', None) or t.__name__: t for t in normalized}
    tools = normalized

    graph = StateGraph(HybridAgentState)

    # Nodes
    graph.add_node("router", lambda s: router_node(s, llm))
    graph.add_node("react_agent", make_react_node(llm, tools))
    graph.add_node("planner", lambda s: planner_node(s, llm))
    graph.add_node("executor", make_executor(llm, tool_map))
    graph.add_node("synthesizer", lambda s: synthesizer_node(s, llm))
    graph.add_node("reflection", lambda s: reflection_node(s, llm))

    graph.set_entry_point("router")

    # Router → ReAct or Plan-Execute
    graph.add_conditional_edges("router", route_after_router, {
        "react": "react_agent",
        "plan_execute": "planner",
    })

    # Plan-Execute path
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", route_after_executor, {
        "executor": "executor",
        "synthesize": "synthesizer",
    })

    # Both paths → Reflection → (Accept or Retry)
    graph.add_edge("react_agent", "reflection")
    graph.add_edge("synthesizer", "reflection")

    graph.add_conditional_edges("reflection", route_after_reflection, {
        "react": "react_agent",
        "plan_execute": "planner",
        "accept": END,
    })

    return graph.compile(checkpointer=checkpointer)
