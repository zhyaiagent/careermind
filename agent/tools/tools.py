"""
JobSense Tools — LangChain @tool functions the LLM can call autonomously.

All tools use dependency injection via module-level globals set by main.py.
The LLM decides which tool to call based on the user's question.
"""
import json
import sqlite3
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta

# ── Injected dependencies (set by main.py) ────────
_retriever = None        # HybridRetriever
_generator = None        # GenerationManager
_web_search = None       # WebSearchTool
_salary_db_path = None   # path to salary.db
_embedding_manager = None


def inject_dependencies(*, retriever=None, generator=None, web_search=None,
                         salary_db_path=None, embedding_manager=None):
    """Set module-level dependencies before building the agent."""
    global _retriever, _generator, _web_search, _salary_db_path, _embedding_manager
    _retriever = retriever
    _generator = generator
    _web_search = web_search
    _salary_db_path = salary_db_path
    _embedding_manager = embedding_manager


# ═══════════════════════════════════════════════════
#  Tool Definitions
# ═══════════════════════════════════════════════════

class SearchKBInput(BaseModel):
    query: str = Field(description="搜索关键词，例如'AI工程师的技能要求'")


@tool(args_schema=SearchKBInput)
def search_knowledge_base(query: str) -> str:
    """
    搜索本地知识库，包括岗位JD、行业报告和用户上传的文档。
    当用户询问岗位要求、技能、行业趋势或文档内容时使用此工具。
    """
    if _retriever is None:
        return "知识库未初始化。"
    try:
        docs = _retriever.retrieve(query, top_k=3)
        if not docs:
            return "未找到相关知识库内容。"
        lines = []
        for i, d in enumerate(docs, 1):
            src = d.get("metadata", {}).get("source", "未知")
            score = d.get("relevance_score", 0)
            lines.append(f"[{i}] (来源:{src}, 相关度:{score:.2f})\n{d['content']}")
        return "\n\n---\n\n".join(lines)
    except Exception as e:
        return f"搜索出错: {e}"


class SearchWebInput(BaseModel):
    query: str = Field(description="网络搜索关键词")


@tool(args_schema=SearchWebInput)
def search_web(query: str) -> str:
    """
    联网搜索最新信息，如薪资行情、行业动态、新闻等实时数据。
    当本地知识库无法回答或需要最新信息时使用。
    """
    if _web_search is None:
        return "联网搜索未配置(Tavily API Key未设置)。"
    result = _web_search.search(query)
    return result if result else "未搜索到相关网络结果。"


class QuerySalaryInput(BaseModel):
    job_title: Optional[str] = Field(default=None, description="岗位名称，如'AI工程师'")
    city: Optional[str] = Field(default=None, description="城市，如'北京'")
    experience: Optional[str] = Field(default=None, description="经验，如'3-5年'")


@tool(args_schema=QuerySalaryInput)
def query_salary(
    job_title: Optional[str] = None,
    city: Optional[str] = None,
    experience: Optional[str] = None,
) -> str:
    """
    查询薪资数据库。可按岗位、城市、经验筛选。
    当用户询问薪资、工资、薪酬时使用。
    """
    db_path = _salary_db_path or "data/processed/salary.db"
    try:
        conn = sqlite3.connect(db_path)
        query_parts = ["SELECT * FROM salaries WHERE 1=1"]
        params = []
        if job_title:
            query_parts.append("AND job_title LIKE ?")
            params.append(f"%{job_title}%")
        if city:
            query_parts.append("AND city = ?")
            params.append(city)
        if experience:
            query_parts.append("AND experience LIKE ?")
            params.append(f"%{experience}%")
        query_parts.append("LIMIT 10")

        import pandas as pd
        df = pd.read_sql(" ".join(query_parts), conn, params=params)
        conn.close()

        if df.empty:
            return "未找到匹配的薪资数据。"
        # Only return key columns
        cols = ["job_title", "company_type", "city", "experience",
                "min_salary", "max_salary", "avg_salary", "source"]
        available = [c for c in cols if c in df.columns]
        return df[available].to_markdown(index=False)
    except Exception as e:
        return f"薪资查询出错: {e}"


class AnalyzeJDInput(BaseModel):
    jd_text: str = Field(description="完整的岗位JD文本")


@tool(args_schema=AnalyzeJDInput)
def analyze_jd(jd_text: str) -> str:
    """
    结构化分析岗位JD，提取：岗位名称、核心技能、薪资预估、难度评级、发展路径。
    当用户要求分析某个JD或岗位描述时使用。
    """
    if _generator is None:
        return "分析引擎未初始化。"
    try:
        result = _generator.analyze_jd(jd_text)
        if "raw_response" in result:
            return result["raw_response"]
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"JD分析出错: {e}"


class MatchSkillsInput(BaseModel):
    user_skills: str = Field(description="求职者的技能描述")
    job_requirements: str = Field(description="岗位的技能要求")


@tool(args_schema=MatchSkillsInput)
def match_skills(user_skills: str, job_requirements: str) -> str:
    """
    对比求职者技能与岗位要求，给出匹配度评分、已匹配技能、缺失技能及学习建议。
    当用户询问'我能胜任吗'或'我的技能是否匹配'时使用。
    """
    if _generator is None:
        return "匹配引擎未初始化。"
    try:
        result = _generator.match_skills(user_skills, job_requirements)
        if "raw_response" in result:
            return result["raw_response"]
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"技能匹配出错: {e}"


class CalendarInput(BaseModel):
    action: str = Field(description="操作: 'today'(今天日期), 'weekday'(某天是星期几), 'countdown'(倒数天数), 'after'(N天后日期)")
    date: Optional[str] = Field(default=None, description="日期，格式YYYY-MM-DD，如'2026-06-15'")
    days: Optional[int] = Field(default=None, description="天数，用于countdown/after操作")


@tool(args_schema=CalendarInput)
def calendar_tool(action: str = "today", date: Optional[str] = None, days: Optional[int] = None) -> str:
    """
    日历工具：查询日期、星期几、倒计时、日期推算。
    当用户询问日期、星期、倒计时、哪天之前/之后时使用。
    例: "今天是几号"、"6月15日是星期几"、"距截止日还有几天"、"30天后是几号"
    """
    weekdays_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

    try:
        if action == "today":
            today = datetime.now()
            return f"今天是 {today.strftime('%Y年%m月%d日')} {weekdays_cn[today.weekday()]}（第{today.isocalendar()[1]}周）"

        elif action == "weekday" and date:
            dt = datetime.strptime(date, "%Y-%m-%d")
            return f"{dt.strftime('%Y年%m月%d日')} 是 {weekdays_cn[dt.weekday()]}"

        elif action == "countdown" and date:
            target = datetime.strptime(date, "%Y-%m-%d")
            today = datetime.now()
            delta = (target - today).days
            if delta > 0:
                wd = weekdays_cn[target.weekday()]
                return f"距 {target.strftime('%Y年%m月%d日')}（{wd}）还有 {delta} 天"
            elif delta == 0:
                return "就是今天！"
            else:
                return f"{target.strftime('%Y年%m月%d日')} 已经过去 {abs(delta)} 天了"

        elif action == "after" and days is not None:
            future = datetime.now() + timedelta(days=days)
            return f"{days}天后是 {future.strftime('%Y年%m月%d日')} {weekdays_cn[future.weekday()]}"

        elif action == "week_info":
            today = datetime.now()
            iso = today.isocalendar()
            return f"当前是 {today.year}年 第{iso[1]}周，{weekdays_cn[today.weekday()]}"

        else:
            return "请提供有效的操作: today / weekday+date / countdown+date / after+days"

    except ValueError as e:
        return f"日期格式错误: {e}。请使用 YYYY-MM-DD 格式。"


# ═══════════════════════════════════════════════════
#  Tool Registry
# ═══════════════════════════════════════════════════

from agent.tools.mcp_bridge import MCP_TOOLS

ALL_TOOLS = [
    search_knowledge_base,
    search_web,
    query_salary,
    analyze_jd,
    match_skills,
    calendar_tool,
] + MCP_TOOLS  # MCP external service tools


def get_tools():
    """Return the list of tools for the agent."""
    return ALL_TOOLS
