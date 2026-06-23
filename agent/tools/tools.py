"""JobSense Tools — LangChain @tool functions with retry + fallback."""
import json, logging, functools, time
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def with_retry(max_attempts=3, fallback_msg=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts:
                        logger.warning(f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}")
                        time.sleep(0.5 * attempt)
            logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            return fallback_msg.format(error=last_error) if fallback_msg else f"[{func.__name__} unavailable]"
        return wrapper
    return decorator

# Injected deps
_retriever = None
_generator = None
_web_search = None
_salary_db_path = None
_embedding_manager = None

def inject_dependencies(**kw):
    global _retriever, _generator, _web_search, _salary_db_path, _embedding_manager
    _retriever = kw.get("retriever")
    _generator = kw.get("generator")
    _web_search = kw.get("web_search")
    _salary_db_path = kw.get("salary_db_path", "")
    _embedding_manager = kw.get("embedding_manager")

# ---------- Tools ----------

class SearchKBInput(BaseModel):
    query: str = Field(description="search query")

@tool(args_schema=SearchKBInput)
def search_knowledge_base(query: str) -> str:
    """Search local knowledge base for job descriptions, reports, and uploaded documents."""
    if _retriever is None: return "KB not initialized."
    try:
        docs = _retriever.retrieve(query, top_k=3)
        if not docs: return "No results."
        lines = []
        for i, d in enumerate(docs, 1):
            src = d.get("metadata", {}).get("source", "?")
            lines.append(f"[{i}] (src:{src})\n{d['content']}")
        return "\n\n---\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"

class SearchWebInput(BaseModel):
    query: str = Field(description="web search query")

@tool(args_schema=SearchWebInput)
@with_retry(3, fallback_msg="Web search temporarily unavailable")
def search_web(query: str) -> str:
    """Search the web for real-time information using Tavily."""
    if _web_search is None: return "Web search not configured."
    result = _web_search.search(query)
    return result if result else "No web results."

class QuerySalaryInput(BaseModel):
    job_title: Optional[str] = Field(default=None)
    city: Optional[str] = Field(default=None)
    experience: Optional[str] = Field(default=None)

@tool(args_schema=QuerySalaryInput)
@with_retry(3, fallback_msg="Salary query failed, check database connection")
def query_salary(job_title=None, city=None, experience=None) -> str:
    """Query salary database by job title, city, and experience level."""
    from core.database import query_salary_db
    rows = query_salary_db(job_title=job_title, city=city, experience=experience)
    if not rows: return "No salary data found."
    import pandas as pd
    df = pd.DataFrame(rows)
    cols = ["job_title","company_type","city","experience","min_salary","max_salary","avg_salary","source"]
    available = [c for c in cols if c in df.columns]
    return df[available].to_markdown(index=False)

class AnalyzeJDInput(BaseModel):
    jd_text: str = Field(description="full JD text")

@tool(args_schema=AnalyzeJDInput)
def analyze_jd(jd_text: str) -> str:
    """Analyze a job description and extract key skills, salary, and career path."""
    if _generator is None: return "Generator not initialized."
    result = _generator.analyze_jd(jd_text)
    return json.dumps(result, ensure_ascii=False, indent=2) if "raw_response" not in result else result["raw_response"]

class MatchSkillsInput(BaseModel):
    user_skills: str = Field(description="candidate skills")
    job_requirements: str = Field(description="job requirements")

@tool(args_schema=MatchSkillsInput)
def match_skills(user_skills: str, job_requirements: str) -> str:
    """Match user skills against job requirements with gap analysis."""
    if _generator is None: return "Generator not initialized."
    result = _generator.match_skills(user_skills, job_requirements)
    return json.dumps(result, ensure_ascii=False, indent=2) if "raw_response" not in result else result["raw_response"]

class CalendarInput(BaseModel):
    action: str = Field(description="today/weekday/countdown/after")
    date: Optional[str] = Field(default=None)
    days: Optional[int] = Field(default=None)

@tool(args_schema=CalendarInput)
def calendar_tool(action: str = "today", date=None, days=None) -> str:
    """Calendar tool: check date, weekday, countdown, and date arithmetic."""
    wd = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
    try:
        if action == "today":
            t = datetime.now()
            return f"今天是 {t.strftime('%Y年%m月%d日')} {wd[t.weekday()]}（第{t.isocalendar()[1]}周）"
        elif action == "weekday" and date:
            dt = datetime.strptime(date, "%Y-%m-%d")
            return f"{date} 是 {wd[dt.weekday()]}"
        elif action == "countdown" and date:
            dt = datetime.strptime(date, "%Y-%m-%d")
            delta = (dt - datetime.now()).days
            return f"距 {date} 还有 {delta} 天" if delta > 0 else f"{date} 已过去 {-delta} 天" if delta < 0 else "就是今天！"
        elif action == "after" and days:
            future = datetime.now() + timedelta(days=days)
            return f"{days}天后是 {future.strftime('%Y年%m月%d日')} {wd[future.weekday()]}"
        return "请提供有效操作: today / weekday+date / countdown+date / after+days"
    except Exception as e:
        return f"日历错误: {e}"

# ---------- Registry ----------
from agent.tools.mcp_bridge import MCP_TOOLS

ALL_TOOLS = [
    search_knowledge_base, search_web, query_salary,
    analyze_jd, match_skills, calendar_tool,
] + list(MCP_TOOLS)

def get_tools():
    return ALL_TOOLS
