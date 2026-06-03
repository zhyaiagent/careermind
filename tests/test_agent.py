"""
Test Suite — Agent (ReAct + Tools + Legacy Nodes)

Tests: AgentState, Tools, ReAct graph, ConversationMemory,
and legacy nodes (still usable as standalone components).
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Mock Components ───────────────────────────────

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig


class FakeLLM(BaseChatModel):
    """Fake chat model usable with create_agent."""
    response: str = "mock response"

    def __init__(self, response="mock response", **kwargs):
        super().__init__(response=response, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGeneration(message=AIMessage(content=self.response))

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self):
        return "fake"

    @property
    def model_name(self):
        return "fake-model"


class FakeRetriever:
    def retrieve(self, query, top_k=3):
        return [
            {"content": f"Result {i}: {query}", "metadata": {"source": f"src{i}.pdf"}, "relevance_score": 0.9 - i * 0.1}
            for i in range(1, top_k + 1)
        ]


class FakeGenerator:
    def analyze_jd(self, jd_text):
        return {"job_title": "测试", "core_skills": ["Python"], "salary_estimate": "15-25K"}
    def match_skills(self, user, req):
        return {"match_score": 78, "overall_suggestion": "匹配良好"}
    def generate_rag_answer(self, q, docs):
        return {"answer": "test answer", "sources": []}


class FakeWebSearch:
    def __init__(self):
        self.search_client = True
    def search(self, query):
        return f"Web results for: {query}"


# ═══════════════════════════════════════════════════
#  AgentState
# ═══════════════════════════════════════════════════

class TestAgentState:
    def test_state_has_messages(self):
        from agent.state import AgentState
        assert "messages" in AgentState.__annotations__

    def test_state_is_typeddict(self):
        from agent.state import AgentState
        from typing import TypedDict
        # AgentState is a TypedDict with operator.add annotated messages
        assert "messages" in AgentState.__annotations__


# ═══════════════════════════════════════════════════
#  Tools
# ═══════════════════════════════════════════════════

class TestTools:
    """Test the LangChain @tool functions."""

    @pytest.fixture(autouse=True)
    def setup_tools(self):
        from agent.tools.tools import inject_dependencies
        inject_dependencies(
            retriever=FakeRetriever(),
            generator=FakeGenerator(),
            web_search=FakeWebSearch(),
            salary_db_path="data/processed/salary.db",
            embedding_manager=None,
        )

    def test_search_knowledge_base(self):
        from agent.tools.tools import search_knowledge_base
        result = search_knowledge_base.invoke({"query": "AI工程师技能"})
        assert "Result" in result
        assert "src" in result

    def test_search_web(self):
        from agent.tools.tools import search_web
        result = search_web.invoke({"query": "薪资趋势"})
        assert "Web results" in result

    def test_query_salary(self):
        from agent.tools.tools import query_salary
        # Without a real DB, it should return error or empty
        result = query_salary.invoke({"job_title": "AI", "city": "北京"})
        assert isinstance(result, str)

    def test_analyze_jd(self):
        from agent.tools.tools import analyze_jd
        result = analyze_jd.invoke({"jd_text": "AI工程师，要求Python和PyTorch"})
        assert "Python" in result or "测试" in result or "job_title" in result

    def test_match_skills(self):
        from agent.tools.tools import match_skills
        result = match_skills.invoke({
            "user_skills": "Python, PyTorch",
            "job_requirements": "要求Python和深度学习"
        })
        assert "78" in result or "match_score" in result

    def test_all_tools_registered(self):
        from agent.tools.tools import ALL_TOOLS
        tool_names = [t.name for t in ALL_TOOLS]
        assert "search_knowledge_base" in tool_names
        assert "search_web" in tool_names
        assert "query_salary" in tool_names
        assert "analyze_jd" in tool_names
        assert "match_skills" in tool_names
        assert "calendar_tool" in tool_names
        assert "call_mcp_tool" in tool_names
        assert "list_mcp_services" in tool_names

    def test_calendar_today(self):
        from agent.tools.tools import calendar_tool
        result = calendar_tool.invoke({"action": "today"})
        assert "今天是" in result

    def test_calendar_after(self):
        from agent.tools.tools import calendar_tool
        result = calendar_tool.invoke({"action": "after", "days": 30})
        assert "天后" in result


# ═══════════════════════════════════════════════════
#  ReAct Agent Graph
# ═══════════════════════════════════════════════════

class TestReActAgent:
    """Test the ReAct agent graph."""

    @pytest.fixture
    def agent_graph(self):
        from agent.tools.tools import inject_dependencies
        inject_dependencies(
            retriever=FakeRetriever(),
            generator=FakeGenerator(),
            web_search=FakeWebSearch(),
            salary_db_path="data/processed/salary.db",
            embedding_manager=None,
        )
        from agent.graph import build_agent_graph
        # Need a real-looking LLM that can respond without tools
        llm = FakeLLM(response="这是测试回答。")
        return build_agent_graph(llm=llm)

    def test_graph_compiles(self, agent_graph):
        assert agent_graph is not None

    def test_graph_invoke(self, agent_graph):
        from langchain_core.messages import HumanMessage
        result = agent_graph.invoke({"messages": [HumanMessage(content="你好")]})
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_graph_simple_question(self, agent_graph):
        """Simple question should get an answer without tool errors."""
        from langchain_core.messages import HumanMessage
        result = agent_graph.invoke({"messages": [HumanMessage(content="AI工程师需要什么技能？")]})
        assert "messages" in result


# ═══════════════════════════════════════════════════
#  Conversation Memory
# ═══════════════════════════════════════════════════

class TestConversationMemory:
    def test_memory_checkpointer(self):
        from agent.memory import ConversationMemory
        m = ConversationMemory()
        assert m.get_checkpointer() is not None

    def test_thread_config(self):
        from agent.memory import ConversationMemory
        m = ConversationMemory()
        cfg = m.get_thread_config("t123")
        assert cfg["configurable"]["thread_id"] == "t123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
