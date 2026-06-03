"""
Test Suite — Generation Manager

Tests: RAG answer generation, JD analysis, skill matching,
prompt formatting, and JSON response parsing.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Mock LLM ──────────────────────────────────────

from langchain_core.runnables import RunnableLambda

class FakeLLM:
    """Fake LLM that returns pre-configured responses (LangChain Runnable compatible)."""
    def __init__(self, response_content=""):
        self.response_content = response_content

    def invoke(self, prompt, **kwargs):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.content = self.response_content
        return resp

    def __or__(self, other):
        """Support pipe operator for LangChain chaining."""
        return RunnableLambda(lambda x: self.invoke(x)) | other

    def __ror__(self, other):
        """Support reverse pipe."""
        return other | RunnableLambda(lambda x: self.invoke(x))


# ── Tests ─────────────────────────────────────────

class TestGenerationManager:
    """Tests for the generation manager."""

    @pytest.fixture
    def sample_docs(self):
        return [
            {
                "content": "AI算法工程师需要掌握Python和PyTorch",
                "metadata": {"source": "jd1.pdf", "page": 1},
                "relevance_score": 0.9,
            },
            {
                "content": "大模型岗位要求有Transformer经验",
                "metadata": {"source": "jd2.pdf", "page": 2},
                "relevance_score": 0.8,
            },
        ]

    @pytest.fixture
    def manager(self):
        from unittest.mock import MagicMock, patch
        from core.generation import GenerationManager

        # Create a real manager (might fail without API keys, so patch __init__)
        gen = object.__new__(GenerationManager)
        gen.model_name = "test"
        gen.llm = FakeLLM("Test response")
        return gen

    def _patch_chain(self, manager, return_value):
        """Helper to mock the chain invoke."""
        from unittest.mock import MagicMock

        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.content = return_value
        mock_chain.invoke.return_value = mock_response
        return mock_chain

    def test_format_docs_with_index(self, manager, sample_docs):
        """Should format documents with [1][2] indices."""
        formatted = manager._format_docs_with_index(sample_docs)
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "jd1.pdf" in formatted
        assert "第1页" in formatted

    def test_parse_json_from_fenced_block(self, manager):
        """Should extract JSON from ```json ... ``` block."""
        response = '这是分析结果\n```json\n{"job_title": "AI工程师", "core_skills": ["Python"]}\n```'
        result = manager._parse_json_response(response)
        assert result["job_title"] == "AI工程师"
        assert "Python" in result["core_skills"]

    def test_parse_json_direct(self, manager):
        """Should parse bare JSON string."""
        response = '{"match_score": 85, "matched_skills": []}'
        result = manager._parse_json_response(response)
        assert result["match_score"] == 85

    def test_parse_json_fallback(self, manager):
        """Should fallback to raw_response on invalid JSON."""
        response = "This is not JSON at all"
        result = manager._parse_json_response(response)
        assert "raw_response" in result
        assert result["raw_response"] == response

    def test_rag_answer_structure(self, manager, sample_docs, monkeypatch):
        """RAG answer should return answer + sources."""
        from unittest.mock import MagicMock

        # Mock the chain pipeline
        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "AI工程师需要掌握Python、PyTorch等技能[1][2]"
        mock_chain.invoke.return_value = mock_response

        # Patch ChatPromptTemplate to return our mock
        monkeypatch.setattr(
            "core.generation.ChatPromptTemplate.from_messages",
            lambda *args, **kwargs: MagicMock(__or__=lambda self, other: mock_chain)
        )

        result = manager.generate_rag_answer("AI工程师需要什么技能？", sample_docs)
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) == 2

    def test_jd_analysis_structure(self, manager, monkeypatch):
        """JD analysis should return parsed JSON or raw response."""
        from unittest.mock import MagicMock

        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '```json\n{"job_title": "AI算法", "core_skills": ["Python", "PyTorch"], '
            '"bonus_skills": [], "salary_estimate": "20-30K", "difficulty": 4, '
            '"career_path": "高级工程师", "key_insights": ["需求增长快"]}\n```'
        )
        mock_chain.invoke.return_value = mock_response

        monkeypatch.setattr(
            "core.generation.ChatPromptTemplate.from_messages",
            lambda *args, **kwargs: MagicMock(__or__=lambda self, other: mock_chain)
        )

        result = manager.analyze_jd("AI算法工程师要求...")
        assert result["job_title"] == "AI算法"
        assert "Python" in result["core_skills"]

    def test_skill_match_structure(self, manager, monkeypatch):
        """Skill match should return parsed JSON with match score."""
        from unittest.mock import MagicMock

        mock_chain = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '```json\n{"match_score": 75, "matched_skills": ['
            '{"skill": "Python", "level": "精通", "evidence": "..."}], '
            '"missing_skills": [], "competitive_advantage": "Agent经验", '
            '"risk_points": [], "overall_suggestion": "良好"}\n```'
        )
        mock_chain.invoke.return_value = mock_response

        monkeypatch.setattr(
            "core.generation.ChatPromptTemplate.from_messages",
            lambda *args, **kwargs: MagicMock(__or__=lambda self, other: mock_chain)
        )

        result = manager.match_skills("Python, PyTorch", "要求Python和深度学习")
        assert result["match_score"] == 75
        assert len(result["matched_skills"]) == 1


class TestSystemPrompts:
    """Verify system prompts are non-empty and well-formed."""

    def test_system_prompt_not_empty(self):
        from core.generation import SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) > 50

    def test_rag_prompt_has_placeholders(self):
        from core.generation import RAG_PROMPT_TEMPLATE
        assert "{context}" in RAG_PROMPT_TEMPLATE
        assert "{question}" in RAG_PROMPT_TEMPLATE

    def test_jd_analysis_prompt_has_json_format(self):
        from core.generation import JD_ANALYSIS_PROMPT
        assert "json" in JD_ANALYSIS_PROMPT.lower()
        assert "job_title" in JD_ANALYSIS_PROMPT

    def test_skill_match_prompt_has_json_format(self):
        from core.generation import SKILL_MATCH_PROMPT
        assert "json" in SKILL_MATCH_PROMPT.lower()
        assert "match_score" in SKILL_MATCH_PROMPT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
