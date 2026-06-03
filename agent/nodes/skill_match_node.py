"""
Skill Match Node — evaluates user skills against job requirements.

Pipeline:
1. Retrieve relevant JD documents
2. Run structured skill match analysis via LLM
3. Format results into a readable markdown response
"""
from core.retrieval import HybridRetriever
from core.generation import GenerationManager


class SkillMatchNode:
    """
    Analyzes the gap between a user's skills and job requirements.

    Uses a built-in user skill profile as the reference,
    retrieves matching JDs, and produces a detailed gap analysis.
    """

    # Built-in user skill profile (in production, this comes from user profile)
    USER_SKILLS = """求职者背景:
1. 编程语言: Python(精通, 掌握Pandas/NumPy/Scikit-learn), SQL(熟练), C++(入门)
2. AI技术: PyTorch(熟练), HuggingFace Transformers(熟练), LlamaFactory(LoRA/QLoRA微调经验)
3. 框架工具: RAG(熟练), LangChain(熟练), Embedding(理解)
4. Agent: GUI Agent(MobileAgent项目经验), MCP(了解协议)
5. 基础能力: Transformer原理理解, 熟悉Git协作
6. 语言能力: CET4+CET6
7. 兴趣方向: AI应用开发, 大模型应用
8. 其他: 有竞赛经历, GPA 3.31, 有实习经历"""

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: GenerationManager,
    ):
        self.retriever = retriever
        self.generator = generator

    def __call__(self, state: dict) -> dict:
        query = state.get("query", "")

        # 1. Retrieve relevant JD documents
        jd_docs = self.retriever.retrieve(query, top_k=2)
        jd_content = "\n".join([d["content"] for d in jd_docs])

        # 2. Match skills
        result = self.generator.match_skills(self.USER_SKILLS, jd_content)

        # 3. Format markdown result
        answer = self._format_match_result(result)

        return {
            "retrieved_docs": jd_docs,
            "raw_answer": answer,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def _format_match_result(self, result: dict) -> str:
        """Convert structured match result to readable markdown."""
        if "raw_response" in result:
            return result["raw_response"]

        lines = [f"## 技能匹配度: {result.get('match_score', 'N/A')}/100"]

        # Matched skills
        matched = result.get("matched_skills", [])
        if matched:
            lines.append("\n### ✅ 已匹配技能")
            for s in matched:
                lines.append(
                    f"- **{s.get('skill', '')}** ({s.get('level', '')}): "
                    f"{s.get('evidence', '')}"
                )

        # Missing skills
        missing = result.get("missing_skills", [])
        if missing:
            lines.append("\n### ⚠️ 待补充技能")
            for s in missing:
                lines.append(
                    f"- **{s.get('skill', '')}** (优先级: {s.get('priority', '')})"
                )
                lines.append(f"  {s.get('suggestion', '')}")
                if s.get("learning_resource"):
                    lines.append(f"  推荐资源: {s.get('learning_resource', '')}")

        # Competitive advantage
        if result.get("competitive_advantage"):
            lines.append(f"\n### 💪 竞争优势")
            lines.append(result["competitive_advantage"])

        # Risk points
        if result.get("risk_points"):
            lines.append(f"\n### ⚡ 风险点")
            for r in result["risk_points"]:
                lines.append(f"- {r}")

        # Overall suggestion
        if result.get("overall_suggestion"):
            lines.append(f"\n### 📝 综合建议")
            lines.append(result["overall_suggestion"])

        return "\n".join(lines)
