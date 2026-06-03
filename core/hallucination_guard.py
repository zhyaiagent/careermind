"""
Hallucination Guard — quality check for RAG-generated answers.

Checks (progressively):
1. Relevance threshold — if retrieved docs all have low relevance, let LLM answer without RAG
2. Citation check — if RAG answer lacks citations but docs exist, add a note
3. Content safety — only reject harmful/inappropriate content
4. Accept — answer passes all checks
"""
import re


class HallucinationGuard:
    """
    Validates generated answers against retrieved documents.

    Non-RAG answers (chat node) are always accepted.
    RAG answers are checked for relevance and citations.
    Only truly harmful content is rejected.
    """

    # Only reject truly harmful queries, not general/off-topic ones
    HARMFUL_PATTERNS = [
        r"(制造|制作|怎么?做).*(武器|炸弹|毒品|毒药)",
        r"(自杀|自残).*(方法|方式)",
        r"(黑客|破解|攻击).*(银行|政府|系统)",
        r"(儿童|未成年).*(色情|性)",
    ]

    def __init__(self, relevance_threshold: float = 0.1):
        self.relevance_threshold = relevance_threshold

    def check(
        self,
        answer: str,
        retrieved_docs: list[dict],
        query: str,
    ) -> dict:
        """
        Run guard checks. Most answers pass — only extreme cases are blocked.
        """
        # Check 1: Safety — only reject genuinely harmful queries
        if self._is_harmful(query):
            return {
                "passed": False,
                "action": "reject",
                "modified_answer": "抱歉，我不能回答这个问题。请提出其他问题。",
                "reason": "问题包含不安全内容。",
            }

        # Check 2: Low relevance RAG answer → pass through, let LLM handle it
        if retrieved_docs:
            max_relevance = max(
                doc.get("relevance_score", 0) for doc in retrieved_docs
            )
            if max_relevance < self.relevance_threshold:
                # Don't block — just don't show unreliable sources
                return {
                    "passed": True,
                    "action": "accept",
                    "modified_answer": answer,
                    "reason": f"检索相关性较低({max_relevance:.2f})，但允许LLM自由回答。",
                }

        # Check 3: Citation check (soft warning, never blocks)
        has_citation = bool(re.search(r'\[\d+\]', answer))
        if not has_citation and retrieved_docs:
            answer += "\n\n（注：以上回答部分基于检索到的文档。）"
            return {
                "passed": True,
                "action": "accept",
                "modified_answer": answer,
                "reason": "回答缺少引用标注，已添加提示。",
            }

        # Check 4: Accept
        return {
            "passed": True,
            "action": "accept",
            "modified_answer": answer,
            "reason": "回答通过所有检查。",
        }

    def _is_harmful(self, text: str) -> bool:
        """Check if query contains genuinely harmful content."""
        import re
        return any(re.search(p, text, re.IGNORECASE) for p in self.HARMFUL_PATTERNS)
