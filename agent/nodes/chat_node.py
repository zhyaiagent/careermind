"""
Chat Node — general conversation, answers any question using LLM knowledge.

No retrieval needed — direct LLM response.
Handles both job-seeking and general questions naturally.
"""
from langchain_core.language_models import BaseChatModel


class ChatNode:
    """
    Answers user questions using the LLM's general knowledge.

    For job-seeking questions, provides specialized advice.
    For general/off-topic questions, still answers helpfully
    using the LLM's broad knowledge base.
    """

    CHAT_PROMPT = """你是 JobSense，一个智能助手。你主要专注于求职相关服务（岗位分析、技能匹配、薪资查询、面试指导、职业规划），但也能回答用户的其他问题。

## 回答规则
- 求职相关问题：给出专业、具体的建议
- 其他问题：正常回答，就像普通的AI助手一样
- 使用中文，简洁清晰
- 如果不确定，诚实说明，给出你能提供的帮助"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def __call__(self, state: dict) -> dict:
        query = state.get("query", "")
        prompt = f"{self.CHAT_PROMPT}\n\n用户: {query}\n助手:"
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        return {"final_answer": content}
