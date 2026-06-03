"""
Intent Router — classifies user queries to downstream agent nodes.

Intent taxonomy (from specific to general):
  - jd_analysis: "分析这个JD" / "解读这个岗位要求"
  - salary_query: "AI工程师薪资多少" / "北京和上海工资对比"
  - skill_match: "我的技能匹配吗" / "我能胜任这个岗位吗"
  - chat: everything else — general questions, casual talk, off-topic — all answered by LLM
  - reject: only for clearly unsafe/harmful content
"""
from langchain_core.language_models import BaseChatModel


class IntentRouter:
    """
    Routes user queries. Most questions go to 'chat' where the LLM
    can answer freely using its general knowledge. Only specific
    job-seeking intents get specialized handling (RAG, NL2SQL, skill match).
    """

    ROUTING_PROMPT = """你是一个意图分类器。根据用户输入判断意图，仅输出以下之一：

- jd_analysis: 用户在分析或解读某个岗位的JD/职位描述
- salary_query: 用户在查询薪资、工资、薪酬相关数据
- skill_match: 用户在对比自己的技能和某个岗位的要求
- chat: 其他所有问题（闲聊、通用知识、技术问题、非求职问题等都用这个）

注意：绝大多数问题都应该输出 chat。只有明确是上述三种特定意图时才输出对应的。

用户输入: {query}
意图: """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def route(self, state: dict) -> dict:
        """
        Classify the user query. Falls back to 'chat' so LLM handles everything.
        """
        messages = state.get("messages", [])
        if messages:
            query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        else:
            query = state.get("query", "")

        prompt = self.ROUTING_PROMPT.format(query=query)
        response = self.llm.invoke(prompt)
        intent = response.content.strip().lower() if hasattr(response, 'content') else str(response).strip().lower()

        valid_intents = ["jd_analysis", "salary_query", "skill_match", "chat", "reject"]
        if intent not in valid_intents:
            intent = "chat"  # fallback to chat, not reject

        return {"intent": intent, "query": query}
