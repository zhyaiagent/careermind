"""
Generation Manager — LLM-powered answer generation with structured prompts.

Supports:
- DeepSeek Chat (via langchain-openai compatible)
- Qwen (via DashScope)
- GLM-4 (via ZhipuAI)

Provides three prompt chains:
1. RAG answer generation (with context + citations)
2. JD analysis (structured JSON output)
3. Skill match (structured JSON output)
"""
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import (
    LLM_MODEL, DEEPSEEK_API_KEY, DASHSCOPE_API_KEY, OPENAI_API_KEY,
)


# ═══════════════════════════════════════════════════
#  System & Task Prompts
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = """你是 JobSense 智能求职助手，专注于帮助求职者分析岗位、匹配技能、查询薪资。

## 你的能力
1. 分析岗位JD，提炼核心技能/薪资范围/发展路径
2. 根据求职者技能评估与岗位的匹配度
3. 回答求职相关的通用问题
4. 查询行业薪资数据

## 回答规则
- 使用中文回答，专业术语保留英文缩写
- 引用文档时使用 [1][2] 标注来源
- 不确定的信息明确说明"根据已有数据推测"
- 拒绝回答与求职无关的敏感问题
- 回答控制在 200 字以内，简洁实用
"""

RAG_PROMPT_TEMPLATE = """请根据以下参考资料回答用户问题。

## 参考资料
{context}

## 用户问题
{question}

## 回答要求
1. 基于参考资料给出准确答案
2. 引用具体来源标注 [1][2]
3. 如需补充说明，明确标注"补充："
4. 如果参考资料不足以回答，请说明并给出建议
"""

JD_ANALYSIS_PROMPT = """请分析以下岗位JD，提取关键信息。

## JD内容
{jd_text}

## 输出格式(严格JSON)
```json
{{
    "job_title": "岗位名称",
    "company_type": "互联网/国企/外企/创业公司",
    "core_skills": ["必备技能1", "必备技能2"],
    "bonus_skills": ["加分技能1"],
    "salary_estimate": "15-25K",
    "difficulty": "(1-5的难度评级)",
    "career_path": "可能的职业发展路径",
    "key_insights": ["关键洞察1", "关键洞察2"]
}}
```
"""

SKILL_MATCH_PROMPT = """请评估求职者技能与岗位要求的匹配度。

## 求职者技能
{user_skills}

## 岗位要求
{job_requirements}

## 输出格式(严格JSON)
```json
{{
    "match_score": 75,
    "matched_skills": [
        {{"skill": "Python", "level": "精通", "evidence": "岗位要求Python，求职者具备Python及Pandas/NumPy/Sklearn经验"}}
    ],
    "missing_skills": [
        {{"skill": "C++", "priority": "高", "suggestion": "建议补充C++基础，2-3个月可入门", "learning_resource": "《C++ Primer》前10章"}}
    ],
    "competitive_advantage": "求职者的Agent开发经验是差异化优势",
    "risk_points": ["学历非985/211可能被筛"],
    "overall_suggestion": "综合建议..."
}}
```
"""


class GenerationManager:
    """
    Manages LLM calls with pre-configured prompts for different tasks.

    Three generation modes:
      - generate_rag_answer: context-grounded Q&A
      - analyze_jd: structured JD extraction
      - match_skills: skill gap analysis
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or LLM_MODEL
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initialize the LLM based on config."""
        if self.model_name == "deepseek":
            return ChatOpenAI(
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
                api_key=DEEPSEEK_API_KEY,
                temperature=0.1,
                max_tokens=2048,
            )
        elif self.model_name == "qwen":
            return ChatOpenAI(
                model="qwen-plus",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=DASHSCOPE_API_KEY,
                temperature=0.1,
            )
        elif self.model_name == "glm":
            return ChatOpenAI(
                model="glm-4",
                base_url="https://open.bigmodel.cn/api/paas/v4",
                api_key=OPENAI_API_KEY,
                temperature=0.1,
            )
        else:
            raise ValueError(f"Unsupported LLM model: {self.model_name}")

    def generate_rag_answer(self, question: str, context_docs: list[dict]) -> dict:
        """
        Generate a RAG-grounded answer.

        Returns:
            {"answer": "...", "sources": [...]}
        """
        formatted_context = self._format_docs_with_index(context_docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", RAG_PROMPT_TEMPLATE),
        ])
        chain = prompt | self.llm
        response = chain.invoke({
            "context": formatted_context,
            "question": question,
        })
        return {
            "answer": response.content,
            "sources": [doc["metadata"] for doc in context_docs],
        }

    def analyze_jd(self, jd_text: str) -> dict:
        """Analyze a job description and extract structured information."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", JD_ANALYSIS_PROMPT),
        ])
        chain = prompt | self.llm
        response = chain.invoke({"jd_text": jd_text})
        return self._parse_json_response(response.content)

    def match_skills(self, user_skills: str, job_requirements: str) -> dict:
        """Match user skills against job requirements."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", SKILL_MATCH_PROMPT),
        ])
        chain = prompt | self.llm
        response = chain.invoke({
            "user_skills": user_skills,
            "job_requirements": job_requirements,
        })
        return self._parse_json_response(response.content)

    def _format_docs_with_index(self, docs: list[dict]) -> str:
        """Format retrieved documents with numbered indices for citation."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("metadata", {}).get("source", "未知来源")
            page = doc.get("metadata", {}).get("page", "?")
            formatted.append(
                f"[{i}] (来源: {source}, 第{page}页)\n{doc['content']}"
            )
        return "\n\n".join(formatted)

    def _parse_json_response(self, text: str) -> dict:
        """
        Extract JSON from LLM response.

        Tries:
        1. ```json ... ``` code block
        2. Raw JSON parse
        3. Fallback to raw_response
        """
        # Try fenced JSON block
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_response": text}
