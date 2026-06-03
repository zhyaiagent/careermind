"""
NL2SQL Node — natural language to SQL for salary data queries.

Pipeline:
1. Generate SQL from natural language query
2. Execute SQL against SQLite salary database
3. Generate natural language summary of results
"""
import sqlite3
import pandas as pd
from langchain_core.language_models import BaseChatModel

from config import SALARY_DB_PATH


class NL2SQLNode:
    """
    Converts natural language salary queries into SQL,
    executes them, and summarizes results in natural language.
    """

    SCHEMA_DESCRIPTION = """表名: salaries
字段说明:
- job_title TEXT: 岗位名称
- company_type TEXT: 公司类型(互联网/国企/外企/创业公司)
- city TEXT: 城市
- experience TEXT: 经验要求
- education TEXT: 学历要求
- min_salary INTEGER: 最低薪资(K)
- max_salary INTEGER: 最高薪资(K)
- avg_salary REAL: 平均薪资(K)
- source TEXT: 数据来源"""

    def __init__(self, llm: BaseChatModel, db_path: str | None = None):
        self.llm = llm
        self.db_path = db_path or SALARY_DB_PATH

    def __call__(self, state: dict) -> dict:
        query = state.get("query", "")

        # 1. NL → SQL
        sql_prompt = (
            f"你是一个SQL专家。根据以下表结构生成SQL查询语句。\n\n"
            f"{self.SCHEMA_DESCRIPTION}\n\n"
            f"注意：只输出SQL语句，不要任何解释或markdown格式。使用SELECT语句。\n\n"
            f"查询需求: {query}"
        )
        raw_sql = self.llm.invoke(sql_prompt)
        sql = raw_sql.content if hasattr(raw_sql, 'content') else str(raw_sql)
        sql = sql.replace("```sql", "").replace("```", "").strip()

        # 2. Execute SQL
        try:
            conn = sqlite3.connect(self.db_path)
            result_df = pd.read_sql(sql, conn)
            result_text = result_df.to_markdown(index=False) if not result_df.empty else "未找到匹配的薪资数据。"
            conn.close()
        except Exception as e:
            result_text = f"数据查询出错: {str(e)}"

        # 3. Generate natural language answer
        answer_prompt = (
            f"请根据SQL查询结果，用中文简洁地回答用户的问题。\n\n"
            f"用户问题: {query}\n"
            f"执行SQL: {sql}\n"
            f"查询结果:\n{result_text}\n\n"
            f"回答:"
        )
        answer_response = self.llm.invoke(answer_prompt)
        answer = answer_response.content if hasattr(answer_response, 'content') else str(answer_response)

        return {
            "tool_calls": [{"tool": "nl2sql", "sql": sql}],
            "tool_results": [{"data": result_text}],
            "raw_answer": answer,
            "final_answer": answer,
        }
