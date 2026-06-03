"""
Salary Query Tool — SQLite-based salary data lookups.

Provides structured queries against the salary database.
"""
import sqlite3
import pandas as pd
from langchain_core.tools import tool

from config import SALARY_DB_PATH


class SalaryQueryTool:
    """
    Queries the salary database with optional filters.

    Usage:
        tool = SalaryQueryTool(db_path="./data/processed/salary.db")
        results = tool.query_salary(job_title="AI工程师", city="北京")
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or SALARY_DB_PATH

    @tool
    def query_salary(
        self,
        job_title: str | None = None,
        city: str | None = None,
        experience: str | None = None,
    ) -> str:
        """
        Query salary data with optional filters.

        Args:
            job_title: 岗位名称，如 "AI工程师"
            city: 城市，如 "北京"
            experience: 经验要求，如 "1-3"

        Returns:
            Markdown table of matching salary records, or "未找到匹配数据" if empty.
        """
        conn = sqlite3.connect(self.db_path)
        query_parts = ["SELECT * FROM salaries WHERE 1=1"]
        params: list = []

        if job_title:
            query_parts.append("AND job_title LIKE ?")
            params.append(f"%{job_title}%")
        if city:
            query_parts.append("AND city = ?")
            params.append(city)
        if experience:
            query_parts.append("AND experience LIKE ?")
            params.append(f"%{experience}%")

        query = " ".join(query_parts)
        df = pd.read_sql(query, conn, params=params)
        conn.close()

        if df.empty:
            return "未找到匹配的薪资数据。"

        return df.to_markdown(index=False)
