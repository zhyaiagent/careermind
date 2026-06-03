"""
Reject Node — handles out-of-domain queries.

Politely declines to answer and redirects to job-seeking domain.
"""


class RejectNode:
    """
    Handles queries that fall outside the job-seeking domain.

    Returns a polite rejection message.
    """

    REJECT_MESSAGE = (
        "抱歉，我是专业的求职助手，专注于岗位分析、技能匹配、薪资查询等求职相关服务。"
        "如有关于求职、面试、职业发展的问题，我很乐意帮助您！"
    )

    def __call__(self, state: dict) -> dict:
        return {"final_answer": self.REJECT_MESSAGE}
