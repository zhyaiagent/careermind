"""
Agent State — type definition for the hybrid agent (ReAct + Plan-Execute).

Messages accumulate across both paths. Plan-Execute adds:
  - route: "react" or "plan_execute"
  - plan: list of execution steps with results
  - final_answer: synthesizer output
"""
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """Hybrid agent state — used by both ReAct and Plan-Execute paths."""
    messages: Annotated[list[BaseMessage], operator.add]
    route: str                      # "react" | "plan_execute"
    plan: list[dict]                # Plan-Execute: [{step, tool, args, result}]
    tool_results: list[dict]        # collected step results
    final_answer: str               # final response to user
