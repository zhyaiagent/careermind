"""
Conversation Memory — persistent session management for the agent.

Uses LangGraph's MemorySaver for checkpoint-based conversation history.
Supports multiple conversation threads, each with up to 20 turns.
"""
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage


class ConversationMemory:
    """
    Manages conversation history across sessions.

    Uses LangGraph MemorySaver for automatic checkpointing.
    Each thread_id maps to an independent conversation.
    """

    def __init__(self):
        self.checkpointer = MemorySaver()

    def get_checkpointer(self) -> MemorySaver:
        """Return the MemorySaver instance for use in graph compilation."""
        return self.checkpointer

    def get_thread_config(self, thread_id: str = "default") -> dict:
        """
        Generate thread configuration for a conversation.

        Args:
            thread_id: unique identifier for the conversation thread

        Returns:
            {"configurable": {"thread_id": thread_id}}
        """
        return {"configurable": {"thread_id": thread_id}}
