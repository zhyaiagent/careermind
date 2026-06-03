"""
Web Search Tool — internet search via Tavily API.

Used by the agent to get real-time job market info.
"""
import json


class WebSearchTool:
    """
    Wraps Tavily web search for real-time job market information.

    Usage:
        tool = WebSearchTool(api_key="tvly-...")
        results = tool.search("2026年AI工程师薪资")
    """

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.search_client = None
        if api_key and api_key != "your_tavily_api_key":
            try:
                from langchain_tavily import TavilySearch
                self.search_client = TavilySearch(
                    tavily_api_key=api_key,
                    max_results=5,
                )
            except ImportError:
                self.search_client = None

    def search(self, query: str) -> str:
        """
        Search the web for real-time information.

        Args:
            query: Search query string

        Returns:
            Formatted search results as a string, ready for LLM context.
        """
        if self.search_client is None:
            return ""

        try:
            result = self.search_client.invoke(query)

            # Tavily returns: {'results': [{url, title, content, score}, ...]}
            if isinstance(result, dict) and "results" in result:
                items = result["results"]
                if not items:
                    return ""
                lines = []
                for i, item in enumerate(items[:5], 1):
                    title = item.get("title", "")
                    content = item.get("content", "")
                    url = item.get("url", "")
                    lines.append(f"[web{i}] {title}\n{content}\n来源: {url}")
                return "\n\n".join(lines)

            return str(result)[:2000]
        except Exception:
            return ""
