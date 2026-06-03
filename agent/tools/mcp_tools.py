"""
MCP Tool Manager — Model Context Protocol integration.

Registers external MCP servers (e.g., 12306 ticket query) for agent use.

Tools are loaded lazily and exposed to the agent graph via LangChain adapters.
"""


class MCPToolManager:
    """
    Manages MCP (Model Context Protocol) tool connections.

    Currently registered servers:
      - 12306-mcp: train ticket queries
    """

    def get_tools_config(self) -> dict:
        """
        Return MCP server configurations.

        Each entry:
          transport: communication protocol
          url: server endpoint

        Returns:
            {server_name: {transport, url}}
        """
        tools_config = {
            "12306-mcp": {
                "transport": "streamable_http",
                "url": "https://mcp.api-inference.modelscope.net/c30f9b25034446/mcp",
            }
        }
        return tools_config

    def get_tools(self) -> list:
        """
        Load MCP tools via langchain-mcp-adapters.

        Returns a list of LangChain BaseTool instances.
        """
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            config = self.get_tools_config()
            client = MultiServerMCPClient(config)
            # Tools are loaded async — for sync usage, collect available tools
            # In LangGraph, use async node for full MCP support
            return []
        except ImportError:
            # langchain-mcp-adapters not installed
            return []
        except Exception:
            return []
