"""
MCP Bridge — connects external MCP servers into the LangChain tool ecosystem.

Architecture:
  MCP Server (remote/external)  ← JSON-RPC over HTTP →  MCPBridge  →  LangChain @tool
                                                         (sync wrapper)

LLM sees MCP tools exactly the same as built-in tools.
The bridge handles protocol translation transparently.
"""
import json
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── MCP Server Registry ───────────────────────────
# Add MCP servers here. Each entry: name → {url, transport}
MCP_SERVERS = {
    "JobTools": {
        "url": "http://127.0.0.1:9020/mcp",
        "transport": "streamable_http",
        "description": "面试建议、税后薪资计算、公司信息查询",
        "enabled": True,
    },
    # 更多 MCP 服务器可以添加在这里:
    # "12306": {
    #     "url": "https://mcp.api-inference.modelscope.net/.../mcp",
    #     "transport": "streamable_http",
    #     "description": "火车票查询",
    #     "enabled": True,
    # },
}


# ── MCP Client ────────────────────────────────────

class MCPClient:
    """
    Minimal sync MCP client using JSON-RPC over HTTP.

    For production, use `mcp` SDK with async support.
    This implementation demonstrates the protocol without heavy deps.
    """

    def __init__(self, url: str):
        self.url = url
        self.session_id: Optional[str] = None
        self._tools_cache: list[dict] = []

    def _parse_sse(self, text: str) -> dict | None:
        """Parse SSE (Server-Sent Events) response body into JSON."""
        for line in text.split("\n"):
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    return json.loads(data_str)
                except json.JSONDecodeError:
                    continue
        return None

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request to the MCP server (supports SSE responses)."""
        import requests

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        try:
            resp = requests.post(self.url, json=payload, headers=headers, timeout=30)
            # Extract session ID from response header
            sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
            if sid:
                self.session_id = sid
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    # MCP streamable_http SSE format
                    parsed = self._parse_sse(resp.text)
                    if parsed:
                        return parsed
                    return {"error": "Failed to parse SSE response"}
                else:
                    return resp.json()
            else:
                logger.warning(f"MCP RPC {method} → HTTP {resp.status_code}")
                # Try SSE parsing even on non-200
                parsed = self._parse_sse(resp.text)
                if parsed:
                    return parsed
                return {"error": resp.text}
        except requests.RequestException as e:
            logger.warning(f"MCP RPC {method} → {e}")
            return {"error": str(e)}

    def connect(self) -> bool:
        """Initialize MCP session."""
        result = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "JobSense", "version": "2.0"},
        })
        if "error" not in result:
            # Send initialized notification
            self._rpc("notifications/initialized", {})
            return True
        return False

    def list_tools(self) -> list[dict]:
        """Discover tools from the MCP server."""
        if self._tools_cache:
            return self._tools_cache

        result = self._rpc("tools/list", {})
        if "result" in result and "tools" in result["result"]:
            self._tools_cache = result["result"]["tools"]
        return self._tools_cache

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return its result as a string."""
        result = self._rpc("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        if "result" in result:
            content = result["result"].get("content", [])
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict):
                        texts.append(item.get("text", str(item)))
                    else:
                        texts.append(str(item))
                return "\n".join(texts)
            return str(content)
        if "error" in result:
            return f"MCP工具调用失败: {result['error']}"
        return "MCP工具未返回内容"


# ── MCP → LangChain Tool Factory ─────────────────

class MCPToolInput(BaseModel):
    """Generic input for any single MCP tool."""
    server: str = Field(description="MCP服务器名称，如'12306'")
    tool_name: str = Field(description="要调用的工具名称")
    arguments: str = Field(description="JSON格式的工具参数，如'{\"from\":\"北京\",\"to\":\"上海\"}'")


# Cache of connected clients
_mcp_clients: dict[str, MCPClient] = {}


def _get_client(server_name: str) -> Optional[MCPClient]:
    """Get or create an MCP client for the given server."""
    if server_name not in MCP_SERVERS:
        return None

    cfg = MCP_SERVERS[server_name]
    if not cfg.get("enabled", True):
        return None

    if server_name not in _mcp_clients:
        client = MCPClient(cfg["url"])
        if client.connect():
            _mcp_clients[server_name] = client
            tools = client.list_tools()
            logger.info(f"MCP [{server_name}]: {len(tools)} tools discovered")
        else:
            logger.warning(f"MCP [{server_name}]: connection failed")
            _mcp_clients[server_name] = client  # cache even failed

    return _mcp_clients[server_name]


@tool(args_schema=MCPToolInput)
def call_mcp_tool(server: str, tool_name: str, arguments: str = "{}") -> str:
    """调用外部MCP服务（自动重试3次，失败则降级提示）。"""
    # try up to 3 times
    import time
    last_err = None
    for attempt in range(1, 4):
        try:
            return _call_mcp_tool_impl(server, tool_name, arguments)
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(0.5 * attempt)
    return f"[MCP {server}/{tool_name} 暂时不可用 (重试3次失败): {last_err}]"


def _call_mcp_tool_impl(server: str, tool_name: str, arguments: str = "{}") -> str:
    """
    调用外部MCP服务。当前可用服务器:

    - JobTools: 求职工具 + 浏览器多步操控。工具:
      · get_interview_tips: 面试建议 (role, level)
      · calculate_after_tax: 税后计算 (monthly_salary, city)
      · get_company_info: 公司信息 (company_name)
      · browser_action: 浏览器多步操控。action类型:
        - navigate: 打开网址 (url)
        - click: 点击按钮 (text或selector)
        - type: 输入文字 (text)
        - press: 按键 (key, 如Enter)
        - get_content: 获取页面文字
        - screenshot: 截图看页面
        - wait: 等待 (seconds, reason如"等待用户扫码")
        - search: Bing搜索 (query)
    多步骤任务: 逐步调用browser_action完成复杂流程(打开→登录→等待扫码→提问→返回)

    使用示例:
    - server='JobTools', tool_name='browser_action', arguments='{"action":"navigate","url":"https://chat.deepseek.com"}'
    - server='JobTools', tool_name='browser_action', arguments='{"action":"click","text":"登录"}'
    - server='JobTools', tool_name='browser_action', arguments='{"action":"wait","seconds":20,"reason":"等待用户扫码登录"}'
    - server='JobTools', tool_name='browser_action', arguments='{"action":"get_content"}'
    """
    if server not in MCP_SERVERS:
        available = ", ".join(f"{k}({v['description']})" for k, v in MCP_SERVERS.items())
        return f"未知的MCP服务 '{server}'。可用服务: {available}"

    client = _get_client(server)
    if client is None:
        return f"MCP服务 '{server}' 未配置或未启用。"

    # Parse arguments
    try:
        args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return f"参数格式错误: {arguments}。请使用JSON格式。"

    result = client.call_tool(tool_name, args_dict)
    if not result:
        return f"MCP工具 '{tool_name}' 执行完成，但无返回内容。"

    return result


@tool
def list_mcp_services(query: str = "") -> str:
    """
    列出当前可用的MCP外部服务及其工具。
    当用户想了解系统支持哪些外部服务时使用。
    """
    lines = ["## 可用的 MCP 外部服务\n"]

    for name, cfg in MCP_SERVERS.items():
        if not cfg.get("enabled", True):
            continue
        lines.append(f"### {name} — {cfg['description']}")
        lines.append(f"连接: {cfg['url']}")
        lines.append(f"协议: {cfg['transport']}")

        # Try to get tools list
        client = _get_client(name)
        if client and client._tools_cache:
            lines.append("工具列表:")
            for t in client._tools_cache:
                desc = t.get("description", "")[:80]
                lines.append(f"  · {t['name']}: {desc}")
        else:
            lines.append("（服务器暂未连接，工具列表不可用）")
        lines.append("")

    if not lines[1:]:
        return "当前没有配置MCP服务。"
    return "\n".join(lines)


# ═══════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════

MCP_TOOLS = [call_mcp_tool, list_mcp_services]
