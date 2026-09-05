"""
MCP Bridge — 使用官方 `mcp` 包作为标准 MCP Client，接通外部 MCP server。

架构:
  外部 MCP server (streamable_http / sse)  ← 官方 mcp.ClientSession →  LangChain @tool

LLM 通过 call_mcp_tool / list_mcp_services 调用外部 MCP 工具，与内置工具无异。
此实现取代了此前手写的 JSON-RPC-over-HTTP 客户端。
"""
import asyncio
import json
import logging
import os
import threading
import time
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── MCP Server Registry ───────────────────────────
# 接通外部 MCP server 只需在此加一条配置。
# transport: "streamable_http"（新版标准，推荐）| "sse"（旧版 HTTP+SSE）
MCP_SERVERS = {
    "ShuidiRisk": {
        "url": "https://mcpmarket.cn/mcp/a06281cf67a6099e43044fcb",
        "transport": "streamable_http",
        "description": "企业司法风险/大数据查询（水滴风险，28 个工具）",
        "enabled": True,
    },
    "playwright": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],
        "description": "微软浏览器自动化（Playwright）",
        "enabled": True,
    },
}


# ── 官方 MCP Client 封装 ──────────────────────────

def _resolve_headers(headers):
    """解析 headers 中的 ${ENV_VAR} 占位符，从环境变量取值（避免硬编码密钥）。"""
    if not headers:
        return None
    resolved = {}
    for k, v in headers.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            v = os.getenv(v[2:-1], "")
        resolved[k] = v
    return resolved


def _build_transport(name: str):
    """按配置构建 MCP transport 异步上下文管理器。

    支持 transport:
    - "stdio": 本地进程（npx/node 启动的 MCP server）
    - "sse": 旧版 HTTP+SSE
    - "streamable_http": 新版标准 HTTP（默认）
    """
    cfg = MCP_SERVERS[name]
    transport = cfg.get("transport", "streamable_http")

    if transport == "stdio":
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(command=cfg["command"], args=cfg.get("args", []))
        return stdio_client(params)

    url = cfg["url"]
    headers = _resolve_headers(cfg.get("headers"))
    if transport == "sse":
        from mcp.client.sse import sse_client
        return sse_client(url, headers=headers)
    from mcp.client.streamable_http import streamable_http_client
    if headers:
        import httpx
        return streamable_http_client(url, http_client=httpx.AsyncClient(headers=headers))
    return streamable_http_client(url)


async def _call_tool_async(server: str, tool_name: str, arguments: dict) -> str:
    """连接外部 MCP server 并调用工具，返回文本结果。"""
    from mcp import ClientSession
    transport = _build_transport(server)
    async with transport as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments or {})
    return _format_tool_result(result)


async def _list_tools_async(server: str) -> list:
    """连接外部 MCP server 并列出其工具。"""
    from mcp import ClientSession
    transport = _build_transport(server)
    async with transport as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            result = await session.list_tools()
            return list(result.tools)


def _format_tool_result(result) -> str:
    """把 CallToolResult.content 转成纯文本。"""
    texts = []
    for item in result.content:
        text = getattr(item, "text", None)
        texts.append(text if text is not None else str(item))
    return "\n".join(texts) if texts else "MCP 工具未返回内容"


def _run_async(coro, timeout: float = 60.0):
    """在独立线程的事件循环里运行协程，规避调用方（FastAPI）事件循环冲突。

    官方 mcp client 是异步的，而 LangChain @tool 是同步的；且同步 tool 会在
    FastAPI 事件循环线程内被调用，直接 asyncio.run() 会抛 "cannot be called
    from a running event loop"。因此每次调用放到独立线程 + 独立事件循环里执行。
    """
    result = {}
    error = {}

    def _runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as e:  # noqa: BLE001
            error["value"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"MCP 调用超时（>{timeout}s）")
    if "value" in error:
        raise error["value"]
    return result["value"]


# ── LangChain Tools ───────────────────────────────

class MCPToolInput(BaseModel):
    """通用 MCP 工具入参。"""
    server: str = Field(description="MCP服务器名称")
    tool_name: str = Field(description="要调用的工具名称")
    arguments: str = Field(description='JSON格式的工具参数，如 {"url":"https://example.com"}')


@tool(args_schema=MCPToolInput)
def call_mcp_tool(server: str, tool_name: str, arguments: str = "{}") -> str:
    """调用外部 MCP 服务。

    当前可用服务:
    - ShuidiRisk: 企业司法风险/大数据查询（28 个工具，入参 company_name 必填）
      · 常用: search_risk(综合风险) / search_lawsuit(诉讼) / search_punishment(行政处罚)
              search_bankruptcy(破产) / get_legal_risk_count(风险统计)
    - playwright: 微软浏览器自动化（24 个工具）
      · 常用: browser_navigate(打开网页) / browser_click(点击) / browser_type(输入)
              browser_snapshot(页面结构) / browser_take_screenshot(截图)

    调用示例:
      call_mcp_tool(server="ShuidiRisk", tool_name="search_risk", arguments='{"company_name":"公司名"}')
      call_mcp_tool(server="playwright", tool_name="browser_navigate", arguments='{"url":"https://example.com"}')
    """
    if server not in MCP_SERVERS:
        available = ", ".join(f"{k}({v['description']})" for k, v in MCP_SERVERS.items())
        return f"未知的 MCP 服务 '{server}'。可用服务: {available}"

    if not MCP_SERVERS[server].get("enabled", True):
        return f"MCP 服务 '{server}' 未启用。"

    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except json.JSONDecodeError:
        return f"参数格式错误: {arguments}。请使用 JSON 格式。"

    last_err = None
    for attempt in range(1, 4):
        try:
            return _run_async(_call_tool_async(server, tool_name, args))
        except Exception as e:
            last_err = e
            logger.warning(f"MCP {server}/{tool_name} 第{attempt}次调用失败: {e}")
            if attempt < 3:
                time.sleep(0.5 * attempt)
    return f"[MCP {server}/{tool_name} 暂时不可用 (重试3次失败): {last_err}]"


@tool
def list_mcp_services(query: str = "") -> str:
    """列出当前可用的 MCP 外部服务及其工具。"""
    lines = ["## 可用的 MCP 外部服务\n"]
    for name, cfg in MCP_SERVERS.items():
        if not cfg.get("enabled", True):
            continue
        lines.append(f"### {name} — {cfg['description']}")
        if cfg.get("transport") == "stdio":
            lines.append(f"连接: {cfg.get('command')} {' '.join(cfg.get('args', []))}")
        else:
            lines.append(f"连接: {cfg.get('url', '')}")
        lines.append(f"协议: {cfg.get('transport', 'streamable_http')}")
        try:
            tools = _run_async(_list_tools_async(name))
            lines.append("工具列表:")
            for t in tools:
                desc = (t.description or "")[:80]
                lines.append(f"  · {t.name}: {desc}")
        except Exception as e:
            lines.append(f"（服务器暂未连接: {e}）")
        lines.append("")

    return "\n".join(lines) if len(lines) > 1 else "当前没有配置 MCP 服务。"


# ── 导出 ─────────────────────────────────────────

MCP_TOOLS = [call_mcp_tool, list_mcp_services]
