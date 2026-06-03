#!/usr/bin/env python
"""
JobTools MCP Server — MCP 协议本地服务，提供求职相关工具。

协议: MCP (Model Context Protocol) JSON-RPC over HTTP
启动: python scripts/mcp_server.py
端口: 9020
端点: http://127.0.0.1:9020/mcp
"""
import json
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="JobTools MCP Server")


# ═══════════════════════════════════════════════════
#  Tool Implementations
# ═══════════════════════════════════════════════════

def get_interview_tips(role: str, level: str = "中级") -> str:
    tips_db = {
        "AI工程师": {
            "算法": "Transformer原理、Attention机制、RLHF/DPO对齐方法",
            "工程": "PyTorch分布式训练、模型量化部署、推理优化",
            "项目": "准备1-2个深度项目，说清楚模型选型→数据处理→训练→评估全链路",
        },
        "Python开发": {
            "基础": "Python高级特性(装饰器/生成器/协程)、GIL机制、内存管理",
            "框架": "Django中间件原理、FastAPI异步处理、ORM优化",
            "系统": "数据库索引优化、Redis数据结构应用、消息队列选型",
        },
        "大模型应用": {
            "核心": "RAG架构设计、Prompt Engineering、向量数据库选型",
            "进阶": "Agent设计模式、Tool Use编排、多Agent协作",
            "项目": "展示LangChain/LangGraph实战经验，能讲清楚为什么这样设计Agent",
        },
        "数据分析": {
            "技能": "SQL复杂查询、Python Pandas高级操作、A/B测试设计",
            "工具": "Tableau/PowerBI看板设计、数据仓库概念",
            "业务": "通过数据分析驱动业务决策的案例",
        },
    }
    role_tips = tips_db.get(role, {
        "通用": f"{role}岗位面试建议: 1.扎实的技术基础 2.有深度的项目经验 3.清晰的职业规划",
        "准备": "研究目标公司业务，准备针对性问题，复盘过往项目亮点",
        "软技能": "沟通表达能力、问题解决思路、团队协作案例",
    })
    lines = [f"## {role} 面试准备 ({level})"]
    for cat, content in role_tips.items():
        lines.append(f"\n### {cat}\n{content}")
    return "\n".join(lines)


def calculate_after_tax(monthly_salary: float, city: str = "北京") -> str:
    rates_map = {
        "北京": 0.225, "上海": 0.175, "深圳": 0.155,
        "杭州": 0.175, "广州": 0.175, "成都": 0.175,
    }
    base_cap = {
        "北京": 35283, "上海": 36549, "深圳": 34612,
        "杭州": 31215, "广州": 29876, "成都": 26547,
    }
    rate = rates_map.get(city, 0.225)
    cap = base_cap.get(city, 35283)
    base_salary = min(monthly_salary, cap)
    insurance = base_salary * rate
    taxable = monthly_salary - insurance - 5000

    if taxable <= 0: tax = 0
    elif taxable <= 3000: tax = taxable * 0.03
    elif taxable <= 12000: tax = taxable * 0.10 - 210
    elif taxable <= 25000: tax = taxable * 0.20 - 1410
    elif taxable <= 35000: tax = taxable * 0.25 - 2660
    elif taxable <= 55000: tax = taxable * 0.30 - 4410
    elif taxable <= 80000: tax = taxable * 0.35 - 7160
    else: tax = taxable * 0.45 - 15160

    after_tax = monthly_salary - insurance - tax
    return (
        f"税前 {monthly_salary:,.0f} 元/月，{city}税后到手:\n"
        f"五险一金: -{insurance:,.0f} | 个税: -{tax:,.0f}\n"
        f"实际到手: {after_tax:,.0f} 元/月 | 年收入: {after_tax*12:,.0f} 元"
    )


def get_company_info(company_name: str) -> str:
    companies = {
        "字节跳动": {"行业": "互联网", "规模": "10万+", "base": "北京/上海/深圳/杭州",
                  "AI方向": "豆包大模型、推荐算法", "薪资": "AI岗35-80K"},
        "阿里巴巴": {"行业": "互联网/云计算", "规模": "20万+", "base": "杭州/北京/上海",
                  "AI方向": "通义大模型、电商AI", "薪资": "AI岗30-65K"},
        "腾讯": {"行业": "互联网/社交", "规模": "10万+", "base": "深圳/北京/上海",
                "AI方向": "混元大模型、游戏AI", "薪资": "AI岗30-60K"},
    }
    info = companies.get(company_name)
    if not info:
        return f"未找到{company_name}。支持: {', '.join(companies.keys())}"
    return "\n".join(f"- **{k}**: {v}" for k, v in info.items())


# ═══════════════════════════════════════════════════
#  Tool 4-6: Web Scraper (requests + BS4 fallback)
# ═══════════════════════════════════════════════════

import requests as _http
from bs4 import BeautifulSoup
import base64, time, threading

# ── Persistent Browser ───────────────────────────
_pw = None
_browser = None
_page = None

def _ensure_browser():
    global _pw, _browser, _page
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    if _browser is None or not _browser.is_connected():
        try:
            if _pw: _pw.stop()
        except: pass
        _pw = sync_playwright().start()
        _browser = _pw.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
        _page = _browser.new_page(viewport={"width": 1280, "height": 900})
        _page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return _page

def browser_action(action: str, **kwargs) -> str:
    """
    浏览器多步操作工具。Agent 通过此工具逐步操控浏览器。

    action 类型:
      - navigate: 打开网址             kwargs: url
      - click: 点击页面元素            kwargs: text (按钮文字) 或 selector (CSS选择器)
      - type: 在输入框输入文字         kwargs: text
      - press: 按键盘按键             kwargs: key (如 "Enter", "Tab")
      - get_content: 获取当前页面文字
      - screenshot: 获取当前页面文字+截图描述
      - wait: 等待(用户操作/页面加载)  kwargs: seconds (默认5), reason (如"等待用户扫码")
      - scroll: 滚动                   kwargs: direction ("up"/"down"), amount (像素)
      - search: 在百度/Bing搜索       kwargs: query
    """
    page = _ensure_browser()
    if page is None:
        return "浏览器未就绪。请运行: pip install playwright && python -m playwright install chromium"

    try:
        if action == "navigate":
            url = kwargs.get("url", "https://www.baidu.com")
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            time.sleep(1)
            return f"已打开: {page.title()}\n{page.inner_text('body')[:2000]}"

        elif action == "click":
            text = kwargs.get("text", "")
            selector = kwargs.get("selector", "")
            if text:
                page.click(f"text={text}", timeout=5000)
            elif selector:
                page.click(selector, timeout=5000)
            else:
                return "请提供 text 或 selector"
            time.sleep(1)
            return f"已点击 '{text or selector}'。当前页面: {page.inner_text('body')[:1000]}"

        elif action == "type":
            txt = kwargs.get("text", "")
            # Strategy 1: Find by role
            try:
                page.get_by_role("textbox").first.fill(txt)
                return f"已填入输入框: {txt[:80]}"
            except Exception:
                pass
            # Strategy 2: Find textarea
            try:
                page.locator("textarea").first.fill(txt)
                return f"已填入textarea: {txt[:80]}"
            except Exception:
                pass
            # Strategy 3: Find by common placeholder patterns
            for ph in ["发送", "消息", "输入", "提问", "message", "chat", "search", "搜索"]:
                try:
                    page.locator(f"[placeholder*='{ph}']").first.fill(txt)
                    return f"已填入输入框({ph}): {txt[:80]}"
                except Exception:
                    continue
            # Strategy 4: Click common input area then type
            try:
                page.locator("textarea, [contenteditable='true'], input[type='text']").first.click()
                time.sleep(0.3)
                page.keyboard.type(txt, delay=50)
                return f"已键盘输入: {txt[:80]}"
            except Exception:
                page.keyboard.type(txt, delay=50)
                return f"已键盘输入(fallback): {txt[:80]}"

        elif action == "press":
            key = kwargs.get("key", "Enter")
            page.keyboard.press(key)
            time.sleep(1)
            return f"已按 {key}。当前页面: {page.inner_text('body')[:1000]}"

        elif action == "get_content":
            return f"【{page.title()}】\n{page.inner_text('body')[:3000]}"

        elif action == "screenshot":
            import os, time
            os.makedirs("data/screenshots", exist_ok=True)
            path = f"data/screenshots/shot_{int(time.time())}.png"
            page.screenshot(path=path, full_page=False)
            text = page.inner_text("body")
            return f"截图已保存到 {path}。页面文字:\n{text[:3000]}"

        elif action == "wait":
            secs = int(kwargs.get("seconds", 5))
            reason = kwargs.get("reason", "")
            msg = f"等待 {secs} 秒" + (f" ({reason})" if reason else "")
            time.sleep(secs)
            return f"{msg}。当前页面: {page.inner_text('body')[:1000]}"

        elif action == "scroll":
            direction = kwargs.get("direction", "down")
            amount = int(kwargs.get("amount", 500))
            page.mouse.wheel(0, amount if direction == "down" else -amount)
            return f"已滚动{direction} {amount}px"

        elif action == "search":
            from urllib.parse import quote
            q = kwargs.get("query", "")
            page.goto(f"https://www.bing.com/search?q={quote(q)}", timeout=10000, wait_until="domcontentloaded")
            time.sleep(2)
            return f"【搜索: {q}】\n{page.inner_text('body')[:3000]}"

        else:
            return f"未知 action: {action}。支持: navigate/click/type/press/get_content/screenshot/wait/scroll/search"

    except Exception as e:
        return f"浏览器操作失败 ({action}): {e}"


# ═══════════════════════════════════════════════════
#  MCP JSON-RPC Handler
# ═══════════════════════════════════════════════════

TOOLS_SCHEMA = [
    {
        "name": "get_interview_tips",
        "description": "获取指定岗位的面试准备建议",
        "inputSchema": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "岗位名称"},
                "level": {"type": "string", "description": "级别，默认'中级'"}
            },
            "required": ["role"]
        }
    },
    {
        "name": "calculate_after_tax",
        "description": "计算税前月薪的税后到手薪资（含五险一金）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "monthly_salary": {"type": "number", "description": "税前月薪(元)"},
                "city": {"type": "string", "description": "城市，默认'北京'"}
            },
            "required": ["monthly_salary"]
        }
    },
    {
        "name": "get_company_info",
        "description": "查询公司基本信息",
        "inputSchema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "公司名称"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "browser_action",
        "description": "浏览器多步操控工具。action类型: navigate(打开网址,需url)/click(点击,需text或selector)/type(输入,需text)/press(按键,需key)/get_content(获取页面文字)/screenshot(截图看页面)/wait(等待,可设seconds和reason)/scroll(滚动)/search(Bing搜索,需query)。多步骤任务请逐步调用此工具。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "操作: navigate/click/type/press/get_content/screenshot/wait/scroll/search"},
                "url": {"type": "string", "description": "网址(navigate时必填)"},
                "text": {"type": "string", "description": "按钮文字(click时)或输入内容(type时)"},
                "selector": {"type": "string", "description": "CSS选择器(click时可选)"},
                "key": {"type": "string", "description": "按键名称(press时,如Enter/Tab)"},
                "query": {"type": "string", "description": "搜索关键词(search时)"},
                "seconds": {"type": "integer", "description": "等待秒数(wait时,默认5)"},
                "reason": {"type": "string", "description": "等待原因(wait时,如'等待用户扫码登录')"}
            },
            "required": ["action"]
        }
    },
]

TOOL_HANDLERS = {
    "get_interview_tips": lambda args: get_interview_tips(**args),
    "calculate_after_tax": lambda args: calculate_after_tax(**args),
    "get_company_info": lambda args: get_company_info(**args),
    "browser_action": lambda args: browser_action(**args),
}


@app.post("/mcp")
async def mcp_handler(request: Request):
    """MCP JSON-RPC over HTTP (streamable_http transport)."""
    body = await request.json()
    method = body.get("method", "")
    req_id = body.get("id")

    if method == "initialize":
        resp = JSONResponse({
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "JobTools", "version": "1.0"}
            }
        })
        resp.headers["Mcp-Session-Id"] = "jobsense-session-001"
        return resp

    elif method == "notifications/initialized":
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {}})

    elif method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0", "id": req_id,
            "result": {"tools": TOOLS_SCHEMA}
        })

    elif method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return JSONResponse({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}
            })

        try:
            # Run browser_action in thread to avoid asyncio conflict
            if tool_name == "browser_action":
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(handler, arguments)
                    result_text = future.result(timeout=30)
            else:
                result_text = handler(arguments)
            return JSONResponse({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}]
                }
            })
        except Exception as e:
            return JSONResponse({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32000, "message": str(e)}
            })

    else:
        return JSONResponse({
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        })


@app.get("/health")
async def health():
    return {"status": "ok", "server": "JobTools MCP", "tools": len(TOOLS_SCHEMA)}


if __name__ == "__main__":
    print("[JobTools MCP Server] http://127.0.0.1:9020/mcp")
    print(f"   Tools: {' / '.join(TOOL_HANDLERS.keys())}")
    uvicorn.run(app, host="127.0.0.1", port=9020)
