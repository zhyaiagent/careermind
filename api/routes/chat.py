"""Chat Routes — Agent controls browser via MCP browser_action tool."""
import json, logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from api.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_agent = None
_llm = None
_conversations: dict[str, list] = {}
MAX_TURNS = 10

# Persistent browser
_pw_inst = None
_browser_inst = None
_browser_page = None

BROWSER_EXECUTE_PROMPT = """你是浏览器自动化专家。将任务转为JSON步骤数组。可用action:
- navigate: {"action":"navigate","url":"网址"}
- click: 点击。参数text="文字"(按文字)/result_index=N(第N个搜索结果)/selector="CSS"
- type: 搜索 {"action":"type","text":"搜索词"} (用Bing搜索，搜索结果自动打开第一条)
- press: {"action":"press","key":"Enter"} (支持Enter/Tab/Escape)
- wait: {"action":"wait","seconds":20,"reason":"等待用户扫码登录"}
- get_content: {"action":"get_content"}
- scroll: {"action":"scroll","direction":"down","amount":500}
- screenshot: {"action":"screenshot"}

关键规则:
- 如果任务涉及登录，登录后必须: click(text="输入")或click(text="对话框")来聚焦输入区域，再wait(2秒)
- 即使用户说"问他XX"，也要写成具体的process步骤(点击→type→press，不是模糊描述)
- 输入文字后必须press("Enter")
- 最后必须是get_content
- 用户说"登录"时: navigate→click("登录")→wait(20s,"等待用户扫码")
- 输出纯JSON数组，不要任何markdown

任务: {query}
步骤:"""

async def _execute_browser_task(msg: str, llm) -> str | None:
    """General browser task executor: LLM plans → Playwright executes."""
    low = msg.lower()
    is_browser_task = any(kw in low for kw in ['浏览器', '打开', '搜索', '登录', 'deepseek', '百度', '查'])
    if not is_browser_task:
        return None

    logger.info(f"Browser task detected: {msg[:80]}")

    # Step 1: LLM generates execution plan
    plan_prompt = BROWSER_EXECUTE_PROMPT.replace("{query}", msg)
    # Auto-add click instruction with result_index hint
    if any(kw in msg for kw in ['点进去', '链接', '点击', '进去看', '打开.*搜索']):
        plan_prompt += "\n用户要求打开搜索结果。搜索结果出来后，用 click(result_index=1) 点击第一个结果，或 result_index=N 点击第N个。"
    try:
        resp = llm.invoke(plan_prompt)
        plan_text = resp.content if hasattr(resp, 'content') else str(resp)
    except Exception as e:
        return f"[浏览器规划失败: {e}]"

    # Parse JSON plan
    import re
    plan_text = plan_text.strip()
    if plan_text.startswith("```"): plan_text = re.sub(r'^```\w*\n?', '', plan_text)
    if plan_text.endswith("```"): plan_text = plan_text[:-3]
    try:
        steps = json.loads(plan_text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', plan_text, re.DOTALL)
        if match:
            try: steps = json.loads(match.group())
            except: return f"[浏览器规划解析失败]"
        else:
            return f"[浏览器规划解析失败]"

    if not isinstance(steps, list):
        return f"[浏览器规划格式错误]"

    logger.info(f"Browser plan: {len(steps)} steps: {[s.get('action','?') for s in steps]}")

    # Step 2: Execute each step directly via Playwright (not MCP, to avoid thread issues)
    global _pw_inst, _browser_inst, _browser_page

    def _exec_steps():
        import time
        global _pw_inst, _browser_inst, _browser_page
        from playwright.sync_api import sync_playwright

        # Always create fresh browser for reliable operation
        if _browser_inst is not None:
            try:
                if _browser_inst.is_connected():
                    _browser_inst.close()
            except: pass
        if _pw_inst:
            try: _pw_inst.stop()
            except: pass
        _pw_inst = sync_playwright().start()
        _browser_inst = _pw_inst.chromium.launch(
            channel="msedge", headless=False,
            args=["--start-fullscreen", "--window-size=1920,1080", "--no-first-run", "--no-default-browser-check"]
        )
        _ctx = _browser_inst.new_context(viewport={"width": 1920, "height": 1080}, no_viewport=False)
        _browser_page = _ctx.new_page()

        results = []
        for i, step in enumerate(steps):
            action = step.get("action", "")
            args = {k: v for k, v in step.items() if k != "action"}
            logger.info(f"  Step {i+1}: {action}")
            try:
                if action == "navigate":
                    _browser_page.goto(args["url"], timeout=30000, wait_until="commit")
                    time.sleep(2)
                    time.sleep(3)
                    results.append(f"[{i+1}] 已打开: {_browser_page.title()}")
                elif action == "click":
                    sel = args.get("selector", "")
                    txt = args.get("text", "")
                    n = int(args.get("result_index", 0))  # click Nth search result
                    if n > 0:
                        # Click the Nth search result link on Baidu
                        # Baidu results are in div.result h3 a or div.c-container h3 a
                        links = _browser_page.locator("h3 a, .result h3 a, .c-container h3 a, #content_left h3 a")
                        count = links.count()
                        if count >= n:
                            links.nth(n-1).click()
                            results.append(f"[{i+1}] 已点击第{n}个搜索结果 (共{count}个)")
                        else:
                            results.append(f"[{i+1}] 搜索结果只有{count}个，无法点击第{n}个")
                    elif sel:
                        _browser_page.locator(sel).first.click(timeout=5000)
                        results.append(f"[{i+1}] 已点击: {sel}")
                    elif txt:
                        try:
                            _browser_page.get_by_text(txt, exact=False).first.click(timeout=3000)
                        except Exception:
                            try:
                                _browser_page.locator(f"a:has-text('{txt}')").first.click(timeout=3000)
                            except Exception:
                                _browser_page.locator(f"text={txt}").first.click(timeout=3000)
                        results.append(f"[{i+1}] 已点击: {txt}")
                    else:
                        _browser_page.locator("textarea, [contenteditable='true']").first.click(timeout=3000)
                        results.append(f"[{i+1}] 已点击: 输入区")
                    time.sleep(1)
                elif action == "type":
                    txt = args.get("text", "")
                    from urllib.parse import quote
                    # Bing is faster and more reliable than Baidu
                    search_url = f"https://www.bing.com/search?q={quote(txt)}"
                    _browser_page.goto(search_url, timeout=20000, wait_until="commit")
                    time.sleep(2)
                    results.append(f"[{i+1}] 已搜索(Bing): {txt[:80]}")
                    # Auto-click first search result
                    click_keywords = ['点进去', '点击', '链接', '打开', '进去看', '进去']
                    if any(kw in msg for kw in click_keywords):
                        try:
                            links = _browser_page.locator("#b_results h2 a")
                            count = links.count()
                            if count > 0:
                                links.first.click()
                                time.sleep(2)
                                results.append(f"[{i+1}a] 已点击第1个搜索结果(共{count}个)")
                            else:
                                results.append(f"[{i+1}a] 未找到搜索结果链接")
                        except Exception as e:
                            results.append(f"[{i+1}a] 点击失败: {e}")
                elif action == "press":
                    _browser_page.keyboard.press(args.get("key", "Enter"))
                    time.sleep(1)
                    results.append(f"[{i+1}] 已按: {args.get('key','Enter')}")
                elif action == "wait":
                    secs = int(args.get("seconds", 5))
                    reason = args.get("reason", "")
                    results.append(f"[{i+1}] 等待 {secs}秒: {reason}")
                    time.sleep(secs)
                elif action == "get_content":
                    txt = _browser_page.inner_text("body")
                    results.append(f"[{i+1}] 页面内容:\n{txt[:3000]}")
                elif action == "screenshot":
                    import os as _os
                    _os.makedirs("data/screenshots", exist_ok=True)
                    path = f"data/screenshots/shot_{int(time.time())}.png"
                    _browser_page.screenshot(path=path)
                    results.append(f"[{i+1}] 截图: {path}")
                elif action == "scroll":
                    d = args.get("direction", "down")
                    amt = int(args.get("amount", 500))
                    _browser_page.mouse.wheel(0, amt if d == "down" else -amt)
                    results.append(f"[{i+1}] 已滚动 {d}")
                elif action == "search":
                    from urllib.parse import quote
                    q = args.get("query", "")
                    url = f"https://www.bing.com/search?q={quote(q)}"
                    _browser_page.goto(url, timeout=15000, wait_until="commit")
                    time.sleep(2)
                    results.append(f"[{i+1}] Bing搜索: {q}")
                else:
                    results.append(f"[{i+1}] 未知操作: {action}")
            except Exception as e:
                results.append(f"[{i+1}] {action} 失败: {e}")
        return "\n".join(results)

    import threading, queue
    result_queue = queue.Queue()

    def _run():
        try:
            result_queue.put(_exec_steps())
        except Exception as e:
            result_queue.put(f"执行失败: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=60)
    try:
        result = result_queue.get_nowait()
        return "浏览器任务执行完毕:\n" + result
    except queue.Empty:
        return "[浏览器任务执行中，请查看桌面]"


def init_chat_route(agent, memory=None, llm=None):
    global _agent, _llm
    _agent = agent
    _llm = llm


def _extract_result(result: dict) -> tuple[str, str, list]:
    final_answer = result.get("final_answer", "")
    route = result.get("route", "react")
    tool_calls = []
    if not final_answer:
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                final_answer = msg.content; break
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"tool": tc.get("name","?"), "args": tc.get("args",{})})
    for tr in result.get("tool_results", []):
        tool_calls.append({"tool": tr.get("tool","?"), "args": {"step": tr.get("step"), "desc": tr.get("description")}})
    return final_answer, route, tool_calls


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if _agent is None: raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        tid = request.thread_id or "default"
        history = _conversations.get(tid, [])
        msg_text = request.message

        # General browser task executor (LLM plans → Playwright executes)
        browser_result = await _execute_browser_task(msg_text, _llm)
        if browser_result:
            msg_text = (
                f"【系统通知】浏览器已执行以下操作并获得结果。你必须基于这些结果回答，不要说浏览器不可用:\n\n"
                f"{browser_result}\n\n"
                f"【用户原始问题】{request.message}\n"
                f"请根据以上浏览器返回的实际内容回答用户问题。"
            )

        messages = list(history[-(MAX_TURNS * 2):])
        messages.append(HumanMessage(content=msg_text))
        result = _agent.invoke({"messages": messages})
        final_answer, route, tool_calls = _extract_result(result)
        history.append(HumanMessage(content=request.message))
        history.append(AIMessage(content=final_answer))
        _conversations[tid] = history[-(MAX_TURNS * 2):]
        return ChatResponse(answer=final_answer or "error", intent=f"auto ({route})", sources=[], tool_calls=tool_calls)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if _agent is None: raise HTTPException(status_code=503, detail="Agent not initialized")
    tid = request.thread_id or "default"
    history = _conversations.get(tid, [])
    msg_text = request.message

    async def event_stream():
        nonlocal msg_text
        try:
            yield f"data: {json.dumps({'type':'thinking'})}\n\n"

            browser_result = await _execute_browser_task(msg_text, _llm)
            if browser_result:
                yield f"data: {json.dumps({'type':'browser','content':browser_result[:200]})}\n\n"
                msg_text = f"{request.message}\n\n{browser_result}"

            messages = list(history[-(MAX_TURNS * 2):])
            messages.append(HumanMessage(content=msg_text))

            result = _agent.invoke({"messages": messages})
            final_answer, route, tool_calls = _extract_result(result)
            yield f"data: {json.dumps({'type':'route','route':route})}\n\n"
            if tool_calls:
                yield f"data: {json.dumps({'type':'tools','tools':tool_calls})}\n\n"
            # Stream tokens character by character for smooth UX
            if final_answer:
                for c in final_answer:
                    yield f"data: {json.dumps({'type':'token','content':c})}\n\n"
                    import asyncio; await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'type':'done'})}\n\n"
            history.append(HumanMessage(content=request.message))
            history.append(AIMessage(content=final_answer))
            _conversations[tid] = history[-(MAX_TURNS * 2):]
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})
