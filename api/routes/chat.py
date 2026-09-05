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
MAX_TURNS = 10

def init_chat_route(agent, llm=None):
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
        from core.database import get_history, save_message, get_user_memory
        history = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in get_history(tid)]
        msg_text = request.message

        # Inject long-term user memory
        memory = get_user_memory(tid)
        if memory:
            context = "【用户画像】" + "；".join(f"{k}:{v}" for k,v in memory.items())
            msg_text = f"{context}\n\n{msg_text}"


        messages = list(history[-(MAX_TURNS * 2):])
        messages.append(HumanMessage(content=msg_text))
        result = _agent.invoke({"messages": messages})
        final_answer, route, tool_calls = _extract_result(result)
        save_message(tid, "user", request.message)
        save_message(tid, "assistant", final_answer)
        return ChatResponse(answer=final_answer or "error", intent=f"auto ({route})", sources=[], tool_calls=tool_calls)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if _agent is None: raise HTTPException(status_code=503, detail="Agent not initialized")
    tid = request.thread_id or "default"
    from core.database import get_history, save_message
    history = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in get_history(tid)]
    msg_text = request.message

    async def event_stream():
        nonlocal msg_text
        try:
            yield f"data: {json.dumps({'type':'thinking'})}\n\n"

            messages = list(history[-(MAX_TURNS * 2):])
            messages.append(HumanMessage(content=msg_text))

            result = _agent.invoke({"messages": messages})
            final_answer, route, tool_calls = _extract_result(result)
            yield f"data: {json.dumps({'type':'route','route':route})}\n\n"
            if tool_calls:
                yield f"data: {json.dumps({'type':'tools','tools':tool_calls})}\n\n"
            if final_answer:
                for c in final_answer:
                    yield f"data: {json.dumps({'type':'token','content':c})}\n\n"
                    import asyncio; await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'type':'done'})}\n\n"
            save_message(tid, "user", request.message)
            save_message(tid, "assistant", final_answer)
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})
