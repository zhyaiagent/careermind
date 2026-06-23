"""
JobSense FastAPI Application — main entry point.

Initializes all components and the ReAct Agent (LLM autonomous tool selection).
"""
import os
import json
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    API_HOST, API_PORT, LOG_LEVEL, LLM_MODEL, EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR, CHUNKS_JSONL, DATABASE_URL,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Persistent Browser State (module-level) ─────
_pw_inst = None
_browser_inst = None
_browser_page = None


def _ensure_browser():
    """Start or reuse a browser instance."""
    global _pw_inst, _browser_inst, _browser_page
    from playwright.sync_api import sync_playwright
    if _browser_inst is None or not _browser_inst.is_connected():
        _pw_inst = sync_playwright().start()
        _browser_inst = _pw_inst.chromium.launch(headless=False)
        _browser_page = _browser_inst.new_page()
    return _browser_page


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="JobSense API",
        version="2.0.0",
        description="JobSense — AI Agent with autonomous tool selection (ReAct)",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    logger.info("Initializing JobSense components...")

    # ── Embedding ──────────────────────────────────
    from core.embedding import EmbeddingManager
    embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL)
    embeddings = embedding_manager.get_embeddings()
    logger.info(f"Embedding: {EMBEDDING_MODEL}")

    # ── Vector Store ───────────────────────────────
    from core.vectorstore import VectorStoreManager
    vectorstore = VectorStoreManager(persist_directory=CHROMA_PERSIST_DIR)
    stats = vectorstore.get_collection_stats()
    logger.info(f"Vector store: {stats}")

    # ── Document processing ────────────────────────
    from core.document_processor import process_document
    from core.chunker import DocumentChunker
    chunker = DocumentChunker()

    # ── BM25 corpus ────────────────────────────────
    bm25_corpus = []
    if CHUNKS_JSONL.exists():
        with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
            bm25_corpus = [json.loads(line) for line in f if line.strip()]

    # ── Hybrid Retriever ───────────────────────────
    from core.retrieval import HybridRetriever
    retriever = HybridRetriever(vectorstore=vectorstore, embeddings=embeddings, bm25_corpus=bm25_corpus)
    logger.info(f"Retriever ready (BM25: {len(bm25_corpus)} docs)")

    # ── LLM ────────────────────────────────────────
    from core.generation import GenerationManager
    generator = GenerationManager(model_name=LLM_MODEL)
    logger.info(f"LLM: {LLM_MODEL}")

    # ── Web Search ─────────────────────────────────
    from agent.tools.web_search import WebSearchTool
    web_search_tool = WebSearchTool(api_key=os.getenv("TAVILY_API_KEY", ""))
    if web_search_tool.search_client:
        logger.info("Web search: ENABLED (Tavily)")
    else:
        logger.info("Web search: disabled")

    # ── Inject tool dependencies ───────────────────
    from agent.tools.tools import inject_dependencies
    inject_dependencies(
        retriever=retriever,
        generator=generator,
        web_search=web_search_tool,
        salary_db_path="",
        embedding_manager=embedding_manager,
    )

    # ── Build Hybrid Agent ──────────────────────────
    from agent.graph import build_agent_graph
    from agent.memory import ConversationMemory

    memory = ConversationMemory()
    # No checkpointer — chat route manages clean history manually
    agent_graph = build_agent_graph(llm=generator.llm, checkpointer=None)
    logger.info("Hybrid Agent compiled (ReAct + Plan-Execute, 8 tools)")

    # ── Register Routes ────────────────────────────
    from api.routes.chat import router as chat_router, init_chat_route
    from api.routes.upload import router as upload_router, init_upload_route
    from api.routes.evaluation import router as eval_router

    init_chat_route(agent_graph, memory, generator.llm)
    init_upload_route(process_document, chunker, embedding_manager, vectorstore, retriever, bm25_corpus)

    app.include_router(chat_router)
    app.include_router(upload_router)
    app.include_router(eval_router)

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "2.0.0", "agent": "Hybrid(ReAct+PlanExecute)", "tools": 8, "mcp_tools": 4}

    @app.get("/browser")
    async def open_browser(url: str = "https://www.baidu.com", search: str = ""):
        """浏览器控制 — 独立线程，绕过asyncio"""
        import traceback, threading, queue

        result_queue = queue.Queue()

        def _sync():
            try:
                from playwright.sync_api import sync_playwright
                import time
                pw = sync_playwright().start()
                b = pw.chromium.launch(headless=False, args=[
                    "--start-fullscreen", "--window-size=1920,1080", "--no-first-run", "--no-default-browser-check"
                ])
                ctx = b.new_context(viewport={"width": 1920, "height": 1080}, no_viewport=False)
                p = ctx.new_page()
                if search:
                    p.goto("https://www.baidu.com", timeout=30000, wait_until="commit")
                    time.sleep(2)
                    p.keyboard.press("Tab")
                    time.sleep(0.5)
                    p.keyboard.type(search, delay=80)
                    time.sleep(0.5)
                    p.keyboard.press("Enter")
                    time.sleep(3)
                else:
                    p.goto(url, timeout=30000, wait_until="commit")
                    time.sleep(2)
                text = p.inner_text("body")
                title = p.title()
                def _close():
                    time.sleep(300)
                    try: p.close(); b.close(); pw.stop()
                    except: pass
                threading.Thread(target=_close, daemon=True).start()
                result_queue.put({"ok": True, "title": title, "url": url, "preview": text[:1000]})
            except Exception as e:
                result_queue.put({"ok": False, "error": str(e), "trace": traceback.format_exc()})

        t = threading.Thread(target=_sync, daemon=True)
        t.start()
        t.join(timeout=30)
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return {"ok": True, "title": "browser launched", "preview": "浏览器已启动，请查看桌面"}

    logger.info("JobSense API v2.0 ready")
    return app


def main():
    app = create_app()
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    main()
