"""
Upload Route — document ingestion endpoint.

Accepts document content (PDF/DOCX/TXT) and processes it
through the document pipeline into the vector store.
Auto-updates BM25 index so new docs are immediately searchable.
"""
import os
import json
import base64
import logging
import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException

from api.schemas import UploadRequest, UploadResponse
from config import DATA_RAW, CHUNKS_JSONL

logger = logging.getLogger(__name__)

router = APIRouter()

# Global references — set by main.py
_document_processor = None
_chunker = None
_embedding_manager = None
_vectorstore = None
_retriever = None       # HybridRetriever — to rebuild BM25 after upload
_bm25_corpus = []       # shared list for BM25, persisted to CHUNKS_JSONL


def init_upload_route(processor, chunker, embedding_manager, vectorstore, retriever, bm25_corpus):
    """Initialize upload route with core components + BM25 awareness."""
    global _document_processor, _chunker, _embedding_manager, _vectorstore, _retriever, _bm25_corpus
    _document_processor = processor
    _chunker = chunker
    _embedding_manager = embedding_manager
    _vectorstore = vectorstore
    _retriever = retriever
    _bm25_corpus = bm25_corpus if bm25_corpus is not None else []


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: UploadRequest):
    """
    Upload and process a document into the knowledge base.

    The file_content should be base64-encoded file bytes.
    After processing, both Chroma and BM25 are updated so
    the new content is immediately searchable.
    """
    if _document_processor is None:
        raise HTTPException(status_code=503, detail="Document processor not initialized")

    try:
        # Decode base64 content and save to temp file
        file_bytes = base64.b64decode(request.file_content)
        ext = Path(request.file_name).suffix or f".{request.file_type}"
        if not ext.startswith("."):
            ext = "." + ext

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, dir=str(DATA_RAW)
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            # 1. Process document (processor is the process_document function)
            docs = _document_processor(tmp_path)
            logger.info(f"Processed {request.file_name}: {len(docs)} document units")

            # 2. Chunk
            chunks = _chunker.chunk_documents(docs)
            logger.info(f"Chunked into {len(chunks)} chunks")

            # 3. Add to Chroma vector store
            embeddings = _embedding_manager.get_embeddings()
            _vectorstore.build_from_chunks(chunks, embeddings)
            logger.info("Added to Chroma vector store")

            # 4. Update BM25 corpus (both in-memory and on-disk)
            if _bm25_corpus is not None:
                _bm25_corpus.extend(chunks)
                # Rebuild BM25 index in the retriever
                if _retriever is not None:
                    _retriever.build_bm25(_bm25_corpus)
                    logger.info(f"BM25 index rebuilt ({len(_bm25_corpus)} total docs)")

                # Persist to JSONL
                CHUNKS_JSONL.parent.mkdir(parents=True, exist_ok=True)
                with open(CHUNKS_JSONL, "a", encoding="utf-8") as f:
                    for c in chunks:
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")

            return UploadResponse(
                status="success",
                chunks_created=len(chunks),
                message=(
                    f"已成功处理 {request.file_name}，生成 {len(chunks)} 个知识块。"
                    f"现在你可以针对这份文档提问了！"
                ),
            )
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
