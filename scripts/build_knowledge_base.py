"""
Knowledge Base Builder — end-to-end pipeline to build the vector store.

Pipeline:
1. Load raw documents from data/raw/
2. Process each document (PDF/TXT/DOCX)
3. Chunk processed documents
4. Embed chunks
5. Store in Chroma vector store
6. Save chunked corpus as JSONL for BM25
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import argparse
from pathlib import Path

from core.document_processor import process_document
from core.chunker import DocumentChunker
from core.embedding import EmbeddingManager
from core.vectorstore import VectorStoreManager

from config import (
    DATA_RAW, DATA_PROCESSED, CHUNKS_JSONL,
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_documents(raw_dir: Path) -> list[Path]:
    """Find all supported documents in raw data directory."""
    supported = {".pdf", ".docx", ".txt", ".md"}
    docs = []
    if not raw_dir.exists():
        return []
    for f in raw_dir.rglob("*"):
        if f.suffix.lower() in supported:
            docs.append(f)
    return docs


def build_knowledge_base(
    raw_dir: Path | None = None,
    persist_dir: Path | None = None,
    rebuild: bool = False,
):
    """
    Build the complete knowledge base from raw documents.

    Args:
        raw_dir: directory with raw documents
        persist_dir: Chroma persistence directory
        rebuild: if True, delete existing collection first
    """
    raw_dir = raw_dir or DATA_RAW
    persist_dir = persist_dir or Path(CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find documents
    docs = find_documents(raw_dir)
    logger.info(f"Found {len(docs)} documents to process")

    if not docs:
        logger.warning("No documents found. Add JDs/reports to data/raw/ first.")
        return

    # 2. Initialize components
    processor = process_document  # module-level function
    chunker = DocumentChunker()
    embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL)
    embeddings = embedding_manager.get_embeddings()
    vectorstore = VectorStoreManager(persist_directory=str(persist_dir))

    if rebuild:
        vectorstore.delete_collection()
        logger.info("Deleted existing collection")

    all_chunks = []

    # 3. Process each document
    for doc_path in docs:
        logger.info(f"Processing: {doc_path.name}")
        try:
            # Process
            processed = processor(str(doc_path))
            logger.info(f"  → {len(processed)} document units")

            # Chunk
            chunks = chunker.chunk_documents(processed)
            logger.info(f"  → {len(chunks)} chunks")
            all_chunks.extend(chunks)

            # Embed & store
            vectorstore.build_from_chunks(chunks, embeddings)
            logger.info(f"  → stored in vector store")
        except Exception as e:
            logger.error(f"Failed to process {doc_path.name}: {e}")

    # 4. Save chunked corpus for BM25
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    stats = vectorstore.get_collection_stats()
    logger.info(f"\nKnowledge base built successfully!")
    logger.info(f"  Total chunks: {len(all_chunks)}")
    logger.info(f"  Collection: {stats}")
    logger.info(f"  Chunks saved: {CHUNKS_JSONL}")


def main():
    parser = argparse.ArgumentParser(description="Build JobSense knowledge base")
    parser.add_argument(
        "--raw-dir", type=str, default=None,
        help="Directory with raw documents"
    )
    parser.add_argument(
        "--persist-dir", type=str, default=None,
        help="Chroma persistence directory"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete existing collection before building"
    )
    args = parser.parse_args()

    build_knowledge_base(
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        persist_dir=Path(args.persist_dir) if args.persist_dir else None,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
