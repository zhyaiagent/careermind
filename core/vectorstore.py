"""
Vector Store Manager — Chroma-based persistent vector database.

Provides:
- Build from chunks
- Similarity search
- Collection statistics
- Metadata filtering (source, chunk_index, doc_type, page)
"""
import os
from langchain_chroma import Chroma
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


class VectorStoreManager:
    """
    Manages a Chroma vector store with persistent storage.

    Supports building from chunk lists, similarity search,
    and collection inspection.
    """

    def __init__(self, persist_directory: str | None = None):
        self.persist_directory = persist_directory or CHROMA_PERSIST_DIR
        os.makedirs(self.persist_directory, exist_ok=True)

    def build_from_chunks(
        self,
        chunks: list[dict],
        embeddings,
        collection_name: str | None = None,
    ):
        """
        Build Chroma collection from document chunks.

        Each chunk is embedded and stored with its metadata.
        Uses batch size 100 for efficiency.
        """
        collection_name = collection_name or CHROMA_COLLECTION_NAME

        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [
            f"{m.get('source', 'unknown')}_p{m.get('page', 0)}_c{m.get('chunk_index', 0)}_{i}"
            for i, m in enumerate(metadatas)
        ]

        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            ids=ids,
            persist_directory=self.persist_directory,
            collection_name=collection_name,
        )
        return vectorstore

    def similarity_search(
        self,
        query: str,
        embeddings,
        k: int = 15,
        filter: dict | None = None,
        collection_name: str | None = None,
    ) -> list[dict]:
        """
        Perform similarity search over the Chroma collection.

        Returns top-k results with content and metadata.
        """
        collection_name = collection_name or CHROMA_COLLECTION_NAME

        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name,
        )

        docs = vectorstore.similarity_search_with_score(
            query, k=k, filter=filter
        )

        results = []
        for doc, score in docs:
            # Chroma returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity: 1 - distance/2 → range [0, 1], higher = better
            relevance = 1.0 - (float(score) / 2.0)
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": max(0.0, min(1.0, relevance)),
            })
        return results

    def get_collection_stats(self, collection_name: str | None = None) -> dict:
        """Get collection size and metadata summary."""
        collection_name = collection_name or CHROMA_COLLECTION_NAME
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection(collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
            }
        except Exception:
            return {"name": collection_name, "count": 0}

    def delete_collection(self, collection_name: str | None = None):
        """Delete a collection (useful for rebuilds)."""
        collection_name = collection_name or CHROMA_COLLECTION_NAME
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(collection_name)
        except Exception:
            pass
