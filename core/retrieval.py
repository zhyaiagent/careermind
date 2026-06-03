"""
Hybrid Retriever — combines vector search + BM25 + RRF + BGE-Reranker.

Pipeline:
1. Vector search → top-15
2. BM25 keyword search → top-15
3. RRF (Reciprocal Rank Fusion) merge with k=60
4. BGE-Reranker re-ranks merged candidates
5. Return top-k (default 3) final results
"""
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class HybridRetriever:
    """
    Multi-strategy retriever for high-recall, high-precision search.

    Combines dense (vector) and sparse (BM25) retrieval,
    fused via RRF and refined by a cross-encoder reranker.
    """

    def __init__(self, vectorstore, embeddings, bm25_corpus: list[dict] | None = None):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.bm25: BM25Okapi | None = None
        self.bm25_docs: list[dict] | None = None

        if bm25_corpus:
            self.build_bm25(bm25_corpus)

    def build_bm25(self, corpus: list[dict]):
        """
        Build BM25 index over a document corpus.

        Tokenizes each document with jieba for Chinese word segmentation.
        """
        self.bm25_docs = corpus
        tokenized_corpus = [list(jieba.cut(doc["content"])) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def vector_search(self, query: str, k: int = 15) -> list[dict]:
        """Dense vector similarity search via Chroma."""
        return self.vectorstore.similarity_search(
            query, self.embeddings, k=k
        )

    def bm25_search(self, query: str, k: int = 15) -> list[dict]:
        """Sparse BM25 keyword search."""
        if self.bm25 is None or self.bm25_docs is None:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [self.bm25_docs[i] for i in top_indices]

    def rrf_merge(
        self,
        vec_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.

        RRF score(d) = Σ 1/(k + rank_i(d))

        Merges two ranked lists by summing reciprocal ranks.
        Deduplicates by content string.
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, doc in enumerate(vec_results):
            content = doc["content"]
            if content not in rrf_scores:
                rrf_scores[content] = 0.0
                doc_map[content] = doc
            rrf_scores[content] += 1.0 / (k + rank + 1)

        for rank, doc in enumerate(bm25_results):
            content = doc["content"]
            if content not in rrf_scores:
                rrf_scores[content] = 0.0
                doc_map[content] = doc
            rrf_scores[content] += 1.0 / (k + rank + 1)

        sorted_contents = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [doc_map[c] for c, _ in sorted_contents]

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """
        Re-rank candidates with BGE-Reranker (cross-encoder).

        Uses BAAI/bge-reranker-v2-m3 for high-quality relevance scoring.
        """
        if not documents:
            return []

        try:
            from langchain_community.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            model = HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-v2-m3"
            )
            reranker = CrossEncoderReranker(model=model, top_n=top_k)

            # Convert dict-style docs to LangChain Documents
            lc_docs = [
                Document(
                    page_content=d["content"],
                    metadata=d.get("metadata", {}),
                )
                for d in documents
            ]

            compressed = reranker.compress_documents(
                documents=lc_docs, query=query
            )
            return [
                {
                    "content": d.page_content,
                    "metadata": d.metadata,
                    "relevance_score": getattr(d, "relevance_score", 1.0),
                }
                for d in compressed
            ]
        except Exception:
            # Fallback: return top_k without reranking
            return documents[:top_k]

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Full retrieval pipeline.

        Returns list of:
        {
            "content": "...",
            "metadata": {"source": "...", "page": 3, ...},
            "relevance_score": 0.92
        }
        """
        # Step 1: Vector search
        vec_results = self.vector_search(query, k=15)

        # Step 2: BM25 search
        bm25_results = self.bm25_search(query, k=15)

        # Step 3: RRF merge
        merged = self.rrf_merge(vec_results, bm25_results)

        # Step 4: Rerank
        reranked = self.rerank(query, merged, top_k=top_k)

        return reranked
