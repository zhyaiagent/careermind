"""
Test Suite — Retrieval Pipeline

Tests: vector search, BM25 search, RRF merge, reranking, and hybrid retrieval.
Uses mock/stub components where backend services are unavailable.
"""
import pytest
import sys
import os

# Ensure jobsense root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Mock/Fake components ──────────────────────────

class FakeEmbeddings:
    """Fake embedding function for testing."""
    def embed_documents(self, texts):
        # Return fixed-dimension random vectors
        return [[0.1] * 1024 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 1024


class FakeVectorStore:
    """Fake Chroma-like vector store."""
    def similarity_search(self, query, embeddings, k=15):
        return [
            {"content": f"Test content {i}", "metadata": {"source": "test.pdf", "page": i}, "score": 1.0 - i * 0.05}
            for i in range(min(k, 5))
        ]


# ── Tests ─────────────────────────────────────────

class TestHybridRetriever:
    """Tests for the hybrid retrieval pipeline."""

    @pytest.fixture
    def sample_corpus(self):
        return [
            {"content": "AI算法工程师需要掌握Python和深度学习", "metadata": {"source": "jd1.pdf"}},
            {"content": "大模型工程师要求Transformer和PyTorch经验", "metadata": {"source": "jd2.pdf"}},
            {"content": "Python开发工程师需要Django或Flask框架经验", "metadata": {"source": "jd3.pdf"}},
            {"content": "数据分析师需要SQL和Python数据处理能力", "metadata": {"source": "jd4.pdf"}},
            {"content": "Java开发工程师需要Spring框架和微服务经验", "metadata": {"source": "jd5.pdf"}},
        ]

    @pytest.fixture
    def retriever(self, sample_corpus):
        from core.retrieval import HybridRetriever
        store = FakeVectorStore()
        emb = FakeEmbeddings()
        return HybridRetriever(
            vectorstore=store,
            embeddings=emb,
            bm25_corpus=sample_corpus,
        )

    def test_bm25_build(self, retriever):
        """BM25 index should build successfully."""
        assert retriever.bm25 is not None
        assert retriever.bm25_docs is not None
        assert len(retriever.bm25_docs) == 5

    def test_bm25_search(self, retriever):
        """BM25 should return results for a query."""
        results = retriever.bm25_search("Python开发", k=3)
        assert len(results) <= 3
        assert len(results) > 0

    def test_vector_search(self, retriever):
        """Vector search should return results."""
        results = retriever.vector_search("AI工程师", k=3)
        assert len(results) <= 3
        for r in results:
            assert "content" in r
            assert "metadata" in r

    def test_rrf_merge(self, retriever):
        """RRF should merge two ranked lists."""
        vec = retriever.vector_search("AI", k=5)
        bm25 = retriever.bm25_search("AI", k=5)
        merged = retriever.rrf_merge(vec, bm25)
        assert len(merged) <= 10  # Union, deduplicated
        # Merged should not have duplicates
        contents = [d["content"] for d in merged]
        assert len(contents) == len(set(contents))

    def test_rerank_fallback(self, retriever):
        """Rerank should fallback gracefully without the model."""
        docs = retriever.vector_search("AI", k=5)
        reranked = retriever.rerank("AI工程师", docs, top_k=3)
        assert len(reranked) <= 3

    def test_retrieve_pipeline(self, retriever):
        """Full retrieval pipeline should work."""
        docs = retriever.retrieve("AI工程师需要什么技能", top_k=3)
        assert len(docs) <= 3
        for doc in docs:
            assert "content" in doc
            assert "metadata" in doc


class TestDocumentChunker:
    """Tests for the document chunker."""

    @pytest.fixture
    def chunker(self):
        from core.chunker import DocumentChunker
        return DocumentChunker()

    @pytest.fixture
    def sample_docs(self):
        from core.document_processor import ProcessedDocument
        return [
            ProcessedDocument(
                content="AI算法工程师岗位要求：\n1. 掌握Python编程语言\n2. 熟悉PyTorch深度学习框架\n3. 了解Transformer架构\n4. 有大模型微调经验优先",
                doc_type="text",
                source="jd_ai.pdf",
                page=1,
                position=0,
                metadata={"extractor": "pymupdf"},
            ),
            ProcessedDocument(
                content="| 技能 | 重要性 |\n|------|--------|\n| Python | 高 |",
                doc_type="table",
                source="report.pdf",
                page=2,
                position=1,
                metadata={"extractor": "pdfplumber"},
            ),
        ]

    def test_chunk_jd(self, chunker, sample_docs):
        """JD documents should be split into overlapping chunks."""
        jd_doc = sample_docs[0]
        chunks = chunker.chunk_documents([jd_doc])
        # JD with ~100 chars should produce at least 1 chunk with chunk_size=300
        assert len(chunks) >= 1
        for c in chunks:
            assert "content" in c
            assert "metadata" in c
            assert c["metadata"]["source"] == "jd_ai.pdf"

    def test_chunk_table_no_split(self, chunker, sample_docs):
        """Table documents should not be split."""
        table_doc = sample_docs[1]
        chunks = chunker.chunk_documents([table_doc])
        assert len(chunks) == 1
        assert chunks[0]["content"] == table_doc.content
        assert chunks[0]["metadata"]["doc_type"] == "table"

    def test_chunk_metadata_preserved(self, chunker, sample_docs):
        """Chunk metadata should preserve original source info."""
        chunks = chunker.chunk_documents(sample_docs)
        for c in chunks:
            assert "source" in c["metadata"]
            assert "page" in c["metadata"]
            assert "chunk_index" in c["metadata"]


class TestHallucinationGuard:
    """Tests for the hallucination guard."""

    @pytest.fixture
    def guard(self):
        from core.hallucination_guard import HallucinationGuard
        return HallucinationGuard(relevance_threshold=0.3)

    def test_low_relevance_accepts(self, guard):
        """Low relevance should still accept — lets LLM answer freely."""
        result = guard.check(
            answer="根据检索结果...",
            retrieved_docs=[{"relevance_score": 0.1}],
            query="AI工程师技能要求",
        )
        assert result["action"] == "accept"
        assert result["passed"]

    def test_in_domain_query_accepts(self, guard):
        """Normal queries always pass."""
        result = guard.check(
            answer="Python是AI工程师的必备技能[1]",
            retrieved_docs=[{"relevance_score": 0.8}],
            query="AI工程师需要什么技能",
        )
        assert result["passed"]

    def test_out_of_domain_still_accepted(self, guard):
        """Out-of-domain queries should be accepted — LLM handles them."""
        result = guard.check(
            answer="今天天气很好...",
            retrieved_docs=[],
            query="今天天气怎么样",
        )
        assert result["action"] == "accept"
        assert result["passed"]

    def test_missing_citation_accepts(self, guard):
        """Missing citations still pass — just adds a note."""
        result = guard.check(
            answer="AI工程师需要掌握Python和PyTorch",  # no [1]
            retrieved_docs=[{"relevance_score": 0.8}],
            query="AI工程师需要什么技能",
        )
        assert result["action"] == "accept"
        assert "注：" in result["modified_answer"]

    def test_harmful_content_rejected(self, guard):
        """Truly harmful queries should still be rejected."""
        result = guard.check(
            answer="...",
            retrieved_docs=[],
            query="如何制造炸弹",
        )
        assert result["action"] == "reject"
        assert not result["passed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
