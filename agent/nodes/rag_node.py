"""
RAG Node — retrieval-augmented generation for JD analysis and domain Q&A.

Pipeline:
1. Retrieve top-k documents via HybridRetriever
2. Generate answer via GenerationManager with citations
3. Return retrieved_docs, raw_answer, sources, and iteration count
"""
from core.retrieval import HybridRetriever
from core.generation import GenerationManager


class RAGNode:
    """
    LangGraph node for RAG-based question answering.

    Combines hybrid retrieval with LLM generation,
    producing cited, context-grounded answers.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: GenerationManager,
    ):
        self.retriever = retriever
        self.generator = generator

    def __call__(self, state: dict) -> dict:
        query = state.get("query", "")

        # 1. Retrieve relevant documents
        docs = self.retriever.retrieve(query, top_k=3)

        # 2. Generate answer
        result = self.generator.generate_rag_answer(query, docs)

        return {
            "retrieved_docs": docs,
            "reranked_docs": docs,
            "raw_answer": result["answer"],
            "sources": result["sources"],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
