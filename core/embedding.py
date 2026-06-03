"""
Embedding Manager — provides text-to-vector embedding models.

Supports:
- BGE-M3 (local, via HuggingFace BGE)
- OpenAI text-embedding-3-small (API)
"""
from config import EMBEDDING_MODEL


class EmbeddingManager:
    """Wraps embedding model initialization and usage."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.embeddings = self._init_embeddings()

    def _init_embeddings(self):
        if self.model_name == "bge-m3":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        elif self.model_name == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            raise ValueError(f"Unsupported embedding model: {self.model_name}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def get_embeddings(self):
        return self.embeddings
