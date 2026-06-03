"""
Document Chunker — splits processed documents into overlapping chunks
for embedding and retrieval.

Strategy:
- JD documents: chunk_size=300, overlap=50
- Reports/other: chunk_size=500, overlap=100
- Tables and image descriptions are kept whole (no splitting)
- Separators: ["\n\n", "\n", "。", "，", "、", " ", ""]
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import from core.document_processor for type hint
from core.document_processor import ProcessedDocument


class DocumentChunker:
    """
    Splits documents into overlapping chunks with metadata tracking.

    Each chunk carries:
      - source: original file
      - chunk_index: position in the original document
      - doc_type: text / table / image_description
      - page: page number
    """

    def __init__(self):
        self.jd_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", "、", " ", ""],
            length_function=len,
        )
        self.report_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "，", "、", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, docs: list[ProcessedDocument]) -> list[dict]:
        """
        Convert a list of ProcessedDocument into a list of chunk dicts.

        Returns list of:
        {
            "content": "...",
            "metadata": {
                "source": "xxx.pdf",
                "chunk_index": 0,
                "doc_type": "text",
                "page": 3
            }
        }
        """
        results: list[dict] = []

        for doc in docs:
            # Tables and image descriptions stay as one chunk
            if doc.doc_type in ("table", "image_description"):
                results.append({
                    "content": doc.content,
                    "metadata": {
                        "source": doc.source,
                        "chunk_index": 0,
                        "doc_type": doc.doc_type,
                        "page": doc.page,
                        **doc.metadata,
                    }
                })
            elif "jd" in doc.source.lower():
                chunks = self.jd_splitter.split_text(doc.content)
                for i, chunk in enumerate(chunks):
                    results.append({
                        "content": chunk,
                        "metadata": {
                            "source": doc.source,
                            "chunk_index": i,
                            "doc_type": doc.doc_type,
                            "page": doc.page,
                            **doc.metadata,
                        }
                    })
            else:
                chunks = self.report_splitter.split_text(doc.content)
                for i, chunk in enumerate(chunks):
                    results.append({
                        "content": chunk,
                        "metadata": {
                            "source": doc.source,
                            "chunk_index": i,
                            "doc_type": doc.doc_type,
                            "page": doc.page,
                            **doc.metadata,
                        }
                    })

        return results
