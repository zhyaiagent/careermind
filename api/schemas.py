"""
API Schemas — Pydantic models for request/response validation.
"""
from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Chat endpoint request."""
    message: str
    thread_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    """Chat endpoint response."""
    answer: str
    intent: str
    sources: list = []
    tool_calls: list = []


class UploadRequest(BaseModel):
    """Document upload request."""
    file_content: str
    file_name: str
    file_type: str  # "pdf" / "docx" / "txt"


class UploadResponse(BaseModel):
    """Document upload response."""
    status: str
    chunks_created: int
    message: str


class EvaluationRequest(BaseModel):
    """Evaluation trigger request."""
    test_set_path: Optional[str] = None


class EvaluationResponse(BaseModel):
    """Evaluation result response."""
    status: str
    report: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"
