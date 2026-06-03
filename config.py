"""
JobSense Configuration
Loads settings from .env file with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

# LLM
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

# Retrieval
RETRIEVAL_TOP_K = 3
VECTOR_SEARCH_K = 15
BM25_SEARCH_K = 15
RRF_K = 60
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Chunking
JD_CHUNK_SIZE = 300
JD_CHUNK_OVERLAP = 50
REPORT_CHUNK_SIZE = 500
REPORT_CHUNK_OVERLAP = 100

# Chroma
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data/chroma_db"))
CHROMA_COLLECTION_NAME = "jobsense"

# Salary DB
SALARY_DB_PATH = os.getenv("SALARY_DB_PATH", str(PROJECT_ROOT / "data/processed/salary.db"))

# Paths
DATA_RAW = PROJECT_ROOT / "data/raw"
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
DATA_EVALUATION = PROJECT_ROOT / "data/evaluation"
JDS_JSONL = DATA_RAW / "jds.jsonl"
CHUNKS_JSONL = DATA_PROCESSED / "jds_chunks.jsonl"
TEST_SET_PATH = DATA_EVALUATION / "test_set.json"

# Agent
MAX_ITERATIONS = 2
HALLUCINATION_RELEVANCE_THRESHOLD = 0.1
CONVERSATION_MAX_TURNS = 20

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# API
API_HOST = "0.0.0.0"
API_PORT = 8001
