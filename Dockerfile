FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    langchain langgraph langchain-openai langchain-community langchain-huggingface \
    langchain-chroma langchain-tavily langchain-text-splitters \
    chromadb \
    "sentence-transformers>=5.0" \
    rank-bm25 jieba \
    fastapi uvicorn[standard] streamlit requests \
    pydantic python-dotenv pandas \
    pytest \
    playwright beautifulsoup4 \
    && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install Playwright with Chromium
RUN playwright install chromium && playwright install-deps chromium

COPY . .

EXPOSE 8001 8501 9020

CMD ["python", "-m", "api.main"]
