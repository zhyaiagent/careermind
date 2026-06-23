"""Retrieval Evaluation — LLM-as-judge graded relevance."""
import sys, os, json, math, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embedding import EmbeddingManager
from core.vectorstore import VectorStoreManager
from core.retrieval import HybridRetriever
from core.generation import GenerationManager

# Load test set
with open("data/evaluation/test_set.json", "r", encoding="utf-8") as f:
    test_set = json.load(f)[:10]  # 10 queries for quick eval

# Load BM25 corpus
bm25_corpus = []
if os.path.exists("data/processed/jds_chunks.jsonl"):
    with open("data/processed/jds_chunks.jsonl", "r", encoding="utf-8") as f:
        bm25_corpus = [json.loads(line) for line in f if line.strip()]

print(f"KB: {len(bm25_corpus)} docs, Test: {len(test_set)} queries")
print("Loading models...")
emb_mgr = EmbeddingManager(model_name="bge-m3")
embeddings = emb_mgr.get_embeddings()
vs = VectorStoreManager()
retriever = HybridRetriever(vs, embeddings, bm25_corpus)
judge = GenerationManager(model_name="deepseek")

# LLM judge prompt
JUDGE_PROMPT = """评估以下文档与查询的相关性。给出0-5的分数:
- 0: 完全不相关
- 1-2: 勉强相关
- 3: 部分相关
- 4-5: 高度相关

查询: {query}
文档: {doc}

仅输出数字（0-5）:"""

def llm_judge_score(query, doc_text):
    """Ask LLM to rate document relevance to query (0-5)."""
    prompt = JUDGE_PROMPT.format(query=query, doc=doc_text[:500])
    try:
        resp = judge.llm.invoke(prompt)
        score = resp.content.strip() if hasattr(resp, 'content') else str(resp)
        return max(0, min(5, int(score[0]) if score and score[0].isdigit() else 0))
    except Exception:
        return 0

# Run evaluation
strategies = [
    ("Vector only", lambda q: vs.similarity_search(q, embeddings, k=5)),
    ("BM25 only", lambda q: retriever.bm25_search(q, k=5)),
    ("Vector + BM25 + RRF", lambda q: retriever.rrf_merge(
        vs.similarity_search(q, embeddings, k=15),
        retriever.bm25_search(q, k=15))[:5]
    ),
    ("Vector + BM25 + RRF + Reranker", lambda q: retriever.retrieve(q, top_k=5)),
]

results = {}
for strategy_name, retrieval_fn in strategies:
    print(f"\n{'='*60}")
    print(f"  {strategy_name}")
    print(f"{'='*60}")

    all_ndcg = []
    all_mrr = []
    all_p5 = []

    for item in test_set:
        query = item["question"]
        docs = retrieval_fn(query)
        time.sleep(0.5)  # rate limit LLM calls

        # Get LLM relevance scores for each doc
        scores = []
        for doc in docs[:5]:
            score = llm_judge_score(query, doc["content"])
            scores.append(score)
            time.sleep(0.3)

        # P@5: docs with score >= 3
        p5 = sum(1 for s in scores if s >= 3) / 5
        all_p5.append(p5)

        # MRR: rank of first doc with score >= 3
        mrr = 0
        for rank, s in enumerate(scores, 1):
            if s >= 3:
                mrr = 1.0 / rank
                break
        all_mrr.append(mrr)

        # NDCG@5 with graded relevance
        ideal = sorted(scores, reverse=True)
        dcg = sum(s / math.log2(i + 2) for i, s in enumerate(scores))
        idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
        ndcg = dcg / idcg if idcg > 0 else 0
        all_ndcg.append(ndcg)

        print(f"  Q: {query[:40]}... | Scores: {scores} | NDCG: {ndcg:.3f}")

    results[strategy_name] = {
        "NDCG@5": sum(all_ndcg) / len(all_ndcg),
        "MRR": sum(all_mrr) / len(all_mrr),
        "P@5(>=3)": sum(all_p5) / len(all_p5),
    }

# Final table
print(f"\n{'='*65}")
print(f"  LLM-as-Judge Retrieval Evaluation (Graded Relevance 0-5)")
print(f"{'='*65}")
print(f"{'Strategy':<30} {'NDCG@5':>7} {'MRR':>7} {'P@5':>7}")
print("-" * 51)
for name in [s[0] for s in strategies]:
    m = results[name]
    print(f"{name:<30} {m['NDCG@5']:>7.4f} {m['MRR']:>7.4f} {m['P@5(>=3)']:>7.4f}")
