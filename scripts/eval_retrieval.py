"""Retrieval Evaluation — LLM-as-Judge graded relevance (0-5 scale)."""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embedding import EmbeddingManager
from core.vectorstore import VectorStoreManager
from core.retrieval import HybridRetriever
from core.generation import GenerationManager

with open("data/evaluation/test_set.json", "r", encoding="utf-8") as f:
    test_set = json.load(f)[:5]

bm25_corpus = []
if os.path.exists("data/processed/jds_chunks.jsonl"):
    with open("data/processed/jds_chunks.jsonl", "r", encoding="utf-8") as f:
        bm25_corpus = [json.loads(line) for line in f if line.strip()]

print(f"KB: {len(bm25_corpus)} docs, Test: {len(test_set)} queries")

emb_mgr = EmbeddingManager(model_name="bge-m3")
embeddings = emb_mgr.get_embeddings()
vs = VectorStoreManager()
retriever = HybridRetriever(vs, embeddings, bm25_corpus)
judge = GenerationManager(model_name="deepseek")

JUDGE_PROMPT = """Rate how relevant this document is to the query (0-5):
- 0: completely irrelevant
- 1-2: tangentially related
- 3: partially relevant
- 4-5: highly relevant (directly answers the query)

Query: {query}
Document: {doc}

Output only a number (0-5):"""

def judge_score(query, doc_text):
    prompt = JUDGE_PROMPT.format(query=query, doc=doc_text[:400])
    try:
        resp = judge.llm.invoke(prompt)
        s = resp.content.strip() if hasattr(resp, 'content') else str(resp)
        return max(0, min(5, int(s[0]) if s and s[0].isdigit() else 0))
    except:
        return 0

strategies = [
    ("Vector+BM25+RRF", lambda q: retriever.rrf_merge(
        vs.similarity_search(q, embeddings, k=15),
        retriever.bm25_search(q, k=15))[:5]),
    ("Full (+Reranker)", lambda q: retriever.retrieve(q, top_k=5)),
]

results = {}
for name, fn in strategies:
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    all_ndcg, all_mrr, all_p5 = [], [], []
    for item in test_set:
        query = item["question"]
        docs = fn(query)
        time.sleep(0.3)
        scores = [judge_score(query, d["content"]) for d in docs[:5]]
        p5 = sum(1 for s in scores if s >= 3) / 5
        mrr = next((1/(i+1) for i, s in enumerate(scores) if s >= 3), 0)
        ideal = sorted(scores, reverse=True)
        dcg = sum(s/math.log2(i+2) for i, s in enumerate(scores))
        idcg = sum(s/math.log2(i+2) for i, s in enumerate(ideal))
        ndcg = dcg/idcg if idcg > 0 else 0
        all_ndcg.append(ndcg); all_mrr.append(mrr); all_p5.append(p5)
        print(f"  Q: {query[:35]}... scores={scores} NDCG={ndcg:.3f}")
    results[name] = {
        "NDCG@5": sum(all_ndcg)/len(all_ndcg),
        "MRR": sum(all_mrr)/len(all_mrr),
        "P@5(>=3)": sum(all_p5)/len(all_p5),
    }

print(f"\n{'='*65}")
print(f"  LLM-as-Judge Evaluation (Graded Relevance 0-5)")
print(f"{'='*65}")
print(f"{'Strategy':<22} {'NDCG@5':>7} {'MRR':>7} {'P@5':>7}")
print("-"*43)
for name in [s[0] for s in strategies]:
    m = results[name]
    print(f"{name:<22} {m['NDCG@5']:>7.4f} {m['MRR']:>7.4f} {m['P@5(>=3)']:>7.4f}")
