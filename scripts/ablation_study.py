"""Retrieval Ablation Study — compare strategies and compute metrics."""
import sys, os, json, math
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embedding import EmbeddingManager
from core.vectorstore import VectorStoreManager
from core.retrieval import HybridRetriever

# Load test set
with open("data/evaluation/test_set.json", "r", encoding="utf-8") as f:
    test_set = json.load(f)

# Load BM25 corpus
bm25_corpus = []
if os.path.exists("data/processed/jds_chunks.jsonl"):
    with open("data/processed/jds_chunks.jsonl", "r", encoding="utf-8") as f:
        bm25_corpus = [json.loads(line) for line in f if line.strip()]

# Init
print("Loading embedding model...")
emb_mgr = EmbeddingManager(model_name="bge-m3")
embeddings = emb_mgr.get_embeddings()
vs = VectorStoreManager()
retriever = HybridRetriever(vs, embeddings, bm25_corpus)


def compute_metrics(queries, ground_truths, retrieval_fn, k=5):
    """Compute Precision@k, Recall@k, MRR, NDCG@k."""
    precisions, recalls, rr_scores, ndcg_scores = [], [], [], []

    for query, gt in zip(queries, ground_truths):
        docs = retrieval_fn(query)
        retrieved = [d["content"] for d in docs[:k]]

        # Ground truth keywords
        gt_keywords = set()
        for kw in gt.replace("、", ",").replace("，", ",").split(","):
            kw = kw.strip()
            if len(kw) > 1:
                gt_keywords.add(kw)

        if not gt_keywords:
            continue

        # Relevant docs: any doc containing ANY ground truth keyword
        relevant = []
        ideal_dcg = 0
        for j, doc in enumerate(retrieved):
            score = 0
            for kw in gt_keywords:
                if kw in doc:
                    score += 1
            relevant.append(1 if score > 0 else 0)
            # DCG
            if j == 0:
                ideal_dcg += 1.0  # first position gets max relevance
            else:
                ideal_dcg += 1.0 / math.log2(j + 1)

        # Precision@k
        prec = sum(relevant) / k if k > 0 else 0
        precisions.append(prec)

        # Recall@k (approximate: how many unique keywords found)
        found_kws = set()
        for j, doc in enumerate(retrieved):
            for kw in gt_keywords:
                if kw in doc:
                    found_kws.add(kw)
        rec = len(found_kws) / len(gt_keywords) if gt_keywords else 0
        recalls.append(rec)

        # MRR
        rr = 0
        for rank, rel in enumerate(relevant, 1):
            if rel:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

        # NDCG@k
        dcg = sum(rel / math.log2(j + 2) for j, rel in enumerate(relevant))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg_scores.append(ndcg)

    return {
        f"Precision@{k}": sum(precisions) / len(precisions) if precisions else 0,
        f"Recall@{k}": sum(recalls) / len(recalls) if recalls else 0,
        "MRR": sum(rr_scores) / len(rr_scores) if rr_scores else 0,
        f"NDCG@{k}": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0,
    }


# Run ablation
queries = [q["question"] for q in test_set[:15]]
gts = [q["ground_truth"] for q in test_set[:15]]

print(f"\n{'='*65}")
print(f"  Retrieval Ablation Study ({len(queries)} queries)")
print(f"{'='*65}")

# Strategy 1: Vector only
def vec_only(query):
    return vs.similarity_search(query, embeddings, k=5)

# Strategy 2: BM25 only
def bm25_only(query):
    return retriever.bm25_search(query, k=5)

# Strategy 3: Vector + BM25 (RRF merge, no reranker)
def vec_bm25(query):
    vec = vec_only(query)
    bm = bm25_only(query)
    return retriever.rrf_merge(vec, bm)[:5]

# Strategy 4: Full pipeline (Vector + BM25 + RRF + Reranker)
def full_pipeline(query):
    return retriever.retrieve(query, top_k=5)

strategies = [
    ("Vector only", vec_only),
    ("BM25 only", bm25_only),
    ("Vector + BM25 + RRF", vec_bm25),
    ("Full pipeline (+Reranker)", full_pipeline),
]

results = {}
for name, fn in strategies:
    m = compute_metrics(queries, gts, fn, k=5)
    results[name] = m
    print(f"\n  {name}:")
    for k, v in m.items():
        print(f"    {k}: {v:.4f}")

# Summary table
print(f"\n{'='*65}")
print(f"  Summary Table")
print(f"{'='*65}")
header = f"{'Strategy':<28} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'NDCG@5':>6}"
print(header)
print("-" * 52)
for name in [s[0] for s in strategies]:
    m = results[name]
    print(f"{name:<28} {m['Precision@5']:>6.4f} {m['Recall@5']:>6.4f} {m['MRR']:>6.4f} {m['NDCG@5']:>6.4f}")
