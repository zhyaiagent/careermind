"""Rebuild knowledge base — pure Python, ZERO external deps beyond chromadb."""
import json, math, os, sys
from collections import Counter

print("Loading JDs...", flush=True)
with open("data/raw/jds.jsonl", "r", encoding="utf-8") as f:
    jds = [json.loads(l) for l in f if l.strip()]
print(f"  {len(jds)} JDs", flush=True)

# ── Pure Python chunking ──────────────────────────
print("Chunking...", flush=True)
chunks = []
for jd in jds:
    text = f"岗位: {jd['job_title']}\n公司: {jd['company']}\n城市: {jd['city']}\n薪资: {jd['salary_range']}\n经验: {jd['experience']}\n学历: {jd['education']}\n{jd['jd_text']}\n技能: {', '.join(jd.get('tags', []))}"
    # Simple chunking: split by double newline, then merge short ones
    parts = text.split("\n\n")
    current = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) < 500:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)

# Assign metadata
result = []
for i, c in enumerate(chunks):
    jd_idx = i * len(jds) // len(chunks)
    j = jds[min(jd_idx, len(jds)-1)]
    result.append({
        "content": c,
        "metadata": {"source": f"jd_{j['job_title']}.txt", "chunk_index": i,
                     "job_title": j["job_title"], "company": j["company"]}
    })
chunks = result
print(f"  {len(chunks)} chunks", flush=True)

# ── Pure Python TF-IDF embedding ──────────────────
print("Embedding...", flush=True)
texts = [c["content"] for c in chunks]
DIM = 1024

def tokenize(text):
    text = text.replace("\n", " ").replace("\r", " ")
    return list(text) + [text[i:i+2] for i in range(len(text)-1)]

# Build vocab
df = Counter()
tokenized_docs = [list(set(tokenize(t))) for t in texts]
for tokens in tokenized_docs:
    df.update(tokens)
vocab = {w: i for i, (w, _) in enumerate(df.most_common(DIM))}

# Compute IDF
N = len(texts)
idf = {w: math.log((N+1)/(df[w]+1)) + 1 for w in vocab}

# Embed
vecs = []
for text in texts:
    tokens = tokenize(text)
    tf = Counter(tokens)
    total = len(tokens) or 1
    vec = [0.0] * DIM
    for word, count in tf.items():
        if word in vocab:
            vec[vocab[word]] = (count / total) * idf.get(word, 1.0)
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    vecs.append([v/norm for v in vec])

print(f"  {len(vecs)} vectors, dim={DIM}", flush=True)

# ── Store in Chroma ───────────────────────────────
print("Storing...", flush=True)
import chromadb
client = chromadb.PersistentClient(path="data/chroma_db")
try:
    client.delete_collection("jobsense")
except Exception:
    pass
col = client.create_collection("jobsense")
ids = [f"jd_{i}" for i in range(len(chunks))]
col.add(embeddings=vecs, documents=texts,
        metadatas=[c["metadata"] for c in chunks], ids=ids)
print(f"  Collection: {col.count()} docs", flush=True)

# Save chunks for BM25
with open("data/processed/jds_chunks.jsonl", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

os.makedirs("data/processed", exist_ok=True)
print("Done!", flush=True)
