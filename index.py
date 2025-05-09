#!/usr/bin/env python3
"""
index.py

1. Reads all JSON files in data/raw_articles/
2. Builds a TF–IDF vectorizer and saves it + the document-term matrix
3. Embeds each article via SentenceTransformers and builds a FAISS index
4. Saves the FAISS index and metadata for lookup
"""

import os
import json
import glob
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths
RAW_DIR = Path("data/raw_articles")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load & preprocess documents
print("Loading raw articles…")
docs = []
meta = []  # to track title + link per doc
for path in RAW_DIR.glob("*.json"):
    with open(path) as f:
        articles = json.load(f)
    for art in articles:
        text = " ".join([art.get("title",""), art.get("summary","")]).strip()
        if not text:
            continue
        docs.append(text)
        meta.append({
            "title": art.get("title",""),
            "link":  art.get("link",""),
            "published": art.get("published","")
        })
print(f"  → {len(docs)} documents loaded.")

# 2. TF–IDF index
print("Building TF–IDF vectorizer…")
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1,2)
)
X_tfidf = tfidf.fit_transform(docs)
# Save vectorizer and matrix
with open(INDEX_DIR / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
# Optionally save sparse matrix as .npz
import scipy.sparse
scipy.sparse.save_npz(INDEX_DIR / "dt_matrix.npz", X_tfidf)
print("  → TF–IDF index saved.")

# 3. Embeddings + FAISS
print("Embedding documents with SentenceTransformer…")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, show_progress_bar=True, batch_size=32)
embeddings = np.array(embeddings).astype("float32")

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)     # exact L2 search
index.add(embeddings)
# Save FAISS index
faiss.write_index(index, str(INDEX_DIR / "faiss_index.idx"))
print("  → FAISS index saved.")

# 4. Save metadata
with open(INDEX_DIR / "metadata.pkl", "wb") as f:
    pickle.dump(meta, f)
print("  → Metadata saved.")

print("Indexing complete. Files in data/index/:")
for p in INDEX_DIR.iterdir():
    print("   ", p.name)
