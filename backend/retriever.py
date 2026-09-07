from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("data/vector_db/rice.index")
METADATA_FILE = Path("data/embeddings/rice_metadata.json")
MODEL_PATH = Path("models/bge-small-en-v1.5")

TOP_K = 5                     # changed from 3 to 5
CANDIDATE_K = 10
MIN_CHUNK_LENGTH = 100        # new constant


print("Loading embedding model...")
model = SentenceTransformer(str(MODEL_PATH))

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_FILE))

print("Loading metadata...")
with METADATA_FILE.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

print(f"Index loaded: {index.ntotal} vectors")
print(f"Metadata loaded: {len(metadata)} chunks")


def retrieve(query, top_k=TOP_K):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(query_embedding, CANDIDATE_K)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = metadata[idx]

        # New filter: skip chunks shorter than MIN_CHUNK_LENGTH
        if len(chunk["text"]) < MIN_CHUNK_LENGTH:
            continue

        results.append({
            "score": float(score),
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "section": chunk["section"],
            "text": chunk["text"]
        })

    return results


if __name__ == "__main__":
    query = input("\nAsk an agriculture question: ")

    results = retrieve(query)

    print("\n" + "=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Score:   {result['score']:.4f}")
        print(f"Chunk:   {result['chunk_id']}")
        print(f"Source:  {result['source']}")
        print(f"Section: {result['section']}")
        print(f"\n{result['text']}")
