from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama


INDEX_FILE = Path("data/vector_db/rice.index")
METADATA_FILE = Path("data/embeddings/rice_metadata.json")
MODEL_PATH = Path("models/bge-small-en-v1.5")

TOP_K = 3
MIN_CHUNK_LENGTH = 100


print("Loading embedding model...")
model = SentenceTransformer(str(MODEL_PATH))

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_FILE))

print("Loading metadata...")
with METADATA_FILE.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

print("RAG system ready.")


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

    scores, indices = index.search(
        query_embedding,
        top_k + 5
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = metadata[idx]

        if len(chunk["text"]) < MIN_CHUNK_LENGTH:
            continue

        results.append({
            "score": float(score),
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "section": chunk["section"],
            "text": chunk["text"]
        })

        if len(results) >= top_k:
            break

    return results


def generate_answer(query, results):

    context_parts = []

    for result in results:
        context_parts.append(
            f"Source: {result['source']}\n"
            f"Section: {result['section']}\n"
            f"Content: {result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an agriculture assistant specializing in rice cultivation.

Answer the user's question using ONLY the provided agriculture knowledge.

If the provided knowledge does not contain enough information to answer,
say that the available knowledge is insufficient.

Do not invent agricultural recommendations.

Agriculture knowledge:
{context}

User question:
{query}

Answer clearly and practically.
"""

    response = ollama.chat(
        model="lfm2.5-thinking:1.2b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    query = input("\nAsk an agriculture question: ")

    results = retrieve(query)

    print("\nGenerating answer...\n")

    answer = generate_answer(query, results)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES USED")
    print("=" * 60)

    for result in results:
        print(
            f"- {result['chunk_id']} | "
            f"{result['source']} | "
            f"{result['section']}"
        )
