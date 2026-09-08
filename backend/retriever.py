from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("data/vector_db/rice.index")
METADATA_FILE = Path("data/embeddings/rice_metadata.json")
MODEL_PATH = Path("models/bge-small-en-v1.5")

TOP_K = 5                   # changed from 3 to 5
CANDIDATE_K = 10
MIN_CHUNK_LENGTH = 100        # new constant

TOPIC_SECTIONS = {
    "irrigation": [
        "Main Field",
        "Precautions for Irrigation",
        "Alternate Wetting and Drying Irrigation (AWDI)",
        "Field Water Tube - AWDI",
        "Nursery - Irrigation Management",
    ],
    "weed": [
        "Nursery - Weed Management",
        "Mainfield: Weed management",
        "Transplanted Puddled Lowland Rice",
    ],
    "fertilizer": [
        "General",
        "Application of P fertilizer",
        "Soil Application",
        "Foliar Nutrition",
        "P & K may be through Site Specific Nutrient Management (SSNM)",
    ],
}

TOPIC_KEYWORDS = {
    "irrigation": [
        "irrigat",
        "water",
        "watering",
        "moisture",
        "drainage",
        "submergence",
    ],
    "weed": [
        "weed",
        "weeds",
        "weeding",
        "herbicide",
        "herbicides",
    ],
    "fertilizer": [
        "fertilizer",
        "fertilizers",
        "nutrient",
        "nutrients",
        "nitrogen",
        "phosphorus",
        "potassium",
        "urea",
    ],
    "seed": [
        "seed",
        "seeds",
        "sowing",
        "germination",
        "seed treatment",
    ],
    "nursery": [
        "nursery",
        "seedbed",
        "seedbeds",
        "seedling",
        "seedlings",
    ],
}

TOPIC_SOURCES = {
    "irrigation": {
        "irrigation_management.txt",
    },
    "weed": {
        "weed_management.txt",
    },
    "fertilizer": {
        "inorganic_nutrients.txt",
        "organic_nutrients.txt",
        "biofertilizers.txt",
    },
    "seed": {
        "seed_treatment.txt",
    },
    "nursery": {
        "nursery_management.txt",
    },
}

SOURCE_BOOST = 0.08


def detect_topic(query):
    query = query.lower()

    topic_scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if keyword in query:
                score += 1

        if score > 0:
            topic_scores[topic] = score

    if not topic_scores:
        return None

    return max(topic_scores, key=topic_scores.get)


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

    topic = detect_topic(query)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = metadata[idx]

        if len(chunk["text"]) < MIN_CHUNK_LENGTH:
            continue

        original_score = float(score)
        reranked_score = original_score

        if topic and chunk["source"] in TOPIC_SOURCES.get(topic, set()):
            reranked_score += SOURCE_BOOST

        results.append({
            "score": reranked_score,
            "original_score": original_score,
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "section": chunk["section"],
            "text": chunk["text"]
        })

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results[:top_k]
 
 
if __name__ == "__main__":
    query = input("\nAsk an agriculture question: ")

    topic = detect_topic(query)
    print(f"Detected topic: {topic}")

    results = retrieve(query)

    print("\n" + "=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Score:   {result['score']:.4f} (Original: {result['original_score']:.4f})")
        print(f"Chunk:   {result['chunk_id']}")
        print(f"Source:  {result['source']}")
        print(f"Section: {result['section']}")
        print(f"\n{result['text']}")
