from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("data/vector_db/rice.index")
METADATA_FILE = Path("data/embeddings/rice_metadata.json")
MODEL_PATH = Path("models/bge-small-en-v1.5")

TOP_K = 5
CANDIDATE_K = 10
MIN_CHUNK_LENGTH = 100

# --------------------------------------------------
# Topic detection
# --------------------------------------------------

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


# --------------------------------------------------
# Intent detection
# --------------------------------------------------

INTENT_KEYWORDS = {
    "weed_management": [
        "how should",
        "how to control",
        "how can",
        "manage",
        "management",
        "control",
        "controlling",
        "prevent",
        "prevention",
        "remove",
    ],

    "weed_identification": [
        "what weeds",
        "which weeds",
        "weed species",
        "types of weeds",
        "identify weeds",
        "common weeds",
    ],

    "irrigation_timing": [
        "when",
        "how often",
        "how frequently",
        "how much water",
        "water level",
        "irrigate",
        "irrigation",
        "critical stage",
        "critical stages",
    ],

    "fertilizer_application": [
        "what fertilizer",
        "which fertilizer",
        "fertilizer",
        "fertilizers",
        "apply",
        "application",
        "how much",
        "dose",
        "nutrient",
        "nutrients",
    ],

    "seed_treatment": [
        "seed treatment",
        "treat seeds",
        "treating seeds",
        "seed preparation",
        "germination",
        "before sowing",
        "treated before sowing",
        "seed",
        "seeds",
    ],

    "nursery_management": [
        "nursery",
        "seedbed",
        "seedbeds",
        "seedling",
        "seedlings",
    ],
}


INTENT_SECTIONS = {
    "weed_management": {
        "Mainfield: Weed management",
        "Nursery - Weed Management",
    },

    "weed_identification": {
        "Transplanted Puddled Lowland Rice",
    },

    "irrigation_timing": {
        "Main Field",
        "Precautions for Irrigation",
        "Alternate Wetting and Drying Irrigation (AWDI)",
        "Field Water Tube - AWDI",
        "Nursery - Irrigation Management",
    },

    "fertilizer_application": {
        "Application of P fertilizer",
        "Soil Application",
        "Foliar Nutrition",
        "P & K may be through Site Specific Nutrient Management (SSNM)",
        "General",
    },

    "seed_treatment": {
        "Seed Treatment",
        "Seed Treatment with Biofertilizers",
        "Seed Treatment with Carrier Based Biofertilizers",
        "Specific gravity grading in rice seeds",
    },

    "nursery_management": {
        "Formation of Seedbeds",
        "Nursery - Irrigation Management",
        "Nursery - Weed Management",
    },
}


def detect_intent(query, topic=None):
    query = query.lower()

    intent_scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if keyword in query:
                score += 1

        if score > 0:
            intent_scores[intent] = score

    if not intent_scores:
        return None

    # Prefer intents belonging to the detected topic.
    if topic == "seed":
        if "seed_treatment" in intent_scores:
            return "seed_treatment"

        return "seed_treatment"

    if topic == "weed":
        if "weed_management" in intent_scores:
            return "weed_management"

        if "weed_identification" in intent_scores:
            return "weed_identification"

    if topic == "irrigation":
        if "irrigation_timing" in intent_scores:
            return "irrigation_timing"

    if topic == "fertilizer":
        if "fertilizer_application" in intent_scores:
            return "fertilizer_application"

    if topic == "nursery":
        if "nursery_management" in intent_scores:
            return "nursery_management"

    return max(intent_scores, key=intent_scores.get)


# --------------------------------------------------
# Ranking configuration
# --------------------------------------------------

SOURCE_BOOST = 0.08
SECTION_BOOST = 0.12


# --------------------------------------------------
# Load models / index
# --------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer(str(MODEL_PATH))

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_FILE))

print("Loading metadata...")
with METADATA_FILE.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

print(f"Index loaded: {index.ntotal} vectors")
print(f"Metadata loaded: {len(metadata)} chunks")


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve(query, top_k=TOP_K):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(
        query_embedding,
        CANDIDATE_K,
    )

    topic = detect_topic(query)
    intent = detect_intent(query, topic)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = metadata[idx]

        if len(chunk["text"]) < MIN_CHUNK_LENGTH:
            continue

        semantic_score = float(score)

        # ------------------------------------------
        # Source relevance
        # ------------------------------------------

        source_boost = 0.0

        if topic:
            if chunk["source"] in TOPIC_SOURCES.get(topic, set()):
                source_boost = SOURCE_BOOST

        # ------------------------------------------
        # Intent / section relevance
        # ------------------------------------------

        section_boost = 0.0

        if intent:
            preferred_sections = INTENT_SECTIONS.get(
                intent,
                set(),
            )

            if chunk["section"] in preferred_sections:
                section_boost = SECTION_BOOST

        # ------------------------------------------
        # Final ranking score
        # ------------------------------------------

        ranking_score = (
            semantic_score
            + source_boost
            + section_boost
        )

        results.append({
            "score": ranking_score,
            "semantic_score": semantic_score,
            "source_boost": source_boost,
            "section_boost": section_boost,
            "topic": topic,
            "intent": intent,
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "section": chunk["section"],
            "text": chunk["text"],
        })

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:top_k]


# --------------------------------------------------
# Command-line test
# --------------------------------------------------

if __name__ == "__main__":

    query = input("\nAsk an agriculture question: ")

    topic = detect_topic(query)
    intent = detect_intent(query, topic)

    print(f"Detected topic:  {topic}")
    print(f"Detected intent: {intent}")

    results = retrieve(query)

    print("\n" + "=" * 70)
    print("RETRIEVED CHUNKS")
    print("=" * 70)

    for i, result in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")

        print(
            f"Ranking score:   {result['score']:.4f}"
        )

        print(
            f"Semantic score:  {result['semantic_score']:.4f}"
        )

        print(
            f"Source boost:    {result['source_boost']:.4f}"
        )

        print(
            f"Section boost:   {result['section_boost']:.4f}"
        )

        print(
            f"Chunk:           {result['chunk_id']}"
        )

        print(
            f"Source:          {result['source']}"
        )

        print(
            f"Section:         {result['section']}"
        )

        print(f"\n{result['text']}")
