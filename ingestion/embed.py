from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/chunks/rice_chunks.jsonl")
MODEL_PATH = Path("models/bge-small-en-v1.5")
OUTPUT_DIR = Path("data/embeddings")


print("Loading BGE model...")

model = SentenceTransformer(str(MODEL_PATH))

print("BGE model loaded successfully.")


print("Loading chunks...")

chunks = []

with CHUNKS_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        chunks.append(json.loads(line))

texts = [chunk["text"] for chunk in chunks]

print(f"Loaded {len(chunks)} chunks.")
print(f"Texts ready for embedding: {len(texts)}")


print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

embeddings = np.asarray(embeddings, dtype="float32")

print("Embeddings generated successfully.")
print(f"Embedding shape: {embeddings.shape}")


print("Saving embeddings and metadata...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(
    OUTPUT_DIR / "rice_embeddings.npy",
    embeddings,
)

with (OUTPUT_DIR / "rice_metadata.json").open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        chunks,
        file,
        ensure_ascii=False,
        indent=2,
    )


print("Saved successfully.")
print(f"Embeddings: {OUTPUT_DIR / 'rice_embeddings.npy'}")
print(f"Metadata:   {OUTPUT_DIR / 'rice_metadata.json'}")
