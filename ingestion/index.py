from pathlib import Path

import faiss
import numpy as np


EMBEDDINGS_FILE = Path("data/embeddings/rice_embeddings.npy")
OUTPUT_DIR = Path("data/vector_db")


print("Loading embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

print(f"Loaded embeddings: {embeddings.shape}")


print("Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS index contains {index.ntotal} vectors.")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

index_file = OUTPUT_DIR / "rice.index"

faiss.write_index(index, str(index_file))

print("FAISS index saved successfully.")
print(f"Index: {index_file}")
