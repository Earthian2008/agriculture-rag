from sentence_transformers import SentenceTransformer
import faiss
import ollama

# -----------------------------
# 1. Load embedding model
# -----------------------------

MODEL_PATH = "./models/bge-small-en-v1.5"

embedding_model = SentenceTransformer(MODEL_PATH)


# -----------------------------
# 2. Our knowledge base
# -----------------------------

texts = [
    "Soil moisture is an important parameter in precision agriculture. "
    "A soil moisture sensor measures the amount of water present in soil.",

    "When soil moisture is too low, plants may experience water stress. "
    "When soil moisture is sufficient, irrigation may not be necessary.",

    "Automated irrigation systems can use soil moisture sensor readings "
    "to decide when irrigation should be activated.",

    "Different crops require different soil moisture levels. "
    "Irrigation decisions should consider crop type, soil type, "
    "weather conditions, and current soil moisture.",

    "Excessive irrigation can waste water and may cause waterlogging "
    "and nutrient leaching."
]


# -----------------------------
# 3. Create embeddings
# -----------------------------

embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True
)

print("Embeddings created:", embeddings.shape)


# -----------------------------
# 4. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings.astype("float32"))

print("FAISS index contains:", index.ntotal, "documents")


# -----------------------------
# 5. Ask a question
# -----------------------------

question = input("\nAsk something: ")


# -----------------------------
# 6. Embed the question
# -----------------------------

question_embedding = embedding_model.encode(
    [question],
    normalize_embeddings=True
)

question_embedding = question_embedding.astype("float32")


# -----------------------------
# 7. Search FAISS
# -----------------------------

k = 3

scores, indices = index.search(
    question_embedding,
    k
)

print("\nRetrieved context:")

retrieved_texts = []

for i, score in zip(indices[0], scores[0]):

    if i != -1:
        print(f"\nScore: {score:.4f}")
        print(texts[i])

        retrieved_texts.append(texts[i])


# -----------------------------
# 8. Build prompt
# -----------------------------

context = "\n\n".join(retrieved_texts)

prompt = f"""
You are an agriculture assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided knowledge base."

Context:
{context}

User question:
{question}

Answer:
"""


# -----------------------------
# 9. Send context to LFM
# -----------------------------

response = ollama.chat(
    model="lfm2.5-thinking:1.2b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# -----------------------------
# 10. Display answer
# -----------------------------

print("\n==============================")
print("LFM2.5 ANSWER")
print("==============================")

print(response["message"]["content"])
