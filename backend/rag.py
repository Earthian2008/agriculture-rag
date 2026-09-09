import ollama

from backend.retriever import retrieve


LLM_MODEL = "lfm2.5-thinking:1.2b"


def build_context(results):
    """
    Convert retrieved chunks into a clean context block
    for the language model.
    """

    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"[Knowledge {i}]\n"
            f"Source: {result['source']}\n"
            f"Section: {result['section']}\n"
            f"Content:\n{result['text']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(query, results):
    """
    Generate an answer using only retrieved agriculture knowledge.
    """

    if not results:
        return (
            "I could not find enough relevant agriculture knowledge "
            "to answer this question."
        )

    context = build_context(results)

    prompt = f"""
You are an agriculture assistant specializing in rice cultivation.

Your job is to answer the user's question using ONLY the agriculture
knowledge provided below.

Rules:

1. Do not invent agricultural recommendations.
2. Do not add facts that are not supported by the provided knowledge.
3. Use the retrieved knowledge as the source of truth.
4. If the knowledge is insufficient, clearly say so.
5. Give a concise, practical answer.
6. Preserve important quantities, timings, stages, and units exactly
   when they are present in the knowledge.
7. Do not mention internal ranking scores, chunk IDs, embeddings,
   FAISS, or other implementation details.

Agriculture knowledge:

{context}

User question:

{query}

Answer:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]


if __name__ == "__main__":

    query = input("\nAsk an agriculture question: ")

    print("\nRetrieving relevant knowledge...")

    results = retrieve(query)

    print("\nGenerating answer...")

    answer = generate_answer(query, results)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES USED")
    print("=" * 60)

    for result in results:
        print(
            f"- {result['source']} | "
            f"{result['section']}"
        )
