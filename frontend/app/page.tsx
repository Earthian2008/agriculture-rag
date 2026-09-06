"use client";

import { FormEvent, useState } from "react";

type Source = {
  chunk_id: string;
  source: string;
  section: string;
  score: number;
};

type ApiResponse = {
  question: string;
  answer: string;
  sources: Source[];
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function askQuestion(event: FormEvent) {
    event.preventDefault();

    if (!question.trim() || loading) return;

    setLoading(true);
    setResponse(null);

    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question.trim(),
        }),
      });

      if (!res.ok) {
        throw new Error("API request failed");
      }

      const data: ApiResponse = await res.json();
      setResponse(data);
    } catch (error) {
      console.error(error);
      alert("Could not connect to Agriculture AI backend.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#07110b] text-white">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-6 py-8">

        {/* Header */}
        <header className="flex items-center justify-between border-b border-white/10 pb-6">
          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-green-500/15 text-xl">
                🌾
              </div>

              <div>
                <h1 className="text-lg font-semibold">
                  Agriculture AI
                </h1>

                <p className="text-xs text-white/40">
                  Knowledge-backed farming assistant
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-green-400/20 bg-green-400/5 px-3 py-1.5 text-xs text-green-300">
            <span className="h-2 w-2 rounded-full bg-green-400" />
            RAG ONLINE
          </div>
        </header>

        {/* Hero */}
        <section className="flex flex-1 flex-col items-center justify-center py-16">

          <div className="mb-4 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs text-white/50">
            🌱 Rice Cultivation • Tamil Nadu
          </div>

          <h2 className="max-w-3xl text-center text-4xl font-bold tracking-tight sm:text-6xl">
            Ask your
            <span className="text-green-400"> agriculture </span>
            questions.
          </h2>

          <p className="mt-5 max-w-2xl text-center text-base leading-7 text-white/45">
            Get answers grounded in agricultural knowledge from your
            retrieval system instead of relying only on an LLM.
          </p>

          {/* Question form */}
          <form
            onSubmit={askQuestion}
            className="mt-10 w-full max-w-3xl"
          >
            <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/[0.04] p-2 shadow-2xl">

              <input
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask about rice cultivation..."
                className="min-w-0 flex-1 bg-transparent px-4 py-3 text-sm outline-none placeholder:text-white/25"
              />

              <button
                type="submit"
                disabled={loading || !question.trim()}
                className="rounded-xl bg-green-500 px-5 py-3 text-sm font-semibold text-black transition hover:bg-green-400 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? "Thinking..." : "Ask"}
              </button>

            </div>
          </form>

          {/* Answer */}
          {response && (
            <section className="mt-8 w-full max-w-3xl rounded-2xl border border-white/10 bg-white/[0.04] p-6">

              <div className="mb-4 flex items-center gap-2">
                <span className="text-xl">🌾</span>

                <h3 className="font-semibold">
                  Agriculture AI
                </h3>
              </div>

              <p className="whitespace-pre-wrap text-sm leading-7 text-white/80">
                {response.answer}
              </p>

              {/* Sources */}
              <div className="mt-6 border-t border-white/10 pt-5">

                <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-white/35">
                  Sources used
                </p>

                <div className="space-y-2">
                  {response.sources.map((source) => (
                    <div
                      key={source.chunk_id}
                      className="rounded-lg border border-white/5 bg-black/20 px-3 py-2 text-xs"
                    >
                      <div className="text-white/70">
                        {source.section}
                      </div>

                      <div className="mt-1 text-white/30">
                        {source.source} • {source.chunk_id}
                      </div>
                    </div>
                  ))}
                </div>

              </div>
            </section>
          )}

        </section>

        {/* Footer */}
        <footer className="border-t border-white/10 pt-5 text-center text-xs text-white/20">
          Agriculture AI • Retrieval-Augmented Generation
        </footer>

      </div>
    </main>
  );
}
