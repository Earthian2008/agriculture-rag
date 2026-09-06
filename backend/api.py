from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag import retrieve, generate_answer


app = FastAPI(title="Agriculture AI API")


# Allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Agriculture AI API is running"
    }


@app.post("/ask")
def ask_question(request: Question):

    results = retrieve(request.question)

    answer = generate_answer(
        request.question,
        results
    )

    sources = [
        {
            "chunk_id": result["chunk_id"],
            "source": result["source"],
            "section": result["section"],
            "score": result["score"]
        }
        for result in results
    ]

    return {
        "question": request.question,
        "answer": answer,
        "sources": sources
    }
