from dataclasses import dataclass
from typing import Annotated

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from app import llm
from app.chunking import chunk_text


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    vector: list[float]


class DocumentIn(BaseModel):
    id: str
    text: str

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("id must not be empty")
        return v

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace only")
        return v


class IngestRequest(BaseModel):
    documents: Annotated[list[DocumentIn], Field(min_length=1)]


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty or whitespace only")
        return v


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def create_app() -> FastAPI:
    """Return a NEW app with an EMPTY in-memory store."""
    app = FastAPI()
    # doc_id -> list of chunks (store lives in this closure, not at module level)
    store: dict[str, list[Chunk]] = {}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        # If the same id appears twice in one request, the last one wins.
        new_docs: dict[str, list[Chunk]] = {}
        for doc in req.documents:
            pieces = chunk_text(doc.text)
            new_docs[doc.id] = [
                Chunk(
                    id=f"{doc.id}:{i}",
                    doc_id=doc.id,
                    text=piece,
                    vector=llm.embed(piece),
                )
                for i, piece in enumerate(pieces)
            ]
        # Replace (not duplicate) existing documents.
        store.update(new_docs)
        return {
            "ingested": len(req.documents),
            "chunks": sum(len(c) for c in new_docs.values()),
        }

    @app.post("/ask")
    def ask(req: AskRequest):
        all_chunks = [c for chunks in store.values() for c in chunks]
        if not all_chunks:
            raise HTTPException(status_code=404, detail="No documents ingested")

        q_vec = llm.embed(req.question)
        scored = [(_dot(q_vec, c.vector), c) for c in all_chunks]
        # Highest score first; ties broken deterministically by chunk id.
        scored.sort(key=lambda sc: (-sc[0], sc[1].id))
        top = scored[: req.top_k]

        answer = llm.generate(req.question, [c.text for _, c in top])
        return {
            "answer": answer,
            "sources": [
                {
                    "doc_id": c.doc_id,
                    "chunk_id": c.id,
                    "score": score,
                    "text": c.text,
                }
                for score, c in top
            ],
        }

    @app.get("/documents")
    def list_documents():
        return {
            "documents": [
                {"id": doc_id, "chunks": len(store[doc_id])}
                for doc_id in sorted(store)
            ]
        }

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str):
        if doc_id not in store:
            raise HTTPException(status_code=404, detail="Document not found")
        del store[doc_id]
        return Response(status_code=204)

    return app


app = create_app()
