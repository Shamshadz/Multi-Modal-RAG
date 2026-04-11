"""
main.py
=======
FastAPI application exposing the Multi-Modal RAG pipeline over HTTP.

Endpoints
---------
GET  /
    Health check.

POST /query
    Text / hybrid / cross-modal retrieval + generation.
    Body: ``{"query": str, "mode": str, "top_k": int}``

POST /image-query
    Upload an image file; retrieve visually similar documents and generate an answer.

Running locally
---------------
.. code-block:: bash

    uvicorn main:app --reload --port 8000

Interactive docs are available at ``http://localhost:8000/docs``.

Environment variables
---------------------
``OPENAI_API_KEY``  – required when using the OpenAI generator backend.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core.embeddings import EmbeddingModel
from core.retriever import Retriever
from core.vector_store import VectorStore
from scripts.generation.generate import Generator, build_prompt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Modal RAG API",
    description=(
        "Retrieval-Augmented Generation over text and image corpora. "
        "Supports text-to-text, text-to-image, and hybrid retrieval modes."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Startup – load models once
# ---------------------------------------------------------------------------

logger.info("Loading stores and models …")

text_store = VectorStore.load("stores/text_store")
image_store = VectorStore.load("stores/image_store")
embedder = EmbeddingModel()

retriever = Retriever(
    text_store=text_store,
    image_store=image_store,
    embedder=embedder,
    fusion_strategy="reciprocal_rank",
)

generator = Generator(model_type=os.getenv("GENERATOR_BACKEND", "openai"))

logger.info("System ready. text_store=%d vectors, image_store=%d vectors.",
            len(text_store), len(image_store))


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural-language question.")
    mode: str = Field(
        "text",
        description="Retrieval mode: 'text' | 'hybrid' | 'text_to_image'.",
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")
    alpha: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Weight of text results in hybrid mode (0=image only, 1=text only).",
    )


class QueryResponse(BaseModel):
    query: str
    mode: str
    answer: str
    sources: list
    context_preview: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check."""
    return {
        "status": "running",
        "text_vectors": len(text_store),
        "image_vectors": len(image_store),
    }


@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
def query_endpoint(request: QueryRequest):
    """
    Retrieve relevant context and generate an answer.

    Modes
    -----
    * **text** – SentenceTransformer text retrieval.
    * **hybrid** – text + CLIP cross-modal, fused via RRF.
    * **text_to_image** – CLIP text→image cross-modal retrieval only.
    """
    try:
        if request.mode == "text":
            results = retriever.retrieve_text(request.query, top_k=request.top_k)
        elif request.mode == "hybrid":
            results = retriever.retrieve_hybrid(
                request.query, top_k=request.top_k, alpha=request.alpha
            )
        elif request.mode == "text_to_image":
            results = retriever.retrieve_text_to_image(request.query, top_k=request.top_k)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {request.mode}")

        context = retriever.build_context(results)
        prompt = build_prompt(request.query, context)
        answer = generator.generate(prompt)

        return QueryResponse(
            query=request.query,
            mode=request.mode,
            answer=answer,
            sources=results,
            context_preview=context[:500],
        )

    except Exception as exc:
        logger.exception("Error processing query.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/image-query", tags=["Retrieval"])
def image_query_endpoint(file: UploadFile = File(...)):
    """
    Upload an image and retrieve visually similar documents.

    Accepts PNG, JPEG, or WEBP files.
    """
    allowed_content_types = {"image/png", "image/jpeg", "image/webp"}
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use PNG, JPEG, or WEBP.",
        )

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

    try:
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        results = retriever.retrieve_image(file_path, top_k=5)
        context = retriever.build_context(results)
        prompt = build_prompt("Describe this image context", context)
        answer = generator.generate(prompt)

        return {"answer": answer, "sources": results}

    except Exception as exc:
        logger.exception("Error processing image query.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)