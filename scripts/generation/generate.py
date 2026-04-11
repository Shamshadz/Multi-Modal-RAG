"""
generate.py
===========
Answer generation layer: wraps either an OpenAI chat model or a local
HuggingFace text-generation pipeline behind a common ``Generator`` interface.

The module also exposes the CLI entry-point for running the full RAG pipeline
from the command line.

Switching backends
------------------
Set ``--model openai`` (default) or ``--model local``.

* **openai**: Calls ``gpt-4o-mini`` via the official ``openai`` Python package.
  Requires the ``OPENAI_API_KEY`` environment variable.
* **local**: Runs any HuggingFace text-generation model (default ``gpt2``).
  Useful for offline / air-gapped environments.  Quality is model-dependent.

Prompt template
---------------
The ``build_prompt`` function formats the retrieved context and user query into
a simple but effective "answer only from context" instruction prompt.

CLI usage
---------
.. code-block:: bash

    python -m scripts.generation.generate \\
        --query "What is CLIP?" \\
        --mode hybrid \\
        --model openai
"""

from __future__ import annotations

import argparse
import logging
import os

import torch

from core.embeddings import EmbeddingModel
from core.retriever import Retriever
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _get_device() -> str:
    """Return ``'cuda'`` when a GPU is available, otherwise ``'cpu'``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """
    LLM wrapper that supports OpenAI and local HuggingFace pipelines.

    For the ``"local"`` backend the model is placed on GPU automatically when
    one is available; it falls back to CPU otherwise.  No manual configuration
    is required — the decision is made at runtime via ``torch.cuda.is_available()``.

    Parameters
    ----------
    model_type : str
        ``"openai"`` or ``"local"``.
    local_model_name : str
        HuggingFace model ID used when ``model_type == "local"``.
    """

    def __init__(
        self,
        model_type: str = "openai",
        local_model_name: str = "gpt2",
    ):
        self.model_type = model_type

        if model_type == "openai":
            from openai import OpenAI  # type: ignore

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is not set."
                )
            self.client = OpenAI(api_key=api_key)
            self.openai_model = "gpt-4o-mini"

        elif model_type == "local":
            from transformers import pipeline  # type: ignore

            device = _get_device()
            logger.info(
                "Loading local model '%s' on device='%s' …",
                local_model_name,
                device,
            )

            # Pass the integer device index for CUDA (0 = first GPU),
            # or -1 to force CPU.  This is more explicit than device_map="auto"
            # and works correctly on single-GPU machines.
            device_arg = 0 if device == "cuda" else -1

            self.pipe = pipeline(
                "text-generation",
                model=local_model_name,
                device=device_arg,
            )
            logger.info(
                "Local model loaded. Running on %s.",
                "GPU (CUDA)" if device == "cuda" else "CPU",
            )

        else:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Use 'openai' or 'local'."
            )

    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate an answer for the given *prompt*.

        Parameters
        ----------
        prompt : str
            Full prompt text (context + question).
        max_tokens : int
            Maximum tokens in the generated answer.

        Returns
        -------
        str
            The model's answer text.
        """
        if self.model_type == "openai":
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            "Answer the question using ONLY the provided context. "
                            "If the context does not contain enough information, "
                            "say 'I don't have enough context to answer that.'"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        else:  # local
            output = self.pipe(
                prompt,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                do_sample=False,
            )
            # Strip the original prompt from the output
            generated = output[0]["generated_text"]
            return generated[len(prompt):].strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(query: str, context: str) -> str:
    """
    Format a retrieval-augmented prompt.

    Parameters
    ----------
    query : str
        The user's question.
    context : str
        Concatenated retrieved chunks (output of ``Retriever.build_context``).

    Returns
    -------
    str
        A formatted prompt string.

    Example
    -------
    >>> print(build_prompt("What is FAISS?", "FAISS is a library for ..."))
    Context:
    FAISS is a library for ...

    Question:
    What is FAISS?

    Answer:
    """
    return (
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full RAG pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal RAG – generate an answer for a query."
    )
    parser.add_argument("--query", type=str, required=True, help="User query text.")
    parser.add_argument(
        "--mode",
        type=str,
        default="text",
        choices=["text", "hybrid", "text_to_image"],
        help="Retrieval mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai",
        choices=["openai", "local"],
        help="Generator backend.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Results per retrieval call.")
    args = parser.parse_args()

    # --- Load stores ---
    text_store = VectorStore.load("stores/text_store")
    image_store = VectorStore.load("stores/image_store")

    retriever = Retriever(
        text_store=text_store,
        image_store=image_store,
        embedder=EmbeddingModel(),
    )
    generator = Generator(model_type=args.model)

    # --- Retrieve ---
    if args.mode == "text":
        results = retriever.retrieve_text(args.query, top_k=args.top_k)
    elif args.mode == "hybrid":
        results = retriever.retrieve_hybrid(args.query, top_k=args.top_k)
    elif args.mode == "text_to_image":
        results = retriever.retrieve_text_to_image(args.query, top_k=args.top_k)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # --- Build context & generate ---
    context = retriever.build_context(results)
    prompt = build_prompt(args.query, context)
    answer = generator.generate(prompt)

    print("\n=== CONTEXT (preview) ===\n")
    print(context[:800])
    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()