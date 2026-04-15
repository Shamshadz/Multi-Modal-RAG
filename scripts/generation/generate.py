"""
generate.py
===========
Answer generation layer using **free, local HuggingFace models only**.
No API keys or paid services required.

Supported model backends
-------------------------
``flan-t5``  (default)
    ``google/flan-t5-base`` — a seq2seq (encoder-decoder) model fine-tuned
    for instruction following and Q&A.  Runs comfortably on CPU (~1 GB RAM).
    Upgrade to ``flan-t5-large`` or ``flan-t5-xl`` for better quality if you
    have more RAM / a GPU.

``phi2``
    ``microsoft/phi-2`` — a 2.7B decoder-only model with strong reasoning.
    Needs ~6 GB RAM.  Significantly better answer quality than flan-t5-base.

``gpt2``
    ``gpt2`` — 117M parameter decoder-only model.  Very lightweight, but
    answer quality is limited.  Useful for smoke-testing on minimal hardware.

Model selection
---------------
All models are downloaded automatically from HuggingFace Hub on first use
and cached locally (~/.cache/huggingface/).  Subsequent runs are fully
offline — no internet connection needed after the first download.

GPU / CPU
---------
Every backend auto-detects GPU via torch.cuda.is_available() and runs
on GPU when one is present.  Falls back to CPU automatically.

CLI usage
---------
.. code-block:: bash

    # Default: flan-t5-base, text retrieval
    python -m scripts.generation.generate --query "What is CLIP?"

    # Better quality: phi-2
    python -m scripts.generation.generate --query "What is CLIP?" --backend phi2

    # Hybrid retrieval
    python -m scripts.generation.generate --query "cat on a sofa" --mode hybrid

    # List available backends
    python -m scripts.generation.generate --list_backends
"""

from __future__ import annotations

import argparse
import logging

import torch

from core.embeddings import EmbeddingModel
from core.retriever import Retriever
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Free model registry
# ---------------------------------------------------------------------------

#: Map of short backend names to HuggingFace model IDs and pipeline task.
#: Add any HuggingFace model here without touching the Generator class.
FREE_MODELS: dict = {
    "flan-t5": {
        "model_id": "google/flan-t5-base",
        "task": "text-generation",   # seq2seq – best for Q&A
        "description": "Flan-T5 base (~250 MB, CPU-friendly, good Q&A)",
    },
    "phi2": {
        "model_id": "microsoft/phi-2",
        "task": "text-generation",
        "description": "Phi-2 2.7B (~6 GB RAM, strong reasoning)",
    },
    "gpt2": {
        "model_id": "gpt2",
        "task": "text-generation",
        "description": "GPT-2 117M (~500 MB, minimal hardware)",
    },
}

DEFAULT_BACKEND = "flan-t5"


def _get_device() -> str:
    """Return 'cuda' when a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def list_backends() -> None:
    """Print available free model backends to stdout."""
    print("\nAvailable free model backends:\n")
    for name, info in FREE_MODELS.items():
        marker = " (default)" if name == DEFAULT_BACKEND else ""
        print(f"  {name:<10} -- {info['description']}{marker}")
    print()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """
    Free, local LLM wrapper backed by HuggingFace Transformers.

    No API key is required.  Models are downloaded once and cached locally.

    Parameters
    ----------
    backend : str
        Short name of the model to use.  Must be a key in FREE_MODELS.
        Defaults to 'flan-t5' (google/flan-t5-base).
    custom_model_id : str or None
        Override with any arbitrary HuggingFace model ID
        (e.g. 'google/flan-t5-large').  When set, backend is ignored.
    custom_task : str
        Pipeline task for custom_model_id.
        'text2text-generation' for seq2seq models (T5 family).
        'text-generation' for decoder-only models (GPT family).

    Examples
    --------
    >>> gen = Generator()                           # flan-t5-base (default)
    >>> gen = Generator(backend="phi2")             # phi-2
    >>> gen = Generator(backend="gpt2")             # GPT-2
    >>> gen = Generator(                            # any custom HF model
    ...     custom_model_id="google/flan-t5-large",
    ...     custom_task="text2text-generation",
    ... )
    """

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        custom_model_id: str = None,
        custom_task: str = "text2text-generation",
    ):
        from transformers import pipeline  # type: ignore

        device = _get_device()
        # CUDA device index (0 = first GPU) or -1 for CPU
        device_arg = 0 if device == "cuda" else -1
        device_label = "GPU (CUDA)" if device == "cuda" else "CPU"

        if custom_model_id:
            model_id = custom_model_id
            task = custom_task
            logger.info("Using custom model '%s' (task=%s) on %s.", model_id, task, device_label)
        else:
            if backend not in FREE_MODELS:
                raise ValueError(
                    f"Unknown backend '{backend}'. "
                    f"Choose from: {list(FREE_MODELS.keys())} "
                    f"or pass custom_model_id."
                )
            cfg = FREE_MODELS[backend]
            model_id = cfg["model_id"]
            task = cfg["task"]
            logger.info(
                "Loading backend='%s' (%s) on %s ...", backend, model_id, device_label
            )

        self._task = task

        # torch_dtype=float16 cuts GPU memory usage by ~half on supported models.
        # On CPU float32 is required (float16 ops are not accelerated on CPU).
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.pipe = pipeline(
            task,
            model=model_id,
            device=device_arg,
            torch_dtype=dtype,
        )

        logger.info("Generator ready. model=%s, device=%s.", model_id, device_label)

    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Generate an answer for the given prompt.

        Parameters
        ----------
        prompt : str
            Full prompt text produced by build_prompt().
        max_new_tokens : int
            Maximum number of new tokens to generate.
            For seq2seq models (flan-t5) this is the output length.
            For decoder-only models (gpt2, phi2) this is tokens added to prompt.

        Returns
        -------
        str
            The generated answer string, with the input prompt stripped out.
        """
        if "t5" in self.pipe.model.name_or_path.lower():
            # seq2seq (Flan-T5): the output contains only the generated answer,
            # never the prompt — no stripping needed.
            output = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy decoding = deterministic
            )
            generated = output[0]["generated_text"]

            # Remove prompt if echoed
            if generated.startswith(prompt):
                generated = generated[len(prompt):]

            return generated.strip()

        else:
            # Decoder-only (GPT-2, Phi-2): the output INCLUDES the prompt.
            # We strip it so only the newly generated text is returned.
            output = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.pipe.tokenizer.eos_token_id,
            )
            generated = output[0]["generated_text"]
            answer = generated[len(prompt):].strip()
            # Fallback: return everything if stripping leaves an empty string
            return answer if answer else generated.strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(query: str, context: str) -> str:
    return f"""
Context:
{context}

Question: {query}
Answer:
""".strip()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full RAG pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal RAG -- free local model generation."
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
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        choices=list(FREE_MODELS.keys()),
        help="Free model backend (default: flan-t5).",
    )
    parser.add_argument(
        "--custom_model",
        type=str,
        default="",
        help="Any HuggingFace model ID to override --backend (e.g. google/flan-t5-large).",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Chunks to retrieve.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="Max tokens to generate."
    )
    parser.add_argument(
        "--list_backends",
        action="store_true",
        help="Print available free backends and exit.",
    )
    args = parser.parse_args()

    if args.list_backends:
        list_backends()
        return

    # --- Load stores ---
    text_store = VectorStore.load("stores/text_store")
    image_store = VectorStore.load("stores/image_store")

    retriever = Retriever(
        text_store=text_store,
        image_store=image_store,
        embedder=EmbeddingModel(),
    )

    generator = Generator(
        backend=args.backend,
        custom_model_id=args.custom_model or None,
    )

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
    answer = generator.generate(prompt, max_new_tokens=args.max_new_tokens)

    print("\n=== CONTEXT (preview) ===\n")
    print(context[:800])
    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()