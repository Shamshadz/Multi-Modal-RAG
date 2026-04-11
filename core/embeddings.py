"""
embeddings.py
=============
Provides a unified interface for generating dense vector embeddings from both
text and images.

Two underlying models are used:

1. **SentenceTransformer** (``all-MiniLM-L6-v2``, 384-d)
   Fast, lightweight text encoder. Used for the main text retrieval path.

2. **CLIP** (``openai/clip-vit-base-patch32``, 512-d)
   Dual-encoder trained on (image, caption) pairs. Its image encoder maps
   images into the same embedding space as its text encoder, enabling
   cross-modal retrieval (text query → find similar images, image query →
   find similar text).

Embedding spaces
----------------
* ``embed_text``      → SentenceTransformer space (384-d), L2-normalised.
* ``embed_image``     → CLIP image space (512-d), L2-normalised.
* ``embed_text_clip`` → CLIP text space (512-d), L2-normalised.

Only compare vectors produced by the *same* encoder; mixing spaces produces
meaningless cosine similarity scores.

Usage
-----
>>> em = EmbeddingModel()
>>> vecs = em.embed_text(["Hello world", "Goodbye moon"])
>>> len(vecs), len(vecs[0])
(2, 384)
"""

from __future__ import annotations

import logging
from typing import List, Union

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Unified embedding model for text and images.

    Parameters
    ----------
    text_model_name : str
        HuggingFace model ID for the SentenceTransformer text encoder.
    clip_model_name : str
        HuggingFace model ID for the CLIP model.
    device : str or None
        ``"cuda"`` or ``"cpu"``.  Auto-detected when *None*.
    """

    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("EmbeddingModel initialising on device=%s", self.device)

        # --- Text encoder (SentenceTransformer) ---
        self.text_model = SentenceTransformer(text_model_name, device=self.device)

        # --- CLIP (image + cross-modal text) ---
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        logger.info("EmbeddingModel ready.")

    # ------------------------------------------------------------------
    # Text → SentenceTransformer space (384-d)
    # ------------------------------------------------------------------

    def embed_text(
        self,
        texts: Union[str, List[str]],
    ) -> List[List[float]]:
        """
        Encode one or more strings with the SentenceTransformer model.

        Parameters
        ----------
        texts : str or list[str]
            Input text(s) to embed.

        Returns
        -------
        list[list[float]]
            One L2-normalised 384-d vector per input string.

        Notes
        -----
        Batch the call when encoding many chunks at once – encoding 100 chunks
        in one call is ~10× faster than 100 single-item calls.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.text_model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.cpu().tolist()

    # ------------------------------------------------------------------
    # Image → CLIP image space (512-d)
    # ------------------------------------------------------------------

    def embed_image(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
    ) -> List[List[float]]:
        """
        Encode one or more images with the CLIP image encoder.

        Parameters
        ----------
        images : str | PIL.Image | list thereof
            File paths or already-loaded PIL images.

        Returns
        -------
        list[list[float]]
            One L2-normalised 512-d vector per image.
        """
        if not isinstance(images, list):
            images = [images]

        pil_images: List[Image.Image] = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))

        inputs = self.clip_processor(
            images=pil_images,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features.cpu().tolist()

    # ------------------------------------------------------------------
    # Text → CLIP text space (512-d, aligned with image embeddings)
    # ------------------------------------------------------------------

    def embed_text_clip(
        self,
        texts: Union[str, List[str]],
    ) -> List[List[float]]:
        """
        Encode text with the CLIP *text* encoder.

        The resulting vectors live in the same 512-d embedding space as
        :meth:`embed_image`, enabling **cross-modal retrieval**: you can
        embed a text query and search it against image embeddings.

        Parameters
        ----------
        texts : str or list[str]
            Query text(s).

        Returns
        -------
        list[list[float]]
            One L2-normalised 512-d vector per input string.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features.cpu().tolist()