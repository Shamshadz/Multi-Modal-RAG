import torch
from PIL import Image
from typing import List, Union

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel


class EmbeddingModel:
    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Text model
        self.text_model = SentenceTransformer(text_model_name, device=self.device)

        # CLIP model for image + text
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # -----------------------------
    # TEXT EMBEDDINGS
    # -----------------------------
    def embed_text(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for text input(s)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.text_model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True
        )

        return embeddings.cpu().tolist()

    # -----------------------------
    # IMAGE EMBEDDINGS (CLIP)
    # -----------------------------
    def embed_image(self, images: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> List[List[float]]:
        """
        Generate embeddings for image(s)
        Accepts file paths or PIL Images
        """
        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            processed_images.append(img)

        inputs = self.clip_processor(
            images=processed_images,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        return image_features.cpu().tolist()

    # -----------------------------
    # CLIP TEXT EMBEDDINGS (for cross-modal search)
    # -----------------------------
    def embed_text_clip(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate CLIP embeddings for text (aligned with image embeddings)
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        return text_features.cpu().tolist()