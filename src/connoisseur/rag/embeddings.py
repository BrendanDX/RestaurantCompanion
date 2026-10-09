"""Embedding models (lazy-loaded so importing this package never pulls in torch)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from connoisseur import config


@lru_cache(maxsize=1)
def get_text_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.TEXT_EMBED_MODEL)


@lru_cache(maxsize=1)
def get_clip():
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to("cpu")
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME, use_fast=True)
    model.eval()
    return model, processor


def _as_tensor(features):
    """Newer transformers versions may return an output object instead of a bare tensor."""
    import torch

    if torch.is_tensor(features):
        return features
    return features.pooler_output


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """384-d, L2-normalised (cosine-ready)."""
    return get_text_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)


def embed_images(paths: list[str | Path], batch_size: int = 16) -> np.ndarray:
    """512-d CLIP image embeddings, L2-normalised."""
    import torch
    from PIL import Image

    model, processor = get_clip()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            inputs = processor(images=imgs, return_tensors="pt")
            feats = _as_tensor(model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)


def embed_query_clip_text(query: str) -> np.ndarray:
    """512-d CLIP *text* embedding, so a text query can search the image collection."""
    import torch

    model, processor = get_clip()
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        feats = _as_tensor(model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype(np.float32)
