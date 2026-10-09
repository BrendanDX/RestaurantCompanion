"""Lab 4: build the multimodal vector indexes (restaurant text + recipe images) in ChromaDB.

Prereqs: data/processed/structured_restaurant_data.json (Lab 1),
         data/processed/augmented_food_recipe.json (Lab 2),
         recipe images extracted under data/raw/recipes/ (Lab 2 / download_data.py).

Run:  python -m connoisseur.rag.index_builder
"""
from __future__ import annotations

import glob
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from connoisseur import config


@dataclass
class Doc:
    page_content: str
    metadata: dict = field(default_factory=dict)


def _clean_metadata(meta: dict) -> dict:
    """Chroma rejects None metadata values, so drop them."""
    return {k: v for k, v in meta.items() if v is not None}


def find_image_paths(image_dir: Path | None = None) -> list[str]:
    image_dir = Path(image_dir or config.RECIPES_DIR)
    return sorted(glob.glob(f"{image_dir}/**/*.png", recursive=True))


def build_article_docs(restaurants: list[dict]) -> list[Doc]:
    docs = []
    for i, r in enumerate(restaurants):
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        text = (
            f"Restaurant: {name}\n"
            f"Cuisine: {r.get('food_style', '')}\n"
            f"Location: {r.get('location', '')}"
        )
        docs.append(Doc(
            page_content=text.strip(),
            metadata=_clean_metadata({
                "doc_id": f"rest_{i}",
                "cuisine": r.get("food_style"),
                "location": r.get("location"),
                "source": "restaurant",
            }),
        ))
    return docs


def build_image_docs(image_paths: list[str], recipes: list[dict]) -> list[Doc]:
    docs = []
    for i, (path, rec) in enumerate(zip(image_paths, recipes)):
        docs.append(Doc(
            page_content=rec.get("name", f"recipe image {i}"),
            metadata=_clean_metadata({
                "doc_id": f"img_{i}",
                "image_path": path,
                "source": "recipe_image",
                "recipe_id": rec.get("id"),
                "cuisine": rec.get("cuisine"),
            }),
        ))
    return docs


def _upsert(client, name: str, docs: list[Doc], vectors) -> None:
    # cosine space => distance = 1 - cosine similarity, which is what `1 - distance` assumes later
    collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=[d.metadata["doc_id"] for d in docs],
        embeddings=vectors.tolist(),
        documents=[d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
    )


def build_indexes(db_dir: Path | None = None) -> None:
    import chromadb

    from connoisseur.rag.embeddings import embed_images, embed_texts

    db_dir = Path(db_dir or config.CHROMA_DIR)
    restaurants = json.loads(config.STRUCTURED_RESTAURANTS_PATH.read_text(encoding="utf-8"))
    recipes = json.loads(config.AUGMENTED_RECIPES_PATH.read_text(encoding="utf-8"))
    print(f"Loaded restaurants: {len(restaurants)} | recipes: {len(recipes)}")

    article_docs = build_article_docs(restaurants)
    image_docs = build_image_docs(find_image_paths(), recipes)
    print(f"article docs: {len(article_docs)} | image docs: {len(image_docs)}")
    if not article_docs or not image_docs:
        raise RuntimeError("Nothing to index. Run Labs 1-2 first and make sure recipe images are extracted.")

    if db_dir.is_dir():
        shutil.rmtree(db_dir)  # reset so reruns are clean
    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))

    _upsert(client, config.ARTICLE_COLLECTION, article_docs, embed_texts([d.page_content for d in article_docs]))
    print("Article DB ready")
    _upsert(client, config.IMAGE_COLLECTION, image_docs, embed_images([d.metadata["image_path"] for d in image_docs]))
    print("Image DB ready")
    print("Multimodal Vector Index Construction COMPLETE")


if __name__ == "__main__":
    build_indexes()
