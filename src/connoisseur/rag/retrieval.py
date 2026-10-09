"""Lab 5: similarity retrieval (with metadata filtering) over the Chroma collections."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from connoisseur import config
from connoisseur.rag import embeddings


@lru_cache(maxsize=None)
def _client(db_dir: str):
    import chromadb

    return chromadb.PersistentClient(path=db_dir)


def get_collection(name: str):
    db_dir = Path(config.CHROMA_DIR)
    if not db_dir.is_dir():
        raise RuntimeError(
            f"Vector database directory not found: '{db_dir}'. "
            "Run `python -m connoisseur.rag.index_builder` first."
        )
    collection = _client(str(db_dir)).get_collection(name)
    if collection.count() <= 0:
        raise RuntimeError(f"Collection '{name}' is empty. Rebuild the index.")
    return collection


def _unwrap(res: dict):
    """Chroma returns lists-of-lists; unwrap the first (only) query."""
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists


def _query(collection_name: str, vector, k: int, where: dict | None):
    res = get_collection(collection_name).query(
        query_embeddings=[vector.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)


def retrieve_articles(query: str, k: int = 5, where: dict | None = None):
    """text -> restaurant articles (MiniLM, 384-d). Returns ids, docs, metas, distances."""
    return _query(config.ARTICLE_COLLECTION, embeddings.embed_texts([query])[0], k, where)


def retrieve_images_by_image(query_image_path: str, k: int = 5, where: dict | None = None):
    """image -> similar food images (CLIP, 512-d)."""
    return _query(config.IMAGE_COLLECTION, embeddings.embed_images([query_image_path])[0], k, where)


def retrieve_images_by_text(query: str, k: int = 5, where: dict | None = None):
    """text -> food images (CLIP text tower into the same space as the image vectors)."""
    return _query(config.IMAGE_COLLECTION, embeddings.embed_query_clip_text(query), k, where)


def print_hits(ids, docs, metas, dists, title: str, max_chars: int = 180) -> None:
    print(f"\n=== {title} ===")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        dist = float(dists[i]) if i < len(dists) else float("nan")
        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        print(
            f"[{i + 1}] id={meta.get('doc_id', ids[i])} | cuisine={meta.get('cuisine', 'N/A')} | "
            f"location={meta.get('location', 'N/A')} | source={meta.get('source', 'N/A')} | "
            f"distance={dist:.4f}"
        )
        print(snippet)
