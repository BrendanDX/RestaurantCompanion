"""Lab 6: weighted multimodal fusion of text-article and image retrieval scores."""
from __future__ import annotations

import numpy as np

from connoisseur.rag import retrieval


def _to_similarity(dists) -> np.ndarray:
    """Chroma distance (smaller = better) -> similarity (larger = better)."""
    return 1.0 - np.array(dists, dtype=np.float32)


def _minmax(x) -> np.ndarray:
    """Min-max normalise to [0, 1]; constant arrays all map to 1."""
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def _articles_with_sims(query: str, k: int, where: dict | None):
    ids, docs, metas, dists = retrieval.retrieve_articles(query, k=k, where=where)
    return ids, docs, metas, _to_similarity(dists)


def _images_with_sims(query: str, k: int, where: dict | None):
    ids, docs, metas, dists = retrieval.retrieve_images_by_text(query, k=k, where=where)
    return ids, docs, metas, _to_similarity(dists)


def _meta(metas, j: int, key: str, default="N/A"):
    return metas[j].get(key, default) if isinstance(metas[j], dict) else default


def _row(modality, ids, docs, metas, j, text_score, img_score, fused) -> dict:
    return {
        "modality": modality,
        "id": _meta(metas, j, "doc_id", ids[j]),
        "cuisine": _meta(metas, j, "cuisine"),
        "location": _meta(metas, j, "location"),
        "source": _meta(metas, j, "source"),
        "text_score": text_score,
        "img_score": img_score,
        "fused": fused,
        "snippet": (docs[j] or "").replace("\n", " ").strip(),
    }


def fuse_rank(
    query: str,
    k_text: int = 5,
    k_img: int = 5,
    w_text: float = 0.6,
    w_img: float = 0.4,
    where_text: dict | None = None,
    where_img: dict | None = None,
    top_n: int | None = 5,
) -> list[dict]:
    t_ids, t_docs, t_metas, t_sims = _articles_with_sims(query, k_text, where_text)
    i_ids, i_docs, i_metas, i_sims = _images_with_sims(query, k_img, where_img)

    t_norm, i_norm = _minmax(t_sims), _minmax(i_sims)  # normalise *within* each modality

    rows = [
        _row("article", t_ids, t_docs, t_metas, j, float(t_norm[j]), 0.0, float(w_text * t_norm[j]))
        for j in range(len(t_ids))
    ]
    rows += [
        _row("image", i_ids, i_docs, i_metas, j, 0.0, float(i_norm[j]), float(w_img * i_norm[j]))
        for j in range(len(i_ids))
    ]
    rows.sort(key=lambda r: r["fused"], reverse=True)

    if top_n is None:
        return rows
    return rows[: max(0, min(int(top_n), len(rows)))]


def print_fused(rows: list[dict], title: str, max_chars: int = 90) -> None:
    print(f"\n=== {title} ===")
    for idx, r in enumerate(rows, start=1):
        snippet = r["snippet"]
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        print(
            f"[{idx}] {r['modality']} | id={r['id']} | cuisine={r['cuisine']} | "
            f"location={r['location']} | fused={r['fused']:.4f} "
            f"(text={r['text_score']:.4f}, img={r['img_score']:.4f})"
        )
        print(snippet)
