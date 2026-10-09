"""Lab 6 demos: multimodal fusion, filters, weight tuning.

Run (after building the index):  python scripts/demo_fusion.py
"""
from connoisseur.rag.fusion import fuse_rank, print_fused


def main() -> None:
    # Demo 1 - no filters
    rows = fuse_rank("cozy noodles with warm atmosphere", top_n=5)
    print_fused(rows, title="Demo 1 — Multimodal fusion (no filters)")

    # Demo 2 - metadata filters
    rows = fuse_rank(
        "handmade pasta and romantic dinner",
        where_text={"location": "Pasadena"},   # change to a location in your dataset
        where_img={"source": "recipe_image"},
        top_n=5,
    )
    if not rows:
        print("No results found. Try relaxing filters (location/cuisine/source).")
    else:
        print_fused(rows, title="Demo 2 — Multimodal fusion (metadata filters)")

    # Demo 3 - weight tuning (this was the TODO in the lab)
    q = "fresh sushi and minimalist presentation"
    rows = fuse_rank(q, k_text=5, k_img=5, w_text=0.8, w_img=0.2, top_n=5)
    print_fused(rows, title="Demo 3A — Text-heavy fusion (w_text=0.8, w_img=0.2)")
    rows = fuse_rank(q, k_text=5, k_img=5, w_text=0.3, w_img=0.7, top_n=5)
    print_fused(rows, title="Demo 3B — Image-heavy fusion (w_text=0.3, w_img=0.7)")


if __name__ == "__main__":
    main()
