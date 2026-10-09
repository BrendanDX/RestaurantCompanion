"""Lab 5 demos: article search, filtered search, image->image search.

Run (after building the index):  python scripts/demo_retrieval.py
"""
from connoisseur.rag import retrieval


def main() -> None:
    # Demo 1 - article similarity search (no filter)
    q = "cozy restaurant with noodles and warm atmosphere"
    retrieval.print_hits(*retrieval.retrieve_articles(q, k=5), title="Demo 1 — Article similarity search (no filter)")

    # Demo 2 - article similarity search + metadata filter
    q = "handmade pasta and romantic dinner"
    where_filter = {"location": "Pasadena"}  # adjust to a location in your dataset
    ids, docs, metas, dists = retrieval.retrieve_articles(q, k=5, where=where_filter)
    if not ids:
        print("No results found with current filter.")
    else:
        retrieval.print_hits(ids, docs, metas, dists, title="Demo 2 — Article similarity search + metadata filter")

    # Demo 3 - image -> image search (this was the TODO in the lab)
    meta_all = retrieval.get_collection("food_images").get(include=["metadatas"])["metadatas"]
    query_index = 0  # change to try different images
    query_img = meta_all[query_index]["image_path"]
    print(f"\nQuery image: {query_img}")

    where_img = None  # e.g. {"source": "recipe_image"}
    ids, docs, metas, dists = retrieval.retrieve_images_by_image(query_img, k=5, where=where_img)
    retrieval.print_hits(ids, docs, metas, dists, title="Demo 3 — Image similarity search (image→image)")


if __name__ == "__main__":
    main()
