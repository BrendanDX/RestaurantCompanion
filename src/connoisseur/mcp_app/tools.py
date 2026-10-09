"""Lab 10: the MCP tool/resource functions as plain Python (so they are unit-testable).

`server.py` registers these with FastMCP. Docstrings are the descriptions the LLM sees,
so keep them accurate.

The course datasets and the Lab 1 output use slightly different field names
(cuisine vs food_style, neighborhood vs location, vibes vs vibe, ...), so fields are
read defensively with fallbacks instead of hard `record["key"]` lookups.
"""
from __future__ import annotations

import json

from connoisseur import config


def load_restaurant_data() -> list[dict]:
    with open(config.MCP_RESTAURANT_DATA, "r", encoding="utf-8") as f:
        return json.load(f)


def load_review_data() -> list[dict]:
    with open(config.MCP_REVIEW_DATA, "r", encoding="utf-8") as f:
        return json.load(f)


def _vibes_of(restaurant: dict) -> list[str]:
    vibes = restaurant.get("vibes", restaurant.get("vibe", []))
    if isinstance(vibes, str):
        vibes = [vibes]
    return [str(v) for v in vibes]


# ----------------------------------------------------------------- resource
def get_culinary_map() -> str:
    """The full raw California Culinary Map text.
    Contains detailed descriptions of 100+ restaurants across California
    including their vibes, cuisines, ratings, and price ranges."""
    return config.CULINARY_MAP_PATH.read_text(encoding="utf-8")


# ------------------------------------------------------------------- tool 1
def get_restaurant_info(restaurant_name: str) -> str:
    """Search for a restaurant by name and return its structured details
    including cuisine, rating, price range, and signature dish."""
    restaurants = load_restaurant_data()
    query = restaurant_name.lower().strip()

    matches = []
    for restaurant in restaurants:
        name = str(restaurant.get("name", "")).lower()
        if name and (query in name or name in query):
            matches.append(restaurant)

    if not matches:
        return json.dumps(
            {
                "status": "not_found",
                "message": f"No restaurant found matching '{restaurant_name}'.",
                "suggestion": "Try a partial name like 'Iron' or 'Sakura'.",
            },
            indent=2,
        )
    return json.dumps({"status": "found", "count": len(matches), "results": matches}, indent=2)


# ------------------------------------------------------------------- tool 2
def recommend_by_vibe(vibe: str) -> str:
    """Find restaurants that match a given vibe or atmosphere keyword.
    Searches both structured vibe tags and raw text descriptions.
    Examples of vibe keywords: "moody", "sun-drenched", "romantic"."""
    restaurants = load_restaurant_data()
    vibe_lower = vibe.lower().strip()

    # Pass 1: structured vibe tags / description in the JSON
    structured_matches = []
    for r in restaurants:
        vibes = _vibes_of(r)
        description = str(r.get("description") or r.get("environment") or "").lower()
        if any(vibe_lower in v.lower() for v in vibes) or vibe_lower in description:
            structured_matches.append(
                {
                    "name": r.get("name"),
                    "neighborhood": r.get("neighborhood") or r.get("location"),
                    "cuisine": r.get("cuisine") or r.get("food_style"),
                    "rating": r.get("rating"),
                    "vibes": vibes,
                    "price_range": r.get("price_range"),
                }
            )

    # Pass 2: the raw culinary map text
    raw_text = config.CULINARY_MAP_PATH.read_text(encoding="utf-8")
    text_excerpts = [
        para.strip()[:300]
        for para in raw_text.split("\n\n")
        if para.strip() and vibe_lower in para.lower()
    ]

    return json.dumps(
        {
            "vibe_searched": vibe,
            "structured_matches": structured_matches,
            "raw_text_excerpts": text_excerpts[:5],
        },
        indent=2,
    )


# ------------------------------------------------------------------- tool 3
def get_review(restaurant_name: str) -> str:
    """Retrieve the full review for a restaurant."""
    reviews = load_review_data()
    query = restaurant_name.lower().strip()

    match = next((r for r in reviews if query in str(r.get("restaurant_name", "")).lower()), None)
    if not match:
        return json.dumps(
            {"status": "not_found", "message": f"No review found matching '{restaurant_name}'."},
            indent=2,
        )

    return json.dumps(
        {
            "status": "found",
            "restaurant": match.get("restaurant_name"),
            "reviewer": match.get("reviewer", "N/A"),
            "rating": match.get("rating"),
            "review_text": match.get("review_text") or match.get("reviews"),
            "image_description": match.get("image_description", "N/A"),
            "visit_date": match.get("visit_date", "N/A"),
        },
        indent=2,
    )
