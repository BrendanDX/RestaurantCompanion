from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/connoisseur/config.py -> parents[2] is the project root (works for `pip install -e .`)
PROJECT_ROOT = Path(os.environ.get("CONNOISSEUR_HOME") or Path(__file__).resolve().parents[2])
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------- directories
DATA_DIR = Path(os.environ.get("CONNOISSEUR_DATA_DIR") or PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"              # downloaded / provided inputs
MCP_DATA_DIR = RAW_DIR / "mcp"          # datasets used by the MCP server (Labs 10-12)
INTERIM_DIR = DATA_DIR / "interim"      # scratch files (e.g. downloaded review image)
PROCESSED_DIR = DATA_DIR / "processed"  # outputs of Module 1
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR") or DATA_DIR / "chroma_db")
RECIPES_DIR = RAW_DIR / "recipes"       # extracted recipe images (+ Recipes.json)

# ---------------------------------------------------------------- input files
CULINARY_MAP_PATH = RAW_DIR / "California-Culinary-Map.txt"
RECIPES_ZIP_PATH = RAW_DIR / "synthetic-recipe-images.zip"
USER_REVIEWS_PATH = RAW_DIR / "Synthetic-User-Reviews.json"

# ---------------------------------------------------------------- Module 1 outputs
STRUCTURED_RESTAURANTS_PATH = PROCESSED_DIR / "structured_restaurant_data.json"
AUGMENTED_RECIPES_PATH = PROCESSED_DIR / "augmented_food_recipe.json"
AUGMENTED_REVIEWS_PATH = PROCESSED_DIR / "augmented_user_review.json"
REVIEW_IMAGE_PLACEHOLDER = INTERIM_DIR / "review_image_placeholder.jpg"

# ---------------------------------------------------------------- MCP datasets
# Course-provided files (hyphenated names). Their schema differs slightly from the
# Module 1 outputs, so the MCP tools read fields defensively.
MCP_RESTAURANT_DATA = Path(os.environ.get("MCP_RESTAURANT_DATA") or MCP_DATA_DIR / "structured-restaurant-data.json")
MCP_REVIEW_DATA = Path(os.environ.get("MCP_REVIEW_DATA") or MCP_DATA_DIR / "augmented-user-review.json")

# ---------------------------------------------------------------- LLM (DeepSeek)
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_ANTHROPIC_BASE_URL = os.environ.get("DEEPSEEK_ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
TEXT_MODEL = os.environ.get("DEEPSEEK_TEXT_MODEL", "deepseek-v4-flash")
VISION_MODEL = os.environ.get("DEEPSEEK_VISION_MODEL", "deepseek-v4-flash-vision-exp")

# ---------------------------------------------------------------- embeddings
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"             # 384-d
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # 512-d
ARTICLE_COLLECTION = "restaurant_articles"
IMAGE_COLLECTION = "food_images"


def get_api_key() -> str:
    """Return the DeepSeek key or raise a helpful error. Read at call time, not import time."""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key or key == "your-key-here":
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and paste your key, "
            "or run: export DEEPSEEK_API_KEY=..."
        )
    return key


def ensure_dirs() -> None:
    """Create the data directories if they do not exist."""
    for d in (RAW_DIR, MCP_DATA_DIR, INTERIM_DIR, PROCESSED_DIR, CHROMA_DIR.parent):
        d.mkdir(parents=True, exist_ok=True)
