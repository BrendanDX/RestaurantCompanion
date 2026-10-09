"""Lab 2: caption recipe images and review images with the DeepSeek vision model.

Run:  python -m connoisseur.data_pipeline.image_captioning [--limit 3] [--skip-recipes] [--skip-reviews]
"""
from __future__ import annotations

import argparse
import ast
import json
import zipfile
from pathlib import Path
from typing import Callable

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from connoisseur import config
from connoisseur.llm_client import vision_chat

VisionFn = Callable[[str, str, "str | Path"], str]


# ---------------------------------------------------------------- recipe archive
def extract_recipe_archive(zip_path: Path | None = None, dest: Path | None = None) -> Path:
    zip_path = Path(zip_path or config.RECIPES_ZIP_PATH)
    dest = Path(dest or config.RECIPES_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    return dest


def find_recipes_json(search_dir: Path | None = None) -> Path:
    search_dir = Path(search_dir or config.RECIPES_DIR)
    for candidate in [*search_dir.rglob("Recipes.json"), config.RAW_DIR / "Recipes.json"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Recipes.json not found in the extracted archive or data/raw/")


def resolve_image_path(image_field: str, base_dir: Path) -> Path:
    """The recipes JSON stores relative image paths; find the file wherever it was extracted."""
    p = Path(image_field)
    if p.exists():
        return p
    if (base_dir / p).exists():
        return base_dir / p
    matches = list(base_dir.rglob(p.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image not found: {image_field}")


# ---------------------------------------------------------------------- prompts
def image_caption_prompt_template(food_name: str) -> tuple[str, str]:
    image_caption_system_msg = """
    You are a professional food photographer and culinary expert. Your job is
    to look at an image of a dish and produce a concise, accurate, and vivid
    description of what is visually shown. Focus only on what can actually be
    seen in the image: ingredients, colors, textures, plating style, garnishes,
    and portion size. Do not speculate about taste, smell, or ingredients that
    are not visible. Keep the description to 2-3 sentences.
    """

    image_caption_prompt_txt = f"""
    This image shows a dish called "{food_name}". Describe what you see in the
    image, focusing on the visible ingredients, presentation style, plating,
    garnishes, and portion size. Keep your description concise (2-3 sentences)
    and grounded only in what is visually present in the image.
    """
    return image_caption_system_msg, image_caption_prompt_txt


def review_context_image_caption_prompt_template(reviews: str) -> tuple[str, str]:
    review_context_image_caption_system_msg = """
    You are an experienced culinary expert and food critic. You will be shown
    an image of a dish along with a written customer review of that dish.
    Your job is to produce a short, accurate description of what is visually
    shown in the image, using the review text only as supporting context to
    help identify the dish and its notable qualities. Do not simply repeat the
    review; focus on describing what can be seen in the image itself.
    """

    review_context_image_caption_prompt_txt = f"""
    Here is a customer review associated with this image:

    "{reviews}"

    Based on the image and using the review above as context, write a concise
    (2-3 sentence) description of what is visually shown in the image,
    including ingredients, presentation, and any details that align with the
    sentiment or specifics mentioned in the review.
    """
    return review_context_image_caption_system_msg, review_context_image_caption_prompt_txt


# --------------------------------------------------------------------- download
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_data_with_retry(url: str) -> requests.Response:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response


# ---------------------------------------------------------------- captioning
def caption_recipes(
    recipe_data: list[dict],
    base_dir: Path,
    vision: VisionFn = vision_chat,
    progress_every: int = 20,
) -> list[dict]:
    for i, recipe in enumerate(recipe_data):
        if (i + 1) % progress_every == 0:
            print(f"{i + 1} out of {len(recipe_data)} is done")
        system_msg, prompt_txt = image_caption_prompt_template(recipe["name"])
        image_path = resolve_image_path(recipe["image"], base_dir)
        recipe["image_description"] = vision(system_msg, prompt_txt, image_path)
    print("ALL DONE!")
    return recipe_data


def caption_reviews(
    user_review_data: list[dict],
    vision: VisionFn = vision_chat,
    downloader: Callable[[str], "requests.Response"] = get_data_with_retry,
    placeholder: Path | None = None,
) -> list[dict]:
    """For each review: download every image, caption it using the review text as context."""
    placeholder = Path(placeholder or config.REVIEW_IMAGE_PLACEHOLDER)
    placeholder.parent.mkdir(parents=True, exist_ok=True)

    for review in user_review_data:
        # 'images' is stored as a *string* that looks like a Python list
        review_images = ast.literal_eval(review["images"])
        captions: list[str] = []
        for img_url in review_images:
            try:
                image_data = downloader(img_url)
            except Exception as e:  # noqa: BLE001 - skip this image, keep the batch going
                print(f"All retries failed at url {img_url}:", e)
                continue
            with open(placeholder, "wb") as img_file:
                img_file.write(image_data.content)  # raw bytes -> binary mode

            system_msg, prompt_txt = review_context_image_caption_prompt_template(review["reviews"])
            captions.append(vision(system_msg, prompt_txt, placeholder))
        review["image_captions"] = captions
    print("ALL DONE!")
    return user_review_data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Caption recipe and review images (Lab 2).")
    parser.add_argument("--limit", type=int, help="only process the first N items of each dataset")
    parser.add_argument("--skip-recipes", action="store_true")
    parser.add_argument("--skip-reviews", action="store_true")
    args = parser.parse_args(argv)
    config.ensure_dirs()

    if not args.skip_recipes:
        base_dir = extract_recipe_archive()
        recipes_json = find_recipes_json(base_dir)
        recipe_data = json.loads(recipes_json.read_text(encoding="utf-8"))
        if args.limit:
            recipe_data = recipe_data[: args.limit]
        caption_recipes(recipe_data, recipes_json.parent)
        out = config.AUGMENTED_RECIPES_PATH
        if args.limit:
            out = out.with_name("augmented_food_recipe.sample.json")
        out.write_text(json.dumps(recipe_data, indent=4), encoding="utf-8")
        print(f"Saved {out}")

    if not args.skip_reviews:
        review_data = json.loads(config.USER_REVIEWS_PATH.read_text(encoding="utf-8"))
        if args.limit:
            review_data = review_data[: args.limit]
        caption_reviews(review_data)
        out = config.AUGMENTED_REVIEWS_PATH
        if args.limit:
            out = out.with_name("augmented_user_review.sample.json")
        out.write_text(json.dumps(review_data, indent=4), encoding="utf-8")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
