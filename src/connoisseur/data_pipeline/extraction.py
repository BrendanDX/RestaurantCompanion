"""Lab 1: turn unstructured restaurant paragraphs into validated JSON.

The prompt builders, validator and repair loop here are also reused by the
CRUD app (Lab 3), so there is one copy of the extraction logic.

Run:  python -m connoisseur.data_pipeline.extraction [--limit 3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from connoisseur import config
from connoisseur.llm_client import chat
from connoisseur.utils import parse_json_response

LLMFn = Callable[[str, str], str]

MAX_REPAIR_ATTEMPTS = 5
ITEM_ID_BASE = 1_000_000

EXAMPLE_RESTAURANT_PARAGRAPH = (
    "Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, "
    "**casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, "
    "it captures the salt-air energy of the coast through its signature beer-battered "
    "snapper tacos and zesty octopus ceviche, making it a premier spot for open-air "
    "dining near the pier. Price range: $"
)

EXAMPLE_OUTPUT = """
{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
}
"""

REQUIRED_FIELDS: dict[str, type | tuple] = {
    "name": str,
    "location": str,
    "type": str,
    "food_style": str,
    "rating": (int, float),
    "price_range": int,
    "signatures": list,
    "vibe": str,
    "environment": str,
    "shortcomings": list,
}


class ExtractionError(RuntimeError):
    """Raised when the model cannot produce valid JSON within the repair budget."""


# --------------------------------------------------------------------- loading
def load_restaurant_paragraphs(path: str | Path | None = None) -> list[str]:
    """Split the culinary map on blank lines and drop the title chunk at the top."""
    text = Path(path or config.CULINARY_MAP_PATH).read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks[1:]


# --------------------------------------------------------------------- prompts
def restaurant_data_structure_prompt_generation(restaurant_paragraph: str) -> tuple[str, str]:
    base_system_msg = """
    You are a precise data extraction assistant. Your job is to read an unstructured
    restaurant description and convert it into a single, valid JSON object containing
    exactly the following fields: name, location, type, food_style, rating, price_range,
    signatures, vibe, environment, and shortcomings.

    Rules:
    - Output ONLY the JSON object. Do not include any explanation, commentary, or markdown code fences.
    - "rating" must be a float (e.g., 4.5).
    - "price_range" must be an integer representing the number of dollar signs (e.g., "$$$$" becomes 4, "$" becomes 1).
    - "signatures" must be a list of strings naming the signature dishes mentioned.
    - "shortcomings" must be a list of strings; if none are mentioned, return an empty list.
    - Use the exact field names and structure shown in the example below.
    """

    base_user_prompt = f"""
    Task:
    Extract the restaurant's name, location, type, food style, rating, price range,
    signature dishes, vibe, environment description, and any shortcomings from the
    restaurant description below. Return the result as a single valid JSON object
    following the exact format shown in the example.

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}
    """
    return base_system_msg, base_user_prompt


def JSON_auto_repair_prompts(candidate_json_output: str, error_message: str) -> tuple[str, str]:
    auto_repair_system_msg = """
    You are a JSON repair assistant. You will be given a JSON output that failed
    validation, along with the specific error message describing what is wrong.
    Your job is to fix ONLY the issue described in the error message while preserving
    all other correct data exactly as it is.

    Rules:
    - Output ONLY the corrected JSON object. Do not include any explanation, commentary, or markdown code fences.
    - Do not change any field values that are not related to the reported error.
    - Ensure the final output is valid, parseable JSON.
    """
    auto_repair_prompt = f"""
    The following JSON output failed validation:

    {candidate_json_output}

    Validation error message:
    {error_message}

    Please correct the JSON so that it passes validation, following the rules above.
    Return only the corrected JSON object.
    """
    return auto_repair_system_msg, auto_repair_prompt


# ------------------------------------------------------------------ validation
def validate_restaurant_json(data: dict) -> bool:
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in data:
            raise ValueError(f"Missing required field: '{field}'")
        if not isinstance(data[field], expected_type):
            raise ValueError(
                f"Field '{field}' should be of type {expected_type}, "
                f"but got {type(data[field])}"
            )

    for field in ("signatures", "shortcomings"):
        if not all(isinstance(item, str) for item in data[field]):
            raise ValueError(f"All items in '{field}' must be strings")

    return True


# ------------------------------------------------------------ extraction logic
def extract_restaurant(
    paragraph: str,
    item_id: int | None = None,
    llm: LLMFn = chat,
    max_repairs: int = MAX_REPAIR_ATTEMPTS,
) -> dict:
    """Generate -> validate -> (repair -> validate)* with a hard cap on repairs."""
    sys_msg, user_prompt = restaurant_data_structure_prompt_generation(paragraph)
    candidate = llm(sys_msg, user_prompt)

    for attempt in range(max_repairs + 1):
        try:
            parsed = parse_json_response(candidate)
            validate_restaurant_json(parsed)
            break
        except ValueError as e:  # json.JSONDecodeError is a ValueError subclass
            if attempt == max_repairs:
                raise ExtractionError(
                    f"Still invalid after {max_repairs} repair attempts: {e}"
                ) from e
            repair_sys, repair_prompt = JSON_auto_repair_prompts(candidate, str(e))
            candidate = llm(repair_sys, repair_prompt)

    if item_id is not None:
        parsed["itemId"] = item_id
    return parsed


def run_extraction(
    paragraphs: list[str],
    output_path: str | Path,
    llm: LLMFn = chat,
    progress_every: int = 20,
) -> list[dict]:
    """Extract every paragraph and write the JSON file (partial results are still saved on a crash)."""
    output_path = Path(output_path)
    records: list[dict] = []
    try:
        for i, paragraph in enumerate(paragraphs):
            try:
                records.append(extract_restaurant(paragraph, ITEM_ID_BASE + i, llm=llm))
            except ExtractionError as e:
                print(f"Skipping paragraph {i}: {e}")
            if (i + 1) % progress_every == 0:
                print(f"{i + 1} out of {len(paragraphs)} is done")
    finally:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
        print(f"Saved {len(records)} records to {output_path}")
    print("ALL DONE!")
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract structured restaurant data (Lab 1).")
    parser.add_argument("--limit", type=int, help="only process the first N paragraphs (cheap test run)")
    parser.add_argument("--output", type=Path, help="output JSON path")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    paragraphs = load_restaurant_paragraphs()
    print(f"Found {len(paragraphs)} restaurant paragraphs")
    default_out = config.STRUCTURED_RESTAURANTS_PATH
    if args.limit:
        paragraphs = paragraphs[: args.limit]
        # never overwrite the full dataset with a sample
        default_out = default_out.with_name("structured_restaurant_data.sample.json")
    run_extraction(paragraphs, args.output or default_out)


if __name__ == "__main__":
    main()
