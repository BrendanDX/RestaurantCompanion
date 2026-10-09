"""Lab 3: interactive CRUD console app over the structured restaurant JSON.

Run:  python -m connoisseur.data_pipeline.restaurant_db
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from connoisseur import config
from connoisseur.data_pipeline.extraction import ITEM_ID_BASE, extract_restaurant


# ---------------------------------------------------------------- persistence
def default_backup_path(file_path: str | Path) -> Path:
    return Path(f"{file_path}.bak")


def load_data(file_path: str | Path) -> list[dict]:
    file_path = Path(file_path)
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data: list[dict], file_path: str | Path, backup_path: str | Path) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        shutil.copy(file_path, backup_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# -------------------------------------------------------------------- helpers
def new_data_entry_process(paragraph: str, itemId: int) -> dict:
    """One paragraph -> one validated record (patched out in unit tests)."""
    return extract_restaurant(paragraph, itemId)


def next_item_id(data: list[dict]) -> int:
    """Max existing itemId + 1 (len(data)+1 can collide with an existing id after a delete)."""
    ids = [r["itemId"] for r in data if isinstance(r.get("itemId"), int)]
    return (max(ids) + 1) if ids else ITEM_ID_BASE


def show_restaurant_card(res: dict, index: int) -> None:
    print(f"\n--- Record {index} ---")
    for key, value in res.items():
        print(f"{key}: {value}")


def coerce_like(old: Any, new_text: str) -> Any:
    """Keep the original type when editing (an edited rating must stay a number, not become '4.5')."""
    if isinstance(old, bool):
        return new_text
    if isinstance(old, int):
        return int(new_text)
    if isinstance(old, float):
        return float(new_text)
    if isinstance(old, list):
        return [part.strip() for part in new_text.split(",") if part.strip()]
    return new_text


def _read_index(data: list[dict]) -> int | None:
    raw = input("Enter record index: ")
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        print("invalid index.")
        return None
    if 0 <= idx < len(data):
        return idx
    print("invalid index.")
    return None


# ------------------------------------------------------------------- main loop
def manage_restaurants(file_path: str | Path | None = None, backup_path: str | Path | None = None) -> None:
    file_path = Path(file_path or config.STRUCTURED_RESTAURANTS_PATH)
    backup_path = Path(backup_path or default_backup_path(file_path))

    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == "1":
            print("\n--- Current Listings ---")
            for i, res in enumerate(data):
                print(f"{i}: {res.get('name', 'N/A')}")

        elif choice == "2":
            idx = _read_index(data)
            if idx is not None:
                show_restaurant_card(data[idx], idx)

        elif choice in ("3", "4", "5"):
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != "yes":
                print("Operation cancelled.")
                continue

            if choice == "3":
                new_paragraph = input("Enter the new restaurant description: ")
                record = new_data_entry_process(new_paragraph, next_item_id(data))
                data.append(record)
                save_data(data, file_path, backup_path)
                print("✅ Restaurant added.")

            elif choice == "4":
                idx = _read_index(data)
                if idx is None:
                    continue
                record = data[idx]
                for key in list(record.keys()):
                    if key == "itemId":
                        continue
                    new_value = input(f"{key} [{record[key]}]: ")
                    if new_value.strip() == "":
                        continue
                    try:
                        record[key] = coerce_like(record[key], new_value.strip())
                    except ValueError:
                        print(f"Could not convert '{new_value}' for '{key}'; keeping the old value.")
                save_data(data, file_path, backup_path)
                print("✅ Record updated.")

            elif choice == "5":
                idx = _read_index(data)
                if idx is None:
                    continue
                data.pop(idx)
                save_data(data, file_path, backup_path)
                print("✅ Record deleted.")

        elif choice == "6":
            break
        else:
            print("Invalid input.")


if __name__ == "__main__":
    manage_restaurants()
