"""Download the course datasets that have known URLs into data/raw/.

Usage:  python scripts/download_data.py [--force]

NOT covered (no URL in your lab files) - copy these into data/raw/ by hand:
  * Synthetic-User-Reviews.json   (Lab 2 input)
  * Recipes.json                  (Lab 2 input; may already be inside the recipe zip)
"""
import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from connoisseur import config  # noqa: E402

BASE = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud"
FILES = {
    config.RAW_DIR / "California-Culinary-Map.txt":
        f"{BASE}/taf6n5kHS3H2ZLRf9tihFw/California-Culinary-Map.txt",
    config.RAW_DIR / "synthetic-recipe-images.zip":
        f"{BASE}/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip",
    config.MCP_DATA_DIR / "structured-restaurant-data.json":
        f"{BASE}/clYj6XUC9urkUzWtiDH0Cw/structured-restaurant-data.json",
    config.MCP_DATA_DIR / "augmented-user-review.json":
        f"{BASE}/q5CWrWtux9-NmnkiWOrQBw/augmented-user-review.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    config.ensure_dirs()

    for dest, url in FILES.items():
        if dest.exists() and not args.force:
            print(f"skip  {dest.relative_to(config.PROJECT_ROOT)} (exists)")
            continue
        print(f"get   {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"saved {dest.relative_to(config.PROJECT_ROOT)} ({len(resp.content):,} bytes)")

    print("\nStill needed by hand: Synthetic-User-Reviews.json (+ Recipes.json if not in the zip) -> data/raw/")


if __name__ == "__main__":
    main()
