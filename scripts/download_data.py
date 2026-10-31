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

#**************************************************************************
# __file__: A Python built-in variable, pronounced as "dunder file"
#           It contains the file path of the script currently being executed
#
# Path(__file__): This creates a pathlib.Path object and
#                 initialize it with the file path string of your current running script.
#
# Path.resolve(): the method resolves any relative path segments (such as . and ..)
#                 and any symbolic links in the object Path,
#                 returns the canonical absolute path as a new Path object.
#
# Path.parents: a immutable sequence object that contains the entire sequence of ancestor directories
#               e.g., path = Path("/usr/local/bin/python3")
#               .parents[0]: immediate parent "/usr/local/bin"
#               .parents[1]: grandparent parent "/usr/local"
#               .parents[2]: great-grandparent "/usr"
#
# Path(__file__).resolve().parents[1] / "src":
#   for this script in "....../restaurant_companion/scripts/download_data.py
#   .parents[1] returns a Path object for "....../restaurant_companion" directory
#   
# str(Path(__file__).resolve().parents[1] / "src"):
#   converts Path object to string for "....../restaurant_companion/src"
#
# sys.path: determines the search path for modules. 
#
# sysm.path.insert(0, search_path_string): 
#   inserts "search_path_string" to be the first in search path.
#

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#**************************************************************************
# According to PEP 8 (Python Enhancement Proposal - Python's official style guide), 
# all import statements must be placed at the absolute top of the file before any other code. 
#
# However, we must execute code (sys.path.insert(...)) before importing 
# the 'connoisseur' module so Python can find the local 'src' directory.
#
# Because the import happens after this sys.path modification, code editors 
# and linters will throw an out-of-order warning. 
# The '# noqa: E402' comment is used to silence this specific alert:
#  - 'noqa' stands for 'No Quality Assurance' (ignore warnings).
#  - 'E402' isolates the exception to the out-of-order import rule, 
#    ensuring the linter still checks the line for other syntax errors.

from connoisseur import config  # noqa: E402

#**************************************************************************
# config.RAW_DIR, config.MCP_DATA_DIR, config.MCP_DATA_DIR:
#    All are Path objects
#
# config.RAW_DIR / "filename.txt":
#   returns a new Path object.
#

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

        # ************************************************************
        # requests:  a popular Python library for handling HTTP requests 
        # urllib: Python built-in library for for hanlding HTTP requests
        #         requests is popular for its simplicity over urllib 
        #         while urllib requires a lot of complex boilerplate code
        #         for the same HTTP requst handling.
        # requests.get(url, timeout=60): 
        #   sends a synchronous HTTP GET request 
        #   returns a response object:
        #       response.status_code, response.content, reponse.headers
        #       response.ok, response.text, response.json()
        #

        resp = requests.get(url, timeout=60)

        # ************************************************************
        # response.raise_for_status():
        #   this method instantly exits your program with an exception 
        #   if the server sends back an error (i.e., status codes 400–599)
        #   

        resp.raise_for_status()


        # ************************************************************
        # Path.write_types(content): 
        #   writes content to the file specifed by Path
        #   if the file exists, it is overwritten
        #   if the file does not exist, it is created
        #

        dest.write_bytes(resp.content)

        # ************************************************************
        # Path.relative_to(project_dir): 
        #    returns a new Path object that holds the relative path to project_dir.
        #    project_dir is a Path object holding the directoy path of your project
        #

        print(f"saved {dest.relative_to(config.PROJECT_ROOT)} ({len(resp.content):,} bytes)")

    print("\nStill needed by hand: Synthetic-User-Reviews.json (+ Recipes.json if not in the zip) -> data/raw/")


if __name__ == "__main__":
    main()
