import os
from pathlib import Path
from categories import FISH_CATEGORIES, DISH_CATEGORIES
from query_engines import google_query, duckduckgo_query


# 📁 Save location
SAVE_DIR = "./data/dish/raw"
os.makedirs(SAVE_DIR, exist_ok=True)
# 🔧 Google API Setup
os.environ["GOOGLE_API_KEY"] = "AIzaSyDjVzmB5LJ8B6_meRGVF6u0YQtfl4yBLfo"
os.environ["GOOGLE_CSE_ID"] = "d539947c708134729"

# 🧠 Helpers
def get_save_path(query: str):
    safe_name = query.replace(" ", "_")
    path = os.path.join(SAVE_DIR, safe_name)
    os.makedirs(path, exist_ok=True)
    return path

def get_start_index(path):
    return len(list(Path(path).glob("*.[jp][pn]g")))


# 🔁 Main
def main():
    for cat in DISH_CATEGORIES:
        query = f"{cat}"
        save_dir = get_save_path(query)
        # start_idx = get_start_index(save_dir)
        google_query(query, save_dir, 20, 0)
        # duckduckgo_query(query, save_dir, 20, 20)

if __name__ == "__main__":
    main()
