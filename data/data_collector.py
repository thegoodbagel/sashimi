import os
from pathlib import Path
from data.fish_categories import CATEGORIES
from query_engines import *

# 📁 Save location
SAVE_DIR = "./data/fish/raw"
os.makedirs(SAVE_DIR, exist_ok=True)
# 🔧 Google API Setup
os.env.set("GOOGLE_API_KEY" "AIzaSyA_CEuDqQ6hSzSdwbwK1uFzQez_dIAJEM4")
os.env.set("GOOGLE_CSE_ID", "d539947c708134729")
os.env.set("BING_API_KEY", )

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
    for cat in CATEGORIES:
        query = f"{cat} sashimi"
        save_dir = get_save_path(query)
        start_idx = get_start_index(save_dir)
        # google_query(query, save_dir, max_results=20, start_idx=start_idx)
        # duckduckgo_query(query, save_dir, max_results=20, start_idx=start_idx)
        bing_query(query, save_dir, max_results=20, start_idx=start_idx)

if __name__ == "__main__":
    main()
