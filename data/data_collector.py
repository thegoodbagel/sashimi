import os
import time
import requests
from pathlib import Path
from duckduckgo_search import DDGS
from concurrent.futures import ThreadPoolExecutor
from categories import CATEGORIES

# 🔧 Google API Setup
GOOGLE_API_KEY = "AIzaSyA_CEuDqQ6hSzSdwbwK1uFzQez_dIAJEM4"
GOOGLE_CSE_ID = "d539947c708134729"

# 📁 Save location
SAVE_DIR = "./data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🧠 Helpers
def get_save_path(source: str, query: str):
    safe_name = query.replace(" ", "_")
    path = os.path.join(SAVE_DIR, source, safe_name)
    os.makedirs(path, exist_ok=True)
    return path

def get_start_index(path):
    return len(list(Path(path).glob("*.[jp][pn]g")))

def download_image(image_url, filename):
    try:
        response = requests.get(image_url, timeout=5)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"✅ Saved: {filename}")
        else:
            print(f"❌ Failed to download: {image_url}")
    except Exception as e:
        print(f"⚠️ Skipped: {e}")

# 🔍 DuckDuckGo
def duckduckgo_query(query, max_results=20, max_workers=5):
    print(f"🔍 DuckDuckGo Search: {query}")
    save_path = get_save_path("duckduckgo", query)
    start_idx = get_start_index(save_path)

    with DDGS() as ddgs:
        results = list(ddgs.images(query, max_results=max_results))

    jobs = []
    for i, result in enumerate(results):
        image_url = result.get("image")
        if not image_url:
            continue
        ext = os.path.splitext(image_url)[-1].split("?")[0].lower()
        ext = ext if ext in [".jpg", ".jpeg", ".png"] else ".jpg"
        filename = os.path.join(save_path, f"{query.replace(' ', '_')}_{start_idx + i}{ext}")
        jobs.append((image_url, filename))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for image_url, filename in jobs:
            executor.submit(download_image, image_url, filename)

# 🔍 Google
def google_query(query, max_results=20, max_workers=5):
    print(f"🔍 Google Search: {query}")
    save_path = get_save_path("google", query)
    start_idx = get_start_index(save_path)

    jobs = []
    for start in range(1, max_results, 10):
        params = {
            "q": query,
            "cx": GOOGLE_CSE_ID,
            "key": GOOGLE_API_KEY,
            "searchType": "image",
            "num": min(10, max_results - start + 1),
            "start": start
        }

        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        if response.status_code != 200:
            print(f"❌ Error: {response.text}")
            continue

        results = response.json().get("items", [])
        for i, item in enumerate(results):
            image_url = item.get("link")
            if not image_url:
                continue
            ext = os.path.splitext(image_url)[-1].split("?")[0].lower() or ".jpg"
            ext = ext if ext in [".jpg", ".jpeg", ".png"] else ".jpg"
            filename = os.path.join(save_path, f"{query.replace(' ', '_')}_{start+i}{ext}")
            jobs.append((image_url, filename))
        time.sleep(1.5)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for image_url, filename in jobs:
            executor.submit(download_image, image_url, filename)

# 🔁 Main
def main():
    for cat in CATEGORIES:
        query = f"{cat} sashimi"
        google_query(query, max_results=20)
        # duckduckgo_query(query, max_results=20)

if __name__ == "__main__":
    main()
