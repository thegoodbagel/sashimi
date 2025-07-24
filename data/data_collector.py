import os
import requests
from urllib.parse import urlencode
from pathlib import Path
import time
from categories import CATEGORIES

# ✏️ Replace with your credentials
API_KEY = "AIzaSyBouYfXAwAfix9aWPYQyrEALtlJjmtHWrQ"
CSE_ID = "d539947c708134729"

# Where to save downloaded images
SAVE_DIR = "./data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_images_for_query(query, max_results=10):
    print(f"🔍 Searching: {query}")
    save_path = os.path.join(SAVE_DIR, query.replace(" ", "_"))
    os.makedirs(save_path, exist_ok=True)

    for start in range(1, max_results, 10):
        print(f"Fetching images {start} to {start + 9} for query: {query}")
        params = {
            "q": query,
            "cx": CSE_ID,
            "key": API_KEY,
            "searchType": "image",
            "num": min(10, max_results - start + 1),
            "start": start
        }

        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        if response.status_code != 200:
            print(f"❌ Error for {query}: {response.text}")
            return

        results = response.json().get("items", [])
        for i, item in enumerate(results):
            try:
                image_url = item["link"]
                ext = os.path.splitext(image_url)[-1].split("?")[0]
                filename = os.path.join(save_path, f"{query.replace(' ', '_')}_{start+i}{ext}")
                img_data = requests.get(image_url, timeout=5).content
                with open(filename, "wb") as f:
                    f.write(img_data)
                print(f"✅ Saved: {filename}")
            except Exception as e:
                print(f"⚠️ Skipped one image for {query}: {e}")

        time.sleep(1.5)  # prevent rate limit

def main():
    for cat in CATEGORIES:
        download_images_for_query(cat + " sashimi", max_results=20)

if __name__ == "__main__":
    main()
