import os
import requests
from urllib.parse import urlencode
from pathlib import Path
import time

# ✏️ Replace with your credentials
API_KEY = "AIzaSyBIRmVvkVL9T5nnoKrCEjA11YuW69K-Fow"
CSE_ID = "d539947c708134729"

# Where to save downloaded images
SAVE_DIR = "./data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

# Your full list of categories
CATEGORIES = [
    "sake salmon toro", "sake salmon harasu", "ikura salmon roe",
    "maguro otoro", "maguro chutoro", "maguro akami",
    "honmaguro", "minamimaguro", "mebachimaguro",
    "kihadamaguro", "binnagamaguro", "ahi tuna",
    "katsuo tataki", "buri hamachi", "hiramasa", "kanpachi",
    "engawa", "hotate scallop",
    "amaebi", "botan ebi", "aka ebi", "kurama ebi",
    "tai madai", "tai kurodai", "tai chidai", "tai kinmedai",
    "tai ishidai", "tai himedai",
    "shime saba", "aji", "suzuki sea bass",
    "hokkigai", "akagai", "tsubagai", "mirugai",
    "ika squid", "tako octopus", "hiougi scallop",
    "ika somen", "kujira whale"
]

def download_images_for_query(query, max_results=10):
    print(f"🔍 Searching: {query} Sashimi")
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

        # time.sleep(1.5)  # prevent rate limit

def main():
    for cat in CATEGORIES:
        download_images_for_query(cat, max_results=20)

if __name__ == "__main__":
    main()
