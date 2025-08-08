
import time
import requests
import os
from duckduckgo_search import DDGS
from concurrent.futures import ThreadPoolExecutor

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
def duckduckgo_query(query, save_path, max_results=20, start_idx=0):
    print(f"🔍 DuckDuckGo Search: {query}")
    print("START INDEX:", start_idx)

    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_results)

        for i, result in enumerate(results):
            image_url = result.get("image")
            if not image_url:
                continue
            ext = os.path.splitext(image_url)[-1].split("?")[0].lower()
            ext = ext if ext in [".jpg", ".jpeg", ".png"] else ".jpg"
            filename = os.path.join(save_path, f"{query.replace(' ', '_')}_{start_idx + i}{ext}")
            
            download_image(image_url, filename)
            time.sleep(1)  # 🔴 Add delay between each DuckDuckGo request


# 🔍 Google
def google_query(query, save_path, max_results=20, start_idx=0):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    print(f"🔍 Google Search: {query}")

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
            filename = os.path.join(save_path, f"{query.replace(' ', '_')}_{start_idx+start+i}{ext}")
            jobs.append((image_url, filename))
        time.sleep(1.5)

    for image_url, filename in jobs:
        download_image(image_url, filename)


def bing_query(query, save_path, max_results=20, start_idx=0):
    BING_API_KEY = os.getenv("BING_API_KEY")  # Set as env var
    endpoint = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": max_results}

    response = requests.get(endpoint, headers=headers, params=params)
    if response.status_code != 200:
        print(f"❌ Bing Error: {response.text}")
        return

    data = response.json()

    for i, img in enumerate(data.get("value", [])):
        image_url = img.get("contentUrl")
        if not image_url:
            continue
        ext = os.path.splitext(image_url)[-1].split("?")[0].lower() or ".jpg"
        ext = ext if ext in [".jpg", ".jpeg", ".png"] else ".jpg"
        filename = os.path.join(save_path, f"{query.replace(' ', '_')}_{start_idx+i}{ext}")
        download_image(image_url, filename)
        time.sleep(1)