import os
import argparse
from PIL import Image, UnidentifiedImageError
import pandas as pd
import torch
from torchvision import transforms
from data.filter.food_filter import SushiFilterModel, predict
import hashlib

RAW_DIR = './data/dish/raw'
PROCESSED_DIR = './data/fidishsh/processed'
LABELS_FILE = './data/dish/sushi_labels.csv'
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# Parse command line args
parser = argparse.ArgumentParser(description="Process sushi images by species filter.")
parser.add_argument(
    "-s", "--species",
    nargs="+",
    help="List of sushi species to process, e.g. -s salmon maguro hokkigai"
)
args = parser.parse_args()

# If no species specified, clear processed directory; otherwise preserve it
os.makedirs(PROCESSED_DIR, exist_ok=True)
if not args.species:
    print(f"Clearing processed directory {PROCESSED_DIR} (full reprocess)")
    for filename in os.listdir(PROCESSED_DIR):
        file_path = os.path.join(PROCESSED_DIR, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
    
    # Clear the CSV file as well
    if os.path.exists(LABELS_FILE):
        try:
            os.remove(LABELS_FILE)
            print(f"Cleared labels file {LABELS_FILE}")
        except Exception as e:
            print(f"Failed to clear labels file: {e}")
else:
    print(f"Processing only specified species: {args.species}")
    print(f"Preserving contents of {PROCESSED_DIR}")

# Load model & device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SushiFilterModel().to(device)
model.load_state_dict(torch.load('./data/best_sushi_filter.pth', map_location=device))
model.eval()

CONFIDENCE_THRESHOLD = 0.7

def get_image_hash(image: Image.Image) -> str:
    hasher = hashlib.sha1()
    hasher.update(image.tobytes())
    return hasher.hexdigest()

def process_and_save_image(src_path, dst_path):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            pred, conf = predict(model, img, device)

            if pred == 1 and conf >= CONFIDENCE_THRESHOLD:
                img.convert("RGB").save(dst_path, format='JPEG')
                print(f"✅ Saved (passed filter): {dst_path}")
                return True
            else:
                print(f"❌ Filtered out (not sushi/sashimi or low confidence): {src_path} (conf: {conf:.2f})")
                return False

    except (UnidentifiedImageError, OSError) as e:
        print(f"❌ Skipping {src_path}: {e}")
        return False

def main():
    labels = []

    for label_folder in os.listdir(RAW_DIR):
        folder_path = os.path.join(RAW_DIR, label_folder)
        if not os.path.isdir(folder_path):
            continue

        # Extract species from folder name
        if label_folder.endswith('_sashimi'):
            species = label_folder[:-len('_sashimi')]
        else:
            species = label_folder
        print("Species: ", species)

        # Skip if species filter is on and doesn't match
        if args.species:
            match_found = any(filter_str.lower() in label_folder.lower() for filter_str in args.species)
            if not match_found:
                continue

        print(f"Processing species '{species}'")
        seen_hashes = set()
        for fname in os.listdir(folder_path):
            name, ext = os.path.splitext(fname.lower())
            if ext not in VALID_EXTENSIONS:
                continue

            src_path = os.path.join(folder_path, fname)
            dst_filename = f"{label_folder}_{name}.jpg"
            dst_path = os.path.join(PROCESSED_DIR, dst_filename)

            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    img_hash = get_image_hash(img)

                    if img_hash in seen_hashes:
                        print(f"⚠️ Duplicate image skipped: {src_path}")
                        continue

                    pred, conf = predict(model, img, device)

                    if pred == 1 and conf >= CONFIDENCE_THRESHOLD:
                        img.save(dst_path, format='JPEG')
                        print(f"✅ Saved (passed filter): {dst_path}")
                        seen_hashes.add(img_hash)
                        labels.append({
                            "filename": dst_filename,
                            "species": species
                        })
                    else:
                        print(f"❌ Filtered out: {src_path} (conf: {conf:.2f})")

            except (UnidentifiedImageError, OSError) as e:
                print(f"❌ Skipping {src_path}: {e}")
                continue


    # Load existing CSV if present and non-empty
    try:
        if os.path.exists(LABELS_FILE) and os.path.getsize(LABELS_FILE) > 0:
            existing_df = pd.read_csv(LABELS_FILE)
        else:
            raise pd.errors.EmptyDataError
    except pd.errors.EmptyDataError:
        existing_df = pd.DataFrame(columns=["filename", "species"])

    # Drop existing rows that match the newly processed species
    if args.species:
        for s in args.species:
            existing_df = existing_df[~existing_df["species"].str.lower().str.contains(s.lower(), na=False)]

    # Add new labels
    df_new = pd.DataFrame(labels)
    final_df = pd.concat([existing_df, df_new], ignore_index=True)

    # Save updated CSV
    final_df.to_csv(LABELS_FILE, index=False)
    print(f"📝 Labels saved to {LABELS_FILE}")



if __name__ == "__main__":
    main()
    print("✅ Data processing complete!")
