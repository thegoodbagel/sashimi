import os
from PIL import Image, UnidentifiedImageError
import pandas as pd

RAW_DIR = './data/raw'
PROCESSED_DIR = './data/processed'
LABELS_FILE = './data/sushi_labels.csv'
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_and_save_image(src_path, dst_path):
    try:
        with Image.open(src_path) as img:
            img.convert("RGB").save(dst_path, format='JPEG')
            print(f"✅ Saved: {dst_path}")
    except (UnidentifiedImageError, OSError) as e:
        print(f"❌ Skipping {src_path}: {e}")

def main():
    labels = []

    for label_folder in os.listdir(RAW_DIR):
        folder_path = os.path.join(RAW_DIR, label_folder)
        if not os.path.isdir(folder_path):
            continue

        # Assume folder name format: species_part, e.g. maguro_otoro
        if '_' in label_folder:
            species, part = label_folder.split('_', 1)
        else:
            # fallback if no underscore present
            species = label_folder
            part = ''

        for fname in os.listdir(folder_path):
            name, ext = os.path.splitext(fname.lower())
            if ext not in VALID_EXTENSIONS:
                continue

            src_path = os.path.join(folder_path, fname)
            dst_filename = f"{label_folder}_{name}.jpg"
            dst_path = os.path.join(PROCESSED_DIR, dst_filename)

            process_and_save_image(src_path, dst_path)

            labels.append({
                "filename": dst_filename,
                "species": species,
                "part": part
            })

    df = pd.DataFrame(labels)
    df.to_csv(LABELS_FILE, index=False)
    print(f"📝 Labels saved to {LABELS_FILE}")

if __name__ == "__main__":
    main()
