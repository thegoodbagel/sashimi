import os
from PIL import Image, UnidentifiedImageError
import pandas as pd
import torch
from torchvision import transforms
from food_filter import SushiFilterModel, predict  # assuming your model code is in food_filter.py

RAW_DIR = './data/raw'
PROCESSED_DIR = './data/processed'
LABELS_FILE = './data/sushi_labels.csv'
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

os.makedirs(PROCESSED_DIR, exist_ok=True)
# Clear all pre-existing files
if os.path.exists(PROCESSED_DIR):
    for filename in os.listdir(PROCESSED_DIR):
        file_path = os.path.join(PROCESSED_DIR, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# Load model & device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SushiFilterModel().to(device)
model.load_state_dict(torch.load('./data/best_sushi_filter.pth', map_location=device))
model.eval()

# Confidence threshold for filtering (tweak as needed)
CONFIDENCE_THRESHOLD = 0.7

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

        # Split on last underscore: e.g. 'hokkigai_(surf_clam)_sashimi'
        if '_' in label_folder:
            parts = label_folder.rsplit('_', 1)
            species = parts[0]
            part = parts[1]
        else:
            species = label_folder
            part = ''

        for fname in os.listdir(folder_path):
            name, ext = os.path.splitext(fname.lower())
            if ext not in VALID_EXTENSIONS:
                continue

            src_path = os.path.join(folder_path, fname)
            dst_filename = f"{label_folder}_{name}.jpg"
            dst_path = os.path.join(PROCESSED_DIR, dst_filename)

            if process_and_save_image(src_path, dst_path):
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
    print("✅ Data processing complete!")