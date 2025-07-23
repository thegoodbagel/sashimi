import os
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Directories
RAW_DIR = './data/raw'
PROCESSED_DIR = './data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Allowed formats
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def process_image(file_path, save_path):
    try:
        with Image.open(file_path) as img:
            img.save(save_path, format='JPEG')
            print(f"Saved: {save_path}")
    except (UnidentifiedImageError, OSError) as e:
        print(f"Skipping {file_path}: {e}")

def main():
    for fname in os.listdir(RAW_DIR):
        fpath = os.path.join(RAW_DIR, fname)
        name, ext = os.path.splitext(fname.lower())

        if ext in VALID_EXTENSIONS:
            out_path = os.path.join(PROCESSED_DIR, f"{name}.jpg")
            process_image(fpath, out_path)

        else:
            # Try to open and convert unsupported formats like .webp
            try:
                with Image.open(fpath) as img:
                    img = img.convert("RGB")
                    out_path = os.path.join(PROCESSED_DIR, f"{name}.jpg")
                    process_image(fpath, out_path)
            except (UnidentifiedImageError, OSError):
                print(f"Unsupported or unreadable file skipped: {fname}")

if __name__ == "__main__":
    main()
