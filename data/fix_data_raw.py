import os
import shutil

RAW_DIR = "./data/raw"

def clean_bad_folders_and_files():
    for folder_name in os.listdir(RAW_DIR):
        folder_path = os.path.join(RAW_DIR, folder_name)

        # Skip if not a folder
        if not os.path.isdir(folder_path):
            continue

        # Remove folder if it doesn't end with _sashimi
        if not folder_name.endswith("_sashimi"):
            print(f"🗑️ Removing folder (invalid name): {folder_name}")
            shutil.rmtree(folder_path)
            continue

        # Inside valid folder, remove any files that don't contain _sashimi
        for fname in os.listdir(folder_path):
            if "_sashimi" not in fname:
                file_path = os.path.join(folder_path, fname)
                print(f"🗑️ Removing file (invalid name): {file_path}")
                os.remove(file_path)

if __name__ == "__main__":
    clean_bad_folders_and_files()
