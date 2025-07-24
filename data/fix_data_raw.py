import os
import re

dir_path = "data/raw/sake_(salmon)_sashimi"

for fname in os.listdir(dir_path):
    match = re.match(r"sake_\(salmon\)_(?:harasu|toro)_sashimi_(\d+)", fname)
    if match:
        number = match.group(1)
        new_name = f"sake_(salmon)_sashimi_{number}"
        src = os.path.join(dir_path, fname)
        dst = os.path.join(dir_path, new_name)
        os.rename(src, dst)
        print(f"✅ Renamed: {fname} → {new_name}")
