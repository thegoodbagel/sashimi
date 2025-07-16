import pandas as pd
import os

data = []

# Go through files in raw image folder
for filename in os.listdir("./data/raw"):
    if filename.endswith(".jpg"):
        if "salmon" in filename.lower():
            label = "salmon"
        elif "bluefin-tuna" in filename.lower():
            label = "bluefin-tuna"
        else:
            continue
        data.append({"filename": filename, "label": label})

# Write to CSV
df = pd.DataFrame(data)
df.to_csv("./data/sushi_labels.csv", index=False)
print("✅ sushi_labels.csv created.")
