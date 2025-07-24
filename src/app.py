import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from sushi_classifier import SushiClassifier, predict
from sushi_guide import show_sushi_guide

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
from categories import CATEGORIES

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = SushiClassifier(num_classes=2)
model.load_state_dict(torch.load("./saved_models/best_model.pth", map_location=device))
model.to(device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Page layout
st.set_page_config(page_title="Sashimi Classifier", layout="wide")
st.title("Sashimi Classifier 🍣")
st.write("Upload an image and get a prediction!")

# 👇 Show guide
show_sushi_guide()
# User option: Upload or Take a Photo
st.markdown("### 📸 Upload or Take a Picture of Your Sashimi")

# 1. Let user take photo with camera
camera_image = st.camera_input("Take a photo")

# 2. Let user upload a photo
uploaded_file = st.file_uploader("...or upload an image", type=["jpg", "png"])

# Prioritize camera image if both are used
input_image = None
if camera_image is not None:
    input_image = Image.open(camera_image).convert("RGB")
elif uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")

# Show prediction if any image is provided
if input_image is not None:
    st.image(input_image, caption='Input Image', use_column_width=True)
    st.write("🔍 Running inference...")

    label, confidence = predict(model, input_image, transform, CATEGORIES, device)
    st.success(f"🍣 Prediction: **{label.title().upper()}** ({confidence * 100:.1f}% confidence)")
