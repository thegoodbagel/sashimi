import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from sushi_classifier import SushiClassifier, predict
from sushi_guide import show_sushi_guide

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ["Maguro (Bluefin Tuna)", "Salmon"]

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
st.write("Upload an image and get a prediction! (Currently only supports salmon and tuna, LOL)")

# 👇 Show guide
show_sushi_guide()

# Image prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Running inference...")

    label, confidence = predict(model, image, transform, class_names, device)
    st.success(f"Prediction: {label.title()} 🍣 (confidence: {confidence * 100:.1f}%)")
