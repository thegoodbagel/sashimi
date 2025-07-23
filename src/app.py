import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from sushi_classifier import SushiClassifier, predict

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ["Otoro (Fatty Bluefin Tuna)", "Salmon"]

# Load model
model = SushiClassifier(num_classes=2)
model.load_state_dict(torch.load("../saved_models/best_model.pth", map_location=device))
model.to(device)

# Image transform (should match your training transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.title("Sashimi Classifier 🍣")
st.write("Upload an image and get a prediction!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Running inference...")

    label, confidence = predict(model, image, transform, class_names, device)
    st.write(f"Prediction: {label.title()} 🍣 (confidence: {confidence * 100:.1f}%)")
