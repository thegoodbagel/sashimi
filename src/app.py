import streamlit as st
import torch
from streamlit_space import space
from torchvision import transforms
from PIL import Image
from fish_classifier import SushiClassifier
from predictor import predict
from sushi_guide import show_sushi_guide
from sushi_info import show_info_page

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load("./saved_models/best_model.pth", map_location=device)
sushi_classifier = SushiClassifier(species_list=checkpoint['label_list'])
sushi_classifier.load_state_dict(checkpoint['model_state_dict'])
sushi_classifier.to(device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Page layout
st.set_page_config(page_title="Sushi Classifier", layout="wide")
st.title("Sushi Classifier 🍣")
st.write("Upload an image and get a prediction!")

# Show guide
show_sushi_guide()

# Initialize session state vars
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False
if "input_image" not in st.session_state:
    st.session_state.input_image = None
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# 🔲 Section: Take a Photo
st.markdown("#### 📸 Take a Photo")
with st.container():
    if st.button("📷 Open Camera to Capture Image", key="open_camera_btn"):
        st.session_state.show_camera = True
        st.session_state.prediction_done = False  # reset prediction

    if st.session_state.show_camera:
        camera_image = st.camera_input("")
        if camera_image:
            st.session_state.input_image = Image.open(camera_image).convert("RGB")
            st.session_state.show_camera = False  # close camera after photo

        if st.button("❌ Cancel", key="cancel_camera_btn"):
            st.session_state.show_camera = False

# 🔲 Section: Upload a Photo
with st.container():
    st.markdown("#### 🖼️ Upload an Image")
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png"])
    if uploaded_file:
        st.session_state.input_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.prediction_done = False  # reset prediction

# Display image preview if available
if st.session_state.input_image:
    st.image(st.session_state.input_image, caption='Input Image', use_column_width=True)

    # Predict button
    if st.button("🔍 Predict Sushi"):
        label, confidence = predict(model, st.session_state.input_image, transform, device)
        st.session_state.prediction_result = (label, confidence)
        st.session_state.prediction_done = True

# Show prediction result if done
if st.session_state.prediction_done and st.session_state.prediction_result:
    label, confidence = st.session_state.prediction_result
    st.success(f"🍣 Prediction: **{label.title().upper()}** ({confidence * 100:.1f}% confidence)")
space(lines=4)
show_info_page()