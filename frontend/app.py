import streamlit as st

st.title("Sashimi Classifier 🍣")
st.write("Upload an image and get a prediction!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("Running inference...")
    # TODO: Load the model and run prediction
    st.write("Prediction: Tuna 🐟 (confidence: 91.3%)")
