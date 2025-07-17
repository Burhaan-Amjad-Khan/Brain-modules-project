import streamlit as st
from model import load_model, predict_image
from PIL import Image
import torch

# Load the trained model
model = load_model("brain_model.pth")

# Streamlit UI
st.set_page_config(page_title="Brain Surgery Disease Prediction", layout="centered")
st.title("🧠 Brain Module - Disease Prediction from MRI")

uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    if st.button("Predict"):
        prediction = predict_image(model, image)
        st.success(f"Predicted Disease: **{prediction}**")
