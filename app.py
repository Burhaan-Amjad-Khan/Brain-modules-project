import streamlit as st
import numpy as np
import torch
import cv2
from model import load_model
from PIL import Image

st.set_page_config(page_title="Brain Tumor Segmentation", layout="centered")

st.title("🧠 Brain Tumor Segmentation with U-Net++")
st.markdown("Upload an MRI brain scan image to segment tumor regions.")

# Load model
model = load_model("brain_model.pth")

# File uploader
uploaded_file = st.file_uploader("Upload MRI scan image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        mask = (output.squeeze().numpy() > 0.5).astype(np.uint8) * 255

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(mask, caption="Predicted Mask", use_column_width=True)
