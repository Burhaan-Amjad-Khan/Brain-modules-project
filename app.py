# app.py

import streamlit as st
from model import load_model
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

model = load_model("brain_model.pth")

CLASS_NAMES = ['glioma', 'meningioma', 'no tumor', 'pituitary']

def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0)  # add batch dimension
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted[0].item()]

st.title("Brain Tumor Classification")

uploaded_file = st.file_uploader("Upload Brain MRI Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label = predict(image)
        st.success(f"Predicted: {label}")
