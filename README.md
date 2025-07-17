# 🧠 Brain Tumor Segmentation with U-Net++

This project performs medical image segmentation to detect brain tumor regions in MRI slices using a U-Net++ architecture built in PyTorch.

## 🚀 Features
- Trained on BraTS2020 dataset
- Uses U-Net++ model for accurate segmentation
- Dice + BCE Loss for performance
- Deployed with Streamlit for live web interaction

## 📁 Files

- `model.py`: Contains model architecture and loading function
- `app.py`: Streamlit frontend to upload MRI and display predictions
- `brain_model.pth`: Trained model weights
- `requirements.txt`: Dependencies


## ✅ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
