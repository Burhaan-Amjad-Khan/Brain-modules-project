import torch
import torchvision.transforms as transforms
from PIL import Image

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model(path):
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model

def predict_image(model, image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_names = ["Tumor", "No Tumor"]  # Change as per your model
        return class_names[predicted.item()]
