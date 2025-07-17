import torch
import torch.nn as nn
import torchvision.models as models

CLASS_NAMES = ['glioma', 'meningioma', 'no tumor', 'pituitary']

def load_model(path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))  # 4 classes

    # Load weights
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model
