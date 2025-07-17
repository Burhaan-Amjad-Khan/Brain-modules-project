# model.py

import torch
import torch.nn as nn

# Define the model architecture (same as used while training)
class BrainTumorModel(nn.Module):
    def __init__(self):
        super(BrainTumorModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),  # adjust this if your image size is different
            nn.ReLU(),
            nn.Linear(128, 4)  # Assuming 4 classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def load_model(path):
    model = BrainTumorModel()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model
