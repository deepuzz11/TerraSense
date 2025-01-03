import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class TerrainClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(TerrainClassifier, self).__init__()
        # Use the ResNet18 model with pre-trained weights from ImageNet
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the fully connected layer (fc) to match the number of output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
