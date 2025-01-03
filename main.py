import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# Check for CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Define your model class
class TerrainClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(TerrainClassifier, self).__init__()
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)  # MobileNetV2 for faster processing
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to smaller images for faster processing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset paths - ensure these paths match the structure of your dataset
train_dir = r'C:/Users/deepi/Desktop/TerraSense/data/train'  # Raw string literals to avoid escape errors
val_dir = r'C:/Users/deepi/Desktop/TerraSense/data/validation'  # Raw string literals

# Check if directories exist
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    print("Warning: One or more data directories do not exist.")
    exit()

# Load the datasets using the correct folder names (Deserts, Forest Cover, Mountains)
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# DataLoader for training and validation with optimizations
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Initialize the model and move it to the CPU
model = TerrainClassifier(num_classes=3).to(device)

# Define the optimizer, loss function, and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop inside the if __name__ == '__main__' block
if __name__ == '__main__':
    # Training loop
    num_epochs = 10  # Reduce epochs for faster testing
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        # Validation loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Validation Accuracy: {100 * correct / total}%")

    # Save the model after training
    torch.save(model.state_dict(), 'terrain_model.pth')
    print("Model saved as terrain_model.pth")
