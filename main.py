import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt
from torchvision.models import ResNet50_Weights

# Check for CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Define your model class
class TerrainClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(TerrainClassifier, self).__init__()
        # Using ResNet50 with pre-trained weights for better performance
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replacing the final fully connected layer with one suited for the number of classes
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # Regularization
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Define the image transformations (data augmentation and normalization)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a larger image for better feature extraction
    transforms.RandomHorizontalFlip(),  # Augmentation: Randomly flip the image
    transforms.RandomRotation(20),  # Augmentation: Randomly rotate the image
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color augmentation
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Random zoom
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset paths
train_dir = r'C:/Users/deepi/Desktop/TerraSense/data/train'
val_dir = r'C:/Users/deepi/Desktop/TerraSense/data/validation'

# Check if directories exist
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    print("Warning: One or more data directories do not exist.")
    exit()

# Load the datasets using ImageFolder
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Initialize the model and move it to the CPU
model = TerrainClassifier(num_classes=3).to(device)

# Unfreeze the final layers for training
for param in model.model.fc.parameters():
    param.requires_grad = True

# Define the optimizer, loss function, and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Use a learning rate scheduler to adjust the learning rate dynamically
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Initialize lists for plotting
train_losses = []
val_accuracies = []

# Training loop
if __name__ == '__main__':
    num_epochs = 10  # Training for 10 epochs
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_train_loss:.4f}")

        # Validation loop
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_val_accuracy = 100 * correct / total
            val_accuracies.append(epoch_val_accuracy)

            print(f"Validation Accuracy: {epoch_val_accuracy:.2f}%")

    # Plot the results after training
    plt.figure(figsize=(10, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy over Epochs')

    plt.tight_layout()
    # Save the plot as an image
    plt.savefig('training_results.png')

    # Save the model after training
    torch.save(model.state_dict(), 'terrain_model.pth')
    print("Model saved as terrain_model.pth")
