import torch
from torch.utils.data import DataLoader
from utils.dataset_loader import TerrainDataset
from utils.image_transformations import get_transform
from models.terrain_classifier import TerrainClassifier
from torch import nn, optim

# Paths
DATA_DIR = 'data/'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# Dataset and DataLoader
transform = get_transform()
train_dataset = TerrainDataset(DATA_DIR, transform, mode='train')
test_dataset = TerrainDataset(DATA_DIR, transform, mode='test')
validation_dataset = TerrainDataset(DATA_DIR, transform, mode='validation')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TerrainClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# Evaluation
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

train_acc = evaluate(train_loader)
val_acc = evaluate(validation_loader)
test_acc = evaluate(test_loader)

print(f"Train Accuracy: {train_acc:.2%}")
print(f"Validation Accuracy: {val_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")
