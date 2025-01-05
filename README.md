# TerraSense - Smart Terrain Awareness for Better Navigation

**TerraSense** uses deep learning to classify terrain types and estimate surface properties like roughness and slipperiness, enabling smarter navigation for autonomous systems.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
- [Training Graphs](#training-graphs)
- [Model](#model)
- [License](#license)

## Project Description

The goal of this project is to develop a model that can classify terrain types based on images. The dataset contains images of **Deserts**, **Forest Cover**, and **Mountains**, and the model is trained using transfer learning with the MobileNetV2 architecture.

**Key features**:

- Pre-trained MobileNetV2 model used for transfer learning
- Image transformations for resizing and normalization
- Training and evaluation on CPU

## Installation

### Prerequisites

- Python 3.11 or later
- PyTorch (CPU version, since GPU is not used in this project)
- torchvision
- matplotlib (for visualization)
- PIL (for image processing)

### Setup

1. Clone the repository :

    `git clone https://github.com/yourusername/TerraSense.git`

    `cd TerraSense`

2. Instal the necessary dependencies :

    ``` pip install -r requirements.txt```

## Usage

1. Prepare your data :
    ```
       TerraSense/
       ├── data/
       ├── train/
       │ ├── Deserts/
       │ ├── Forest Cover/
       │ └── Mountains/
       └── validation/
       ├── Deserts/
       ├── Forest Cover/
       └── Mountains/
    ```
2. **Training**: To start training the model, run: ` python main.py` 
   This will begin the training process on your local machine (CPU), print training and validation loss/accuracy per epoch, and save the final trained model as **terrain_model.pth**.
3. **Testing**: After training, you can use the trained model to make predictions on new images .

## Training Details

- **Model Architecture**: ResNet50 pre-trained on ImageNet (using ResNet50_Weights.IMAGENET1K_V1)
- **Loss Function**: CrossEntropyLoss
- **Optimizer** : AdamW optimizer with a learning rate of 0.001
- **Learning Rate Scheduler**: StepLR with a step size of 5 epochs and a gamma of 0.1
- **Epochs**: 10 epochs
- **Data Augmentation**: Includes random horizontal flip, random rotation, color jitter, and random resized crop.

### Example Training Output:
```
Epoch 1/10, Loss: 0.6535  Validation Accuracy: 88.03%
Epoch 2/10, Loss: 0.2825  Validation Accuracy: 90.26%
Epoch 3/10, Loss: 0.2283  Validation Accuracy: 91.11%
Epoch 4/10, Loss: 0.1903  Validation Accuracy: 92.48%
Epoch 5/10, Loss: 0.1741  Validation Accuracy: 92.82%
Epoch 6/10, Loss: 0.1452  Validation Accuracy: 92.48%
Epoch 7/10, Loss: 0.1426  Validation Accuracy: 91.97%
Epoch 8/10, Loss: 0.1256  Validation Accuracy: 91.62%
Epoch 9/10, Loss: 0.1353  Validation Accuracy: 92.99%
Epoch 10/10, Loss: 0.1465  Validation Accuracy: 92.65%
```

## Results

The model achieves a validation accuracy of approximately **92.65%** after 10 epochs of training.

## Training Graphs
After the training is completed, two graphs will be generated to help visualize the model's performance:

### 1. Training Loss over Epochs:

This graph shows how the training loss decreases as the model trains over the epochs. A decreasing loss indicates that the model is improving its performance on the training data.

### 2. Validation Accuracy over Epochs:

This graph tracks the validation accuracy, showing how well the model performs on the unseen validation data at each epoch. A steady increase in validation accuracy suggests that the model is generalizing well to new data.

![image](https://github.com/user-attachments/assets/6c941443-8d71-4b31-9824-f67613df47d4)

**The graphs are saved as training_results.png and will show the trends of training loss and validation accuracy throughout the 10 epochs of training.**

## Model

The trained model is saved as **terrain_model.pth** and can be loaded for inference as follows:
```
import torch
from model import TerrainClassifier  # Load the trained model 
model = TerrainClassifier(num_classes=3) 
model.load_state_dict(torch.load('terrain_model.pth'))
model.eval() 
```

License
-------

This project is licensed under the MIT License - see the [LICENSE] file for details.
