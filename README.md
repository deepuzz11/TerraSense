# TerraSense - Smart Terrain Awareness for Better Navigation

**TerraSense** uses deep learning to classify terrain types and estimate surface properties like roughness and slipperiness, enabling smarter navigation for autonomous systems.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
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

```git clone https://github.com/yourusername/TerraSense.git
 cd TerraSense
```

2. Instal the necessary dependencies :

```bash pip install -r requirements.txt

```

## Usage

1. Prepare your data :
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

2. Training: To start training the model, run: ` python main.py`
   This will begin the training process on your local machine (CPU), print training and validation loss/accuracy per epoch, and save the final trained model as terrain_model.pth.
3. **Testing**: After training, you can use the trained model to make predictions on new images (not provided in the project but can be added to the script).

## Training Details

- **Model Architecture**: MobileNetV2 pre-trained on ImageNet
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam optimizer with a learning rate of 0.001
- **Learning Rate Scheduler**: StepLR with a step size of 5 epochs and a gamma of 0.1
- **Epochs**: 10 epochs

### Example Training Output:

`Epoch 1/10, Loss: 0.1975638451139358  Validation Accuracy: 92.13%  Epoch 2/10, Loss: 0.10591830498411912  Validation Accuracy: 83.76%  ...  Epoch 10/10, Loss: 0.005207964283975095  Validation Accuracy: 88.20%  `

## Results

The model achieves a validation accuracy of approximately **88%** after 10 epochs of training.

## Model

The trained model is saved as **terrain_model.pth** and can be loaded for inference as follows:

````import torch
 from model import TerrainClassifier  # Load the trained model  model = TerrainClassifier(num_classes=3)  model.load_state_dict(torch.load('terrain_model.pth'))  model.eval()  ```

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

````
