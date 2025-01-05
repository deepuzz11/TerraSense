# TerraSense - Smart Terrain Awareness for Better Navigation

**TerraSense** uses deep learning to classify terrain types and estimate surface properties like roughness and slipperiness, enabling smarter navigation for autonomous systems.

**TerraSense** is a machine learning project designed for terrain classification using deep learning techniques. The model leverages a pre-trained MobileNetV2 architecture to classify images into three terrain categories : **Deserts**, **Forest Cover**, and **Mountains**. The project includes data preprocessing, model training, and evaluation workflows, all implemented in Python using the PyTorch framework.

## Table of Contents

- Project Description
- Installation
- Usage
- Training Details
- Results
- Model
- License

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

1.  git clone https://github.com/yourusername/TerraSense.gitcd TerraSense
2.  pip install -r requirements.txtThe requirements.txt file should include:torch==1.12.0torchvision==0.13.0matplotlib==3.4.3pillow==8.2.0

## Usage

1.  TerraSense/├── data/ ├── train/ │ ├── Deserts/ │ ├── Forest Cover/ │ └── Mountains/ └── validation/ ├── Deserts/ ├── Forest Cover/ └── Mountains/
2.  python main.pyThis will begin the training process on your local machine (CPU), print training and validation loss/accuracy per epoch, and save the final trained model as terrain_model.pth.
3.  **Testing**: After training, you can use the trained model to make predictions on new images (not provided in the project but can be added to the script).

## Training Details

- **Model Architecture**: MobileNetV2 pre-trained on ImageNet
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam optimizer with a learning rate of 0.001
- **Learning Rate Scheduler**: StepLR with a step size of 5 epochs and a gamma of 0.1
- **Epochs**: 10 epochs

### Example Training Output:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`  Epoch 1/10, Loss: 0.1975638451139358  Validation Accuracy: 92.13%  Epoch 2/10, Loss: 0.10591830498411912  Validation Accuracy: 83.76%  ...  Epoch 10/10, Loss: 0.005207964283975095  Validation Accuracy: 88.20%  `

## Results

The model achieves a validation accuracy of approximately **88%** after 10 epochs of training.

## Model

The trained model is saved as terrain_model.pth and can be loaded for inference as follows:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`  import torch  from model import TerrainClassifier  # Load the trained model  model = TerrainClassifier(num_classes=3)  model.load_state_dict(torch.load('terrain_model.pth'))  model.eval()  `

## License

This project is licensed under the MIT License - see the LICENSE file for details.
