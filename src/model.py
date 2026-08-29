"""
Defines the neural network architecture used for ISL alphabet recognition.

Responsibilities:
- Load a pretrained ResNet-18 backbone
- Replace the final classification layer
- Fine-tune the last residual block (layer4)
- Return a model configured for the required number of output classes
"""

from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

from src.dataset import classes


def create_model():
    NUM_CLASSES = len(classes)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    # Freeze all pretrained parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune the last residual block
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    # Train the new classifier
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model