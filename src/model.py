"""
Defines the neural network architecture used for ISL alphabet recognition.

Responsibilities:
- Load the pretrained ResNet-18
- Replace the final classification layer
- Return a model configured for 36 output classes
"""

from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

def create_model():
    NUM_CLASSES = 36

    model = resnet18(
        weights = ResNet18_Weights.DEFAULT
        )
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=NUM_CLASSES
    )
    for parameter in model.parameters():
        parameter.requires_grad = False #Freezing the pretrained parameters

    for parameter in model.fc.parameters():
        parameter.requires_grad = True #Letting only the newly created layer train

    return model