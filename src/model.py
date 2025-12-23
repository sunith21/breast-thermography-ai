import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=2):
    """
    Returns a ResNet18 model for binary classification
    """

    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all convolutional layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# Quick test
if __name__ == "__main__":
    model = get_model()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
