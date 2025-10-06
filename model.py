"""model.py

Author  : Mahmoud Saleh
Created : 2025-10-06
Purpose : Defines transfer-learning models (VGG16, ResNet18) with custom classifier heads.
"""

from torch import nn

from torchvision import models

from collections import OrderedDict

def get_model(model_arch, hidden_units):
    """
    Build a transfer learning model using a pretrained VGG16 or ResNet18 backbone.

    Loads a pretrained convolutional base (VGG16 or ResNet18) from torchvision,
    freezes its parameters, and replaces the classifier (fully connected layers)
    with a custom feed-forward network using ReLU activations and dropout.

    Args:
        model_arch (str): Name of the pretrained architecture to use.
            Typically 'vgg16' or 'resnet18'.
        hidden_units (int): Number of hidden units in the custom classifier layer.

    Returns:
        torch.nn.Module: A model ready for training or inference, with
            the new classifier attached and the pretrained base frozen.
    """
    if model_arch == "resnet":
        model = models.resnet18(pretrained=True)
        input_layer = 512
    else:
        model = models.vgg16(pretrained=True)
        input_layer = 25088

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    feedforwardclassifier = nn.Sequential(OrderedDict([
                                         ('fc1', nn.Linear(input_layer, hidden_units)),
                                         ('relu1', nn.ReLU()),
                                         ('dropout1', nn.Dropout(0.5)),
                                         ('output', nn.Linear(hidden_units, 102)),
                                         ('logsoftmax', nn.LogSoftmax(dim=1))]))
    
    if model_arch == "resnet":
        model.fc = feedforwardclassifier
    else:
        model.classifier = feedforwardclassifier

    return model
