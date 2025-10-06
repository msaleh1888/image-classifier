# =============================================================================
# Train.py
# -----------------------------------------------------------------------------
# Author  : Mahmoud Saleh
# Created : 2025-10-06
# Purpose : Main training script for the flower image classifier.
# Loads datasets, builds the model, trains it, and saves checkpoints.
# =============================================================================

import torch
from torch import nn, optim

import numpy as np

import time

from dataloaders import get_dataloaders
from model import get_model

import argparse

def save_model(model, model_arch, hidden_units, class_to_idx, cat_to_name):
    """
    Save a trained model checkpoint to disk.

    Stores the model's architecture name, classifier configuration, class mappings,
    and learned weights (state_dict) into a dictionary, then serializes it to
    'checkpoint.pth' using torch.save().

    Args:
        model (torch.nn.Module): The trained model to save.
        model_arch (str): Name of the pretrained architecture used (e.g., 'vgg16', 'resnet18').
        hidden_units (int): Number of hidden units in the classifier layer.
        class_to_idx (dict): Mapping of class labels to indices used during training.
        cat_to_name (dict): Mapping of class labels to human-readable category names.

    Returns:
        None
    """
    checkpoint = {
    'model_arch': model_arch,
    'hidden_units': hidden_units,
    'class_to_idx' : class_to_idx,
    'cat_to_name': cat_to_name,
    'model_state_dict': model.state_dict()
    }
    torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath):
    """
    Load a saved model checkpoint from disk and rebuild the model.

    Restores the model architecture, classifier configuration, and trained
    weights (state_dict) from a checkpoint file. Also returns the stored
    class-to-index and category-name mappings for inference.

    Args:
        filepath (str or Path): Path to the checkpoint file (e.g., 'checkpoint.pth').

    Returns:
        tuple:
            model (torch.nn.Module): Reconstructed model with loaded weights.
            class_to_idx (dict): Mapping of class labels to indices.
            cat_to_name (dict): Mapping of class labels to human-readable category names.
    """
    checkpoint = torch.load(filepath)
    model = get_model(checkpoint['model_arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['class_to_idx'], checkpoint['cat_to_name']

if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description="Train VGG or ResNet18 model with custom settings")
    parser.add_argument("--model_arch", type=str, default="vgg", help="Model architecture to use")
    parser.add_argument("--hidden_units", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Train on GPU or CPU")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(torch.cuda.is_available()) # Should return True when GPU is enabled. 
    print(device)

    dataloaders_train, dataloaders_val, dataloaders_test, class_to_idx, cat_to_name = get_dataloaders()
    model = get_model(args.model_arch, args.hidden_units)

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    if args.model_arch == "resnet":
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    model.to(device)

    epochs = args.epochs

    for e in range(epochs):
        start = time.time()
        # Training loop
        running_loss = 0
        model.train()
        for images, labels in dataloaders_train:
            # Move images and label tensors to the GPU
            images = images.to(device)
            labels = labels.to(device)

            logps = model.forward(images)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        # Validation Loop
        validation_loss = 0
        accuracy = 0
        model.eval()

        for images, labels in dataloaders_val:
            with torch.no_grad():
                
                images = images.to(device)
                labels = labels.to(device)
                
                logps = model.forward(images)
                loss = criterion(logps, labels)
                
                validation_loss += loss.item()
                
                # Get the class probabilities
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        end = time.time()
        print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders_train)),
            "Validation Loss: {:.3f}.. ".format(validation_loss/len(dataloaders_val)),
            "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders_val)))
        print(f"Epoch time: {(end - start) / 60:.2f} minutes")

    # Do validation on the test set
    test_loss = 0
    accuracy = 0
    model.eval()
    for images, labels in dataloaders_test:
        with torch.no_grad():
            
            images = images.to(device)
            labels = labels.to(device)
                
            logps = model.forward(images)
            loss = criterion(logps, labels)
                
            test_loss += loss.item()
                
            # Get the class probabilities
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            
        print("Test Loss: {:.3f}.. ".format(validation_loss/len(dataloaders_val)),
            "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders_val)))
        
    # Save the checkpoint 
    save_model(model, args.model_arch, args.hidden_units, class_to_idx, cat_to_name)
