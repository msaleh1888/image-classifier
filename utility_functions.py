"""utility_functions.py

Author  : Mahmoud Saleh
Created : 2025-10-06
Purpose : Contains image preprocessing and inference helper functions.
"""

import numpy as np

import torch

from torchvision.transforms import ToPILImage, ToTensor

from PIL import Image

def process_image(image):
    """
    Preprocess a PIL image for use in a PyTorch model.

    Performs resizing, center-cropping, scaling, and normalization using
    ImageNet mean and standard deviation. Converts the resulting image
    into a PyTorch tensor ready for inference or validation.

    Args:
        image (PIL.Image.Image): Input image to process.

    Returns:
        torch.Tensor: Normalized image tensor of shape (3, 224, 224)
        suitable for input to pretrained CNN architectures.
    """
    
    # Process a PIL image for use in a PyTorch model
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # Undo normalization
    image = image * std + mean
    image = ToPILImage()(torch.tensor(image))
    # Resize
    image = image.resize((256, int(image.height * 256 / image.width))) if image.width < image.height else \
          image.resize((int(image.width * 256 / image.height), 256))
    image.thumbnail((256, 256))
    
    # Center crop
    left = (image.width - 224) // 2
    top = (image.height - 224) // 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    # Divide by 255 and normalize
    np_image = np.array(image).transpose(2, 0, 1) / 255
    normalzied_np_image = (np_image - mean.numpy()) / std.numpy()
    return torch.tensor(normalzied_np_image, dtype=torch.float32)

def imshow(image, ax=None, title=None):
    """
    Display a PyTorch tensor as an image using Matplotlib.

    Converts a normalized tensor (with ImageNet mean and std) back to a
    displayable image by undoing preprocessing and transposing the color
    channel to the last dimension expected by Matplotlib.

    Args:
        image (torch.Tensor): The image tensor to display, typically of shape (3, H, W).
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to display the image on.
            If None, a new figure and axes are created.
        title (str, optional): Title to display above the image.

    Returns:
        matplotlib.axes.Axes: The Axes object containing the displayed image.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # But matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    """
    Predict the top-k classes of an image using a trained PyTorch model.

    Opens and preprocesses an input image, performs a forward pass through
    the trained network in evaluation mode, and returns the top-k predicted
    class indices along with their associated probabilities.

    Args:
        image_path (str or Path): Path to the image file for prediction.
        model (torch.nn.Module): Trained model used for inference.
        topk (int, optional): Number of top predictions to return. Default is 5.

    Returns:
        tuple:
            top_p (torch.Tensor): Probabilities of the top-k predicted classes.
            top_class (torch.Tensor): Corresponding class indices for the top-k predictions.

    Raises:
        FileNotFoundError: If the image file is not found at the specified path.
        IOError: If the file exists but cannot be opened as an image.
    """
    # Implement the code to predict the class from an image file
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print("Image file not found.")
    except IOError:
        print("File exists but is not a valid image.")
    
    transform = ToTensor()
    image_tensor = transform(image)
    processed_image = process_image(image_tensor).unsqueeze(0)
    processed_image = processed_image.to(next(model.parameters()).device)

    model.eval()
    with torch.no_grad():
        logps = model.forward(processed_image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
    return top_p, top_class
    