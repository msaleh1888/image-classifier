"""dataloaders.py

Author  : Mahmoud Saleh
Created : 2025-10-06
Purpose : Builds and returns data loaders (train, valid, test) for the dataset.
"""

import torch

from torchvision import transforms, datasets

import json

def get_dataloaders(batch_size=32):
    """Build and return the project data loaders.
    Args:
        batch_size (int): Batch size for all loaders.
    Returns:
        tuple: (trainloader, validloader, testloader, class_to_idx, cat_to_name)
            - class_to_idx maps class folder name (e.g., '1') -> integer index.
    """
    data_dir = 'data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.Resize((256)),
                                                transforms.CenterCrop((224, 224)),
                                                transforms.RandomHorizontalFlip(p=0.2),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets_val = transforms.Compose([transforms.Resize((256)),
                                                transforms.CenterCrop((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets_test = transforms.Compose([transforms.Resize((256)),
                                                transforms.CenterCrop((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    dataset_train = datasets.ImageFolder(train_dir, transform=data_transforms_train)

    dataset_val = datasets.ImageFolder(valid_dir, transform=image_datasets_val)

    dataset_test = datasets.ImageFolder(test_dir, transform=image_datasets_test)

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)

    dataloaders_val = torch.utils.data.DataLoader(dataset_val, batch_size=64, shuffle=True)

    dataloaders_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True)

    class_to_idx = dataset_train.class_to_idx  # <-- folder label -> integer index

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    return dataloaders_train, dataloaders_val, dataloaders_test, class_to_idx, cat_to_name
