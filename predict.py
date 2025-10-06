# =============================================================================
# Predict.py
# -----------------------------------------------------------------------------
# Author  : Mahmoud Saleh
# Created : 2025-10-06
# Purpose : Inference script for predicting the class of a single image.
# Loads checkpoint, rebuilds the model, and prints top-K predictions.
# =============================================================================

import numpy as np

import torch

import matplotlib.pyplot as plt

from utility_functions import process_image, imshow, predict
from train import load_checkpoint

import argparse

parser  = argparse.ArgumentParser(description="Predict an image class")
parser.add_argument("--image_path", type=str, default="flowers/test/1/image_06743.jpg", help="Choose image to predict")
parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth", help="Choose model file to load")
parser.add_argument("--topk", type=int, default=5, help="Top K probabilities to show")
parser.add_argument("--device", type=str, default="cuda", help="Train on GPU or CPU")
args = parser.parse_args()

model, class_to_idx, cat_to_name = load_checkpoint(args.checkpoint_path)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
model.to(device)
top_p, top_idx = predict(args.image_path, model, args.topk)

top_idx_list = top_idx[0].tolist()
idx_to_class = {v: k for k, v in class_to_idx.items()}  # reverse
top_labels = [idx_to_class[int(i)] for i in top_idx_list]       # labels like '14','61',...

top_names = [cat_to_name.get(lbl, lbl) for lbl in top_labels]
top_classes = [cat_to_name[str(i)] for i in top_idx_list]
top_probs = top_p[0]

print("The most likely class is : {}".format(top_classes[0]))
print("The associated probability is: {}".format(top_probs[0]))

print("The top {} classes are: {}".format(args.topk, top_classes))
print("The associated probabilities are: {}".format(top_probs))
