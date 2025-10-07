# 🌼 Image Classifier Project

**Author:** Mahmoud Saleh  
**Created:** 2025-10-06  

---

## Overview
This project implements an image classification model using transfer learning with pretrained architectures (VGG16 or ResNet18).  
It predicts flower species from images by fine-tuning the final classifier layers of a pretrained CNN.  

The project includes modular scripts for:
- Loading and preprocessing datasets  
- Building and training the model  
- Saving and loading checkpoints  
- Running predictions on new images  
- A demo notebook for visual inference  

---

## Model Architecture
- **Backbones:** VGG16 or ResNet18 pretrained on ImageNet  
- **Classifier:** Fully connected feed‑forward network with ReLU activations and dropout  
- **Loss Function:** Negative Log Likelihood Loss (`nn.NLLLoss`)  
- **Optimizer:** Adam optimizer with configurable learning rate  
- **Training Device:** Supports both CPU and GPU (with `--gpu` flag)  

---

## Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/msaleh1888/image-classifier.git
cd image-classifier
```

### 2️⃣ Create and activate a conda environment
```bash
conda create -n imageclf python=3.10 -y
conda activate imageclf
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Train the Model
Train a new model on the flower dataset and save the checkpoint:
```bash
python train.py --model_arch vgg --hidden_units 512 --lr 0.0001 --epochs 5 --device cpu
```

### Predict an Image Class
Use a trained checkpoint to predict the top‑K classes for a new image:
```bash
python predict.py --image_path data/test/1/image_06743.jpg --checkpoint_path checkpoint.pth --topk 5 --device cpu
```

---

## Demo Notebook
The `demo_polished.ipynb` notebook provides a guided, visual demonstration of model inference.  
It walks through the following steps:

1. **Setup & Imports** – load libraries and helper functions.  
2. **Load Checkpoint** – rebuild the model and restore mappings.  
3. **Preprocess Image** – resize, crop, normalize, and convert to tensor.  
4. **Predict Top‑K Classes** – run inference on a test image.  
5. **Display Results** – show the image and top predictions with probabilities.

---

## Example Results
Example output after running a prediction:

```
Top 5 Predictions:
1. Sunflower – 94.1%
2. Black‑eyed Susan – 3.8%
3. Coreopsis – 1.1%
4. Dandelion – 0.6%
5. Buttercup – 0.4%
```

---

## Project Structure

```
Image_Classifier_Project/
├── data/                      # Training and validation datasets
├── cat_to_name.json           # Mapping from category labels to flower names
├── train.py                   # Training script
├── predict.py                 # Inference script
├── model.py                   # Model architecture definition
├── dataloaders.py             # Dataset loading utilities
├── utility_functions.py       # Helper functions for image processing and visualization
├── checkpoint.pth             # Example trained model checkpoint
├── demo.ipynb                 # Demo notebook
|
├── requirements.txt           # Dependencies list
├── LICENSE                    # MIT License
├── CHANGELOG.md               # Version history and release notes
└── README.md                  # Project documentation (this file)
```

---

## Requirements
Key dependencies (see `requirements.txt` for full list):

```
torch>=2.0
torchvision>=0.15
numpy>=1.21
matplotlib>=3.5
Pillow>=9.0
packaging>=23.0
```

---

## Changelog

See **[CHANGELOG.md](CHANGELOG.md)** for a detailed version history.

### Latest Version: `v1.0.0`
Released: *October 6, 2025*

#### Initial Release
- Added full project code and demo notebook  
- Implemented GPU support and Top-K visualization  
- Included MIT License and project documentation  
- Added versioning with `v1.0.0` tag

---

## License
This project is released under the **MIT License**.  
You are free to use, modify, and distribute it with proper attribution.  

---

> *Developed by Mahmoud Saleh — Flower Image Classifier Project (2025)*

---

## Acknowledgements

- **Udacity AI Programming with Python Nanodegree** — for providing the foundational structure of this project.  
- **Oxford Visual Geometry Group (VGG)** — for the [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).  
- **PyTorch** — for its flexible deep learning framework used throughout model training and inference.  
- **Matplotlib** — for visualization and Top-K probability plots.  
- **Community Tutorials & Open-Source Contributors** — for inspiration and practical implementation patterns.