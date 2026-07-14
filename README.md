# Car Body Type Classifier

A fine-tuned ResNet18 model that classifies car images into 5 body types: **Convertible, Coupe, Hatchback, SUV, and Sedan**.

🚗 **Live Demo:** [Car Body Type Classifier on Hugging Face Spaces]
undermaintainance
---

## Project Overview

This project was built as a complete end-to-end deep learning pipeline — from dataset acquisition and preprocessing, through model training and evaluation, to deployment on Hugging Face Spaces. The goal was to classify car images by body type using transfer learning with a pretrained ResNet18 backbone.

---

## Dataset

- **Source:** [Cars Body Type Cropped](https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped) (Kaggle)
- **Total images:** ~4,000 across 5 classes (after removing Pick-Up and VAN from the original 7-class dataset)
- **Pre-split:** train / valid / test provided by the dataset
- **Class distribution (train set):**

| Class       | Images |
|-------------|--------|
| Convertible | 811    |
| Coupe       | 721    |
| Hatchback   | 701    |
| SUV         | 847    |
| Sedan       | 729    |

Class imbalance was verified to be mild (max/min ratio ~1.2), ruling it out as a significant contributor to model errors.

---

## Model Architecture

- **Base model:** ResNet18 pretrained on ImageNet (`IMAGENET1K_V1`)
- **Strategy:** Frozen backbone (feature extractor) + trainable classification head
- **Modification:** Final fully connected layer replaced — `nn.Linear(512, 1000)` → `nn.Linear(512, 5)`
- **Rationale for freezing:** The pretrained backbone already encodes rich general visual features (edges, textures, shapes). For a dataset of ~4,000 images, fine-tuning only the classification head prevents overfitting while leveraging ImageNet's learned representations.

---

## Training Details

| Parameter       | Value                        |
|-----------------|------------------------------|
| Optimizer       | Adam                         |
| Learning rate   | 0.001                        |
| Loss function   | CrossEntropyLoss             |
| Batch size      | 32                           |
| Epochs          | 8                            |
| Input size      | 224×224 (ImageNet standard)  |
| Normalization   | ImageNet mean/std            |
| Hardware        | Google Colab T4 GPU          |

**Transforms pipeline:**
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## Results

**Validation Accuracy: ~75%**

### Per-Class Performance

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Convertible | 0.99      | 0.75   | 0.85     |
| Coupe       | 0.44      | 0.72   | 0.54     |
| Hatchback   | 0.67      | 0.71   | 0.69     |
| SUV         | 0.84      | 0.85   | 0.85     |
| Sedan       | 0.71      | 0.67   | 0.69     |

### Confusion Matrix Analysis

The primary confusion pairs were:
- **Hatchback ↔ SUV** (50 total misclassifications) — compact SUVs and tall hatchbacks share similar side-profile silhouettes, with ground clearance and roofline height being subtle cues the frozen backbone struggled to distinguish.
- **Coupe ↔ Sedan** (44 total misclassifications) — structurally similar body styles differing mainly in rear-end design details (presence/absence of a separate trunk lid seam), which are visually subtle features.

---

## Why ~75% Accuracy?

The model performed well on visually distinct classes (Convertible: 99% precision, SUV: 84% precision) but struggled at two specific visual boundaries — Coupe/Sedan and Hatchback/SUV.

**Root cause: frozen backbone using generic ImageNet features.**

The frozen ResNet18 backbone was never adapted specifically for car body-type discrimination. Its pretrained features (trained on dogs, furniture, everyday objects) are rich enough to separate visually distinct classes (open-roof Convertible vs. boxy SUV), but lack the fine-grained discriminative power needed to cleanly separate classes that differ in subtle structural details.

**This was confirmed by ruling out alternative explanations:**
- Class imbalance was checked and found to be mild (ratio ~1.2) — not a significant factor
- Coupe had comparable training data to other classes (721 images) yet showed the lowest F1-score — confirming the issue is visual ambiguity, not data scarcity

**What would improve accuracy:**
- Unfreezing the last 1-2 convolutional blocks of ResNet18 and fine-tuning at a lower learning rate (e.g. 1e-4) would allow the backbone to develop car-specific features
- Data augmentation (random crops, flips, color jitter) to improve generalization
- A larger, more diverse dataset — particularly for Coupe, which overlaps visually with both Sedan and Convertible in many angles

---

## Project Structure

```
car-body-type-classifier/
├── car_classifier.ipynb          # Full training pipeline notebook
└── car_classifier_resnet18.pth   # Trained model weights
```

---

## How to Run Locally

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load model
resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(512, 5)
resnet.load_state_dict(torch.load('car_classifier_resnet18.pth', map_location='cpu'))
resnet.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Convertible', 'Coupe', 'Hatchback', 'SUV', 'Sedan']

# Predict
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = resnet(img_tensor)
        probs = F.softmax(output, dim=1)[0]
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}
```

---

## Tech Stack

- **PyTorch** — model building, training loop, inference
- **torchvision** — pretrained ResNet18, ImageFolder, transforms
- **scikit-learn** — confusion matrix, classification report
- **Gradio** — interactive web demo
- **Hugging Face Spaces** — deployment
- **Google Colab** — training environment (T4 GPU)

---



*Built as an end-to-end CNN project focusing on transfer learning, fine-tuning, and model deployment.*
