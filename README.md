# Pancreas Segmentation using TransUNet

Automated pancreas segmentation from CT scans using TransUNet architecture. This project implements a production-ready deep learning pipeline for medical image segmentation, trained on the Medical Segmentation Decathlon (MSD) Task07 Pancreas dataset.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [License](#license)

## Overview

Pancreas segmentation is a challenging task in medical image analysis due to the organ's small size, variable shape, and poor contrast with surrounding tissues. This project addresses these challenges using TransUNet, a hybrid architecture that combines the local feature extraction capabilities of Convolutional Neural Networks with the global context modeling of Vision Transformers.

### Key Features

- Complete implementation of TransUNet architecture from scratch
- MONAI-based preprocessing pipeline with HU windowing and isotropic resampling
- Hybrid loss function combining Dice Loss and Cross-Entropy Loss
- 2D slice-based training strategy for memory efficiency
- 3D volume inference with slice-and-stack approach
- Attention map visualization for model interpretability

## Architecture

TransUNet consists of three main components working in sequence:

```
Input CT Slice (224 x 224)
         |
         v
+---------------------+
|    CNN Encoder      |    ResNet-style backbone
|   (Multi-scale)     |    Features at 1/4, 1/8, 1/16, 1/32 resolution
+----------+----------+
           |
           v
+---------------------+
|  Vision Transformer |    12 layers, 12 attention heads
|    Bottleneck       |    768-dim embeddings (base variant)
+----------+----------+
           |
           v
+---------------------+
|    CNN Decoder      |    U-Net style upsampling
|  (Skip Connections) |    Progressive resolution recovery
+----------+----------+
           |
           v
   Segmentation Map
   (2 classes: background, pancreas)
```

### Model Variants

| Variant | Embedding Dim | Heads | Layers | Parameters |
| ------- | ------------- | ----- | ------ | ---------- |
| Small   | 384           | 6     | 6      | 17M        |
| Base    | 768           | 12    | 12     | 105M       |
| Large   | 1024          | 16    | 24     | 300M       |

## Dataset

This project uses the Medical Segmentation Decathlon (MSD) Task07 Pancreas dataset.

| Attribute | Value                                  |
| --------- | -------------------------------------- |
| Volumes   | 420 (282 train, 139 test)              |
| Modality  | Portal venous phase CT                 |
| Source    | Memorial Sloan Kettering Cancer Center |
| Labels    | Pancreas and pancreatic tumor          |
| Format    | NIfTI compressed (.nii.gz)             |
| License   | CC-BY-SA 4.0                           |

The dataset is automatically downloaded when running the first notebook.

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ GPU memory for base variant

### Setup

```bash
# Clone the repository
git clone https://github.com/nghidanh2005/TransUNet-Pancreas-Segmentation.git
cd TransUNet-Pancreas-Segmentation

# Create virtual environment and install dependencies using UV
uv sync

# Activate the virtual environment
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### Dependencies

Core dependencies are managed via `pyproject.toml`:

- PyTorch 2.0+
- MONAI 1.3+
- NumPy
- Matplotlib
- nibabel
- einops
- timm
- tqdm
- scikit-learn

## Project Structure

```
TransUNet-Pancreas-Segmentation/
|
+-- 01_Data_Exploration_and_Processing.ipynb    Data download and preprocessing
+-- 02_Model_Architecture.ipynb                  TransUNet implementation
+-- 03_Training_Pipeline.ipynb                   Training loop and validation
+-- 04_Evaluation_and_Demo.ipynb                 Inference and visualization
|
+-- src/
|   +-- __init__.py
|   +-- model.py              TransUNet architecture (400+ lines)
|   +-- dataset.py            SlicingDataset for 2D extraction
|   +-- transforms.py         MONAI preprocessing pipeline
|   +-- loss.py               HybridLoss implementation
|   +-- utils.py              Visualization utilities
|
+-- data/
|   +-- Task07_Pancreas/      Downloaded dataset (auto-created)
|
+-- checkpoints/
|   +-- best_metric_model.pth Trained model weights
|
+-- outputs/
|   +-- data_splits.json      Train/val/test split indices
|   +-- evaluation_results.json
|
+-- pyproject.toml            UV project configuration
+-- README.md
+-- .gitignore
```

## Usage

### Step 1: Data Preparation

```bash
jupyter notebook 01_Data_Exploration_and_Processing.ipynb
```

This notebook downloads the MSD pancreas dataset, creates train/validation/test splits (80/10/10), and verifies the preprocessing pipeline.

### Step 2: Model Architecture

```bash
jupyter notebook 02_Model_Architecture.ipynb
```

Explore the TransUNet implementation with component-by-component explanation and forward pass verification.

### Step 3: Training

```bash
jupyter notebook 03_Training_Pipeline.ipynb
```

Train the model with configurable hyperparameters:

```python
CONFIG = {
    "batch_size": 8,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "model_variant": "small",  # or "base", "large"
}
```

### Step 4: Evaluation

```bash
jupyter notebook 04_Evaluation_and_Demo.ipynb
```

Run inference on test volumes and generate visualizations.

### Programmatic Usage

```python
from src.model import create_transunet
from src.transforms import get_val_transforms
import torch

# Create model
model = create_transunet(
    img_size=224,
    in_channels=1,
    out_channels=2,
    variant="small"
)

# Load trained weights
checkpoint = torch.load("checkpoints/best_metric_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
with torch.no_grad():
    input_slice = torch.randn(1, 1, 224, 224)
    output = model(input_slice)
    prediction = torch.argmax(output, dim=1)
```

## Results

### Preprocessing Pipeline

The MONAI-based preprocessing includes:

1. Orientation standardization to RAS (Right-Anterior-Superior)
2. Isotropic resampling to 1.0mm voxel spacing
3. HU intensity windowing: [-175, 250] mapped to [0, 1]
4. Foreground cropping to remove empty background

### Loss Function

The hybrid loss combines two complementary objectives:

```
Total Loss = 0.5 * Dice Loss + 0.5 * Cross-Entropy Loss
```

- Dice Loss: Handles extreme class imbalance (pancreas < 1% of volume)
- Cross-Entropy Loss: Provides stable per-pixel gradients

### Expected Performance

Based on TransUNet paper benchmarks for pancreas segmentation:

| Metric                   | Expected Range |
| ------------------------ | -------------- |
| Dice Score               | 0.75 - 0.85    |
| Hausdorff Distance (95%) | 5 - 15 mm      |

## References

1. Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L., and Zhou, Y. (2021). TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. arXiv:2102.04306

2. Simpson, A.L., Antonelli, M., Bakas, S., et al. (2019). A large annotated medical image dataset for the development and evaluation of segmentation algorithms. arXiv:1902.09063

3. Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.

4. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.

5. Medical Segmentation Decathlon: http://medicaldecathlon.com/

## License

This project is licensed under the MIT License. See the LICENSE file for details.

The MSD dataset is licensed under CC-BY-SA 4.0.

## Acknowledgments

- Medical Segmentation Decathlon organizers for the publicly available dataset
- MONAI team for the medical imaging framework
- TransUNet authors for the architecture design
