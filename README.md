# TransUNet: Pancreas Segmentation from CT Scans

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[📄 Paper](https://arxiv.org/abs/2102.04306) | [🤗 Dataset](http://medicaldecathlon.com/) | [📚 Notebooks](./notebooks/) | [📖 Docs](./docs/)

---

TransUNet is a hybrid deep learning architecture combining **CNN encoders** with **Vision Transformers** for medical image segmentation. This implementation focuses on automated pancreas segmentation from CT scans using the Medical Segmentation Decathlon dataset.

![TransUNet Architecture](assets/architecture_diagram.png)

## Overview

Pancreas segmentation from CT imaging is challenging due to:
- **Small organ size** (<1% of scan volume)
- **Variable shape** across patients
- **Low contrast** with surrounding soft tissue

TransUNet addresses these challenges through:
- ✅ Multi-scale CNN feature extraction
- ✅ Global context modeling via transformer self-attention  
- ✅ U-Net decoder with skip connections for precise localization
- ✅ Hybrid loss (Dice + Cross-Entropy) for class imbalance

### Key Features

- 🏗️ **Complete TransUNet implementation** from scratch (533 lines)
- 🔬 **MONAI preprocessing pipeline** with HU windowing & isotropic resampling
- ⚡ **2D slice training** for memory efficiency on consumer GPUs
- 📊 **3D volume inference** with slice-and-stack aggregation
- 🎯 **Hybrid loss function** combining Dice & Cross-Entropy
- 🔍 **Attention visualization** for model interpretability

---

## Architecture

TransUNet pipeline consists of three components:

```
Input CT (224×224) → CNN Encoder → Transformer → CNN Decoder → Segmentation Map
                         ↓             ↓              ↑
                    Multi-scale    Global        Skip
                    Features      Context     Connections
```

### Model Variants

| Variant | Embed Dim | Heads | Layers | Parameters | Memory |
|---------|-----------|-------|--------|-----------|---------|
| Small   | 384       | 6     | 6      | 17M       | ~4GB    |
| Base    | 768       | 12    | 12     | 105M      | ~12GB   |
| Large   | 1024      | 16    | 24     | 300M      | ~24GB   |

---

## Dataset

**Medical Segmentation Decathlon - Task07 Pancreas**

| Attribute | Details |
|-----------|---------|
| **Volumes** | 420 CT scans (282 train, 139 test) |
| **Modality** | Portal venous phase CT |
| **Labels** | Background (0), Pancreas (1), Tumor (2) |
| **Format** | NIfTI compressed (`.nii.gz`) |
| **Size** | ~11.4GB (compressed) |
| **Source** | Memorial Sloan Kettering Cancer Center |
| **License** | [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

Dataset is auto-downloaded via `monai.apps.download_and_extract` in Notebook 01.

---

## Installation

### Prerequisites

**Hardware Requirements:**
- CUDA-capable GPU (recommended for training)
- 8GB+ VRAM for base variant, 4GB for small variant
- 16GB+ system RAM

**Software Requirements:**
- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/ihatesea69/TransUNet-Pancreas-Segmentation.git
cd TransUNet-Pancreas-Segmentation

# Create virtual environment and install dependencies
uv sync

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
```

**Dependencies** (managed via `pyproject.toml`):
- `torch>=2.0.0` - Deep learning framework
- `monai>=1.3.0` - Medical imaging toolkit
- `nibabel>=5.0.0` - NIfTI file I/O
- `einops>=0.8.0` - Tensor operations
- `matplotlib`, `numpy`, `scikit-learn` - Scientific computing

### Quick Start

```bash
# Launch Jupyter for interactive exploration
jupyter notebook notebooks/01_Data_Exploration_and_Processing.ipynb

# Or use CLI for training/inference
python main.py train --variant small --epochs 50
python main.py inference --checkpoint model.pth --input scan.nii.gz
```

---

## Project Structure

```
TransUNet-Pancreas-Segmentation/
│
├── 📦 src/                   # Core source code
│   ├── model.py              # TransUNet architecture (533 lines)
│   ├── dataset.py            # SlicingDataset for 2D extraction
│   ├── transforms.py         # MONAI preprocessing pipeline
│   ├── loss.py               # HybridLoss (Dice + CrossEntropy)
│   └── utils.py              # Visualization utilities
│
├── 📓 notebooks/             # Interactive Jupyter notebooks
│   ├── 01_Data_Exploration_and_Processing.ipynb
│   ├── 02_Model_Architecture.ipynb
│   ├── 03_Training_Pipeline.ipynb
│   └── 04_Evaluation_and_Demo.ipynb
│
├── 📄 paper/                 # Academic paper (LaTeX)
│   ├── paper.tex             # Main document
│   ├── references.bib        # Bibliography
│   └── paper.pdf             # Compiled PDF
│
├── 📖 docs/                  # Documentation
│   ├── QUICKSTART.md         # Quick setup guide
│   ├── CHANGELOG.md          # Version history
│   └── DEPLOY.md             # GitHub Pages deployment
│
├── 🔧 scripts/               # Utility scripts
│   ├── compile.ps1           # LaTeX compilation
│   └── clean.ps1             # Project cleanup
│
├── 📁 data/                  # Dataset (~11.4GB, gitignored)
├── 💾 checkpoints/           # Model weights (gitignored)
├── 📊 outputs/               # Results & metadata
├── 🖼️ assets/                # Images & diagrams
│
├── main.py                   # CLI entry point
├── pyproject.toml            # UV project config
├── index.html                # GitHub Pages landing page
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## Usage

### Interactive Notebooks (Recommended)

The project is organized into 4 self-contained Jupyter notebooks:

#### 📓 Notebook 01: Data Exploration & Processing
```bash
jupyter notebook notebooks/01_Data_Exploration_and_Processing.ipynb
```
- Downloads MSD Task07 Pancreas dataset (~11.4GB)
- Creates 80/10/10 train/val/test splits
- Defines MONAI preprocessing pipeline
- Visualizes CT volumes with segmentation masks

#### 📓 Notebook 02: Model Architecture
```bash
jupyter notebook notebooks/02_Model_Architecture.ipynb
```
- TransUNet component-by-component explanation
- Multi-head attention visualization
- Forward pass verification
- Parameter count analysis

#### 📓 Notebook 03: Training Pipeline
```bash
jupyter notebook notebooks/03_Training_Pipeline.ipynb
```
- SlicingDataset for 2D slice extraction
- Hybrid loss (Dice + CrossEntropy)
- Training loop with validation
- Model checkpointing

**Training Configuration:**
```python
CONFIG = {
    "model_variant": "small",     # or "base", "large"
    "batch_size": 8,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "img_size": 224,
}
```

#### 📓 Notebook 04: Evaluation & Demo
```bash
jupyter notebook notebooks/04_Evaluation_and_Demo.ipynb
```
- Load trained checkpoints
- 3D volume inference (slice-by-slice)
- Dice score & Hausdorff distance metrics
- Visualization with mask overlays

---

### CLI Interface

For scripted workflows, use the CLI:

```bash
# Training
python main.py train \
  --variant small \
  --batch-size 8 \
  --epochs 50 \
  --lr 1e-4

# Inference
python main.py inference \
  --checkpoint checkpoints/best_model.pth \
  --input data/test_scan.nii.gz \
  --output outputs/prediction.nii.gz
```

**Options:**
- `--variant`: Model size (`small` | `base` | `large`)
- `--batch-size`: Training batch size (default: 8)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)

---

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
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference on 2D slice
with torch.no_grad():
    input_slice = torch.randn(1, 1, 224, 224)
    output = model(input_slice)
    prediction = torch.argmax(output, dim=1)
```

---

## Preprocessing Pipeline

MONAI-based preprocessing ensures consistent data formatting:

| Step | Transform | Purpose |
|------|-----------|---------|
| 1 | `LoadImaged` | Load NIfTI files |
| 2 | `EnsureChannelFirstd` | Add channel dimension |
| 3 | `Orientationd` | Standardize to RAS orientation |
| 4 | `Spacingd` | Resample to 1.0mm isotropic |
| 5 | `ScaleIntensityRanged` | HU windowing [-175, 250] → [0, 1] |
| 6 | `CropForegroundd` | Remove empty background |

**HU Windowing:** Focuses on soft tissue (pancreas, liver, kidneys) while suppressing bone and air.

---

## Loss Function

**Hybrid Loss** combines complementary objectives:

```python
Total Loss = 0.5 × Dice Loss + 0.5 × Cross-Entropy Loss
```

- **Dice Loss:** Handles extreme class imbalance (pancreas < 1% of volume)
- **Cross-Entropy Loss:** Provides stable per-pixel gradients

---

## Expected Performance

Based on TransUNet paper benchmarks:

| Metric | Expected Range |
|--------|----------------|
| **Dice Score** | 0.75 - 0.85 |
| **Hausdorff Distance (95%)** | 5 - 15 mm |
| **Inference Time** | ~2-3 sec/slice (GPU) |

*Performance varies with model variant and GPU hardware.*

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

The MSD Task07 Pancreas dataset is licensed under **CC-BY-SA 4.0**.

---

## Citation

If you use TransUNet in your research, please cite the original paper:

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
```

For the MSD dataset:

```bibtex
@article{simpson2019large,
  title={A large annotated medical image dataset for the development and evaluation of segmentation algorithms},
  author={Simpson, Amber L and Antonelli, Michela and Bakas, Spyridon and others},
  journal={arXiv preprint arXiv:1902.09063},
  year={2019}
}
```

---

## References

1. **Chen et al. (2021)** - TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. [arXiv:2102.04306](https://arxiv.org/abs/2102.04306)

2. **Simpson et al. (2019)** - A large annotated medical image dataset for the development and evaluation of segmentation algorithms. [arXiv:1902.09063](https://arxiv.org/abs/1902.09063)

3. **Ronneberger et al. (2015)** - U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.

4. **Dosovitskiy et al. (2020)** - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.

5. **Medical Segmentation Decathlon** - [http://medicaldecathlon.com/](http://medicaldecathlon.com/)

---

## Acknowledgments

- **Medical Segmentation Decathlon** organizers for the publicly available dataset
- **MONAI** team for the medical imaging framework
- **TransUNet** authors for the architecture design
- **Memorial Sloan Kettering Cancer Center** for data collection

---

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/ihatesea69/TransUNet-Pancreas-Segmentation/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ihatesea69/TransUNet-Pancreas-Segmentation/discussions)

For questions about the implementation, please open an issue on GitHub.
