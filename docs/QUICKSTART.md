# TransUNet Quick Start Guide

**5-Minute Setup for Pancreas Segmentation**

---

## 🚀 Installation (2 minutes)

```bash
# Clone and setup
git clone https://github.com/ihatesea69/TransUNet-Pancreas-Segmentation.git
cd TransUNet-Pancreas-Segmentation
uv sync && .venv\Scripts\activate
```

---

## 📚 Run Notebooks (Interactive)

```bash
# Notebook 01: Download data & preprocessing (one-time)
jupyter notebook 01_Data_Exploration_and_Processing.ipynb

# Notebook 02: Explore TransUNet architecture
jupyter notebook 02_Model_Architecture.ipynb

# Notebook 03: Train model (requires GPU)
jupyter notebook 03_Training_Pipeline.ipynb

# Notebook 04: Evaluate & visualize results
jupyter notebook 04_Evaluation_and_Demo.ipynb
```

---

## ⚡ CLI Commands (Scripted)

```bash
# Training
python main.py train --variant small --epochs 50

# Inference
python main.py inference \
  --checkpoint checkpoints/model.pth \
  --input data/scan.nii.gz
```

---

## 🐍 Python API (Programmatic)

```python
from src.model import create_transunet
import torch

# Load model
model = create_transunet(variant="small")
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])

# Inference
with torch.no_grad():
    output = model(input_slice)
    prediction = torch.argmax(output, dim=1)
```

---

## 📊 Project Structure

```
TransUNet-Pancreas-Segmentation/
├── 📓 01_Data_*.ipynb       ← Start here
├── 📓 02_Model_*.ipynb      ← Architecture
├── 📓 03_Training_*.ipynb   ← Train
├── 📓 04_Evaluation_*.ipynb ← Evaluate
├── 📦 src/                  ← Core code
├── 💾 checkpoints/          ← Model weights
└── 📊 outputs/              ← Results
```

---

## ⚙️ Model Variants

| Variant | VRAM | Parameters | Speed |
|---------|------|-----------|-------|
| small   | 4GB  | 17M       | Fast  |
| base    | 12GB | 105M      | Medium|
| large   | 24GB | 300M      | Slow  |

---

## 🔧 Common Issues

**Dataset download fails?**
```python
# In notebook 01, manually set:
DATASET_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"
```

**CUDA out of memory?**
```python
# Reduce batch size in notebook 03:
CONFIG["batch_size"] = 4  # or 2
```

**Windows num_workers error?**
```python
# Set num_workers to 0:
CONFIG["num_workers"] = 0
```

---

## 📖 Learn More

- [Full Documentation](README.md)
- [Architecture Details](02_Model_Architecture.ipynb)
- [Original Paper](https://arxiv.org/abs/2102.04306)
- [Dataset Info](http://medicaldecathlon.com/)

---

## 🆘 Support

- [Open an Issue](https://github.com/ihatesea69/TransUNet-Pancreas-Segmentation/issues)
- [Discussions](https://github.com/ihatesea69/TransUNet-Pancreas-Segmentation/discussions)
