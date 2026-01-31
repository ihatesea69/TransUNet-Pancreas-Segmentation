# Assets Folder

This folder contains visual assets for the project README and documentation.

## Files

### `architecture_diagram.png` (To be added)
Visual representation of the TransUNet architecture showing:
- CNN Encoder (ResNet-style)
- Vision Transformer Bottleneck
- CNN Decoder with skip connections
- Input/Output flow

**How to create:**
You can create this diagram using:
- Draw.io / diagrams.net
- Figma
- PowerPoint/Keynote
- Python visualization (matplotlib/plotly)

**Recommended dimensions:** 1200x600 px

---

### Example Architecture Diagram Content:

```
┌─────────────────────────────────────────────────────────┐
│                   TransUNet Architecture                 │
└─────────────────────────────────────────────────────────┘

Input CT Slice (1×224×224)
         │
         ▼
┌────────────────────┐
│   CNN Encoder      │ ← ResNet-style
│   (4 stages)       │   Multi-scale features
└─────────┬──────────┘
          │ Features: 64→128→256→512 channels
          │ Resolution: 1/4→1/8→1/16→1/32
          ▼
┌────────────────────┐
│ Patch Embedding    │ ← Project to 768-dim
│ (512 → 768)        │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Vision Transformer │ ← 12 layers
│ (12 heads, 768d)   │   Self-attention
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   CNN Decoder      │ ← U-Net style
│   (4 stages)       │   + Skip connections
└─────────┬──────────┘
          │
          ▼
    Segmentation Map (2×224×224)
    [Background | Pancreas]
```

---

## Other Potential Assets

- `example_ct_slice.png` - Sample CT input
- `example_prediction.png` - Model output visualization
- `training_curves.png` - Loss/accuracy plots
- `attention_maps.png` - Transformer attention visualization
