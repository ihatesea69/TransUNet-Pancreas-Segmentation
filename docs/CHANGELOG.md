# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-01-31

### Added
- Initial implementation of TransUNet architecture for pancreas segmentation
- Complete MONAI preprocessing pipeline with HU windowing
- SlicingDataset for 2D slice extraction from 3D volumes
- Hybrid loss function combining Dice Loss and Cross-Entropy Loss
- Four Jupyter notebooks covering data exploration, model architecture, training, and evaluation
- CLI interface in main.py for training and inference
- Comprehensive documentation in README.md (NVIDIA personaplex-inspired design)
- CLAUDE.md for AI assistant development guidance
- Visualization utilities for CT slices and predictions
- Professional README with badges, clear sections, and navigation
- Assets folder structure for diagrams and visual documentation

### Changed
- **README.md completely redesigned** following NVIDIA personaplex style:
  - Added badges for Python, PyTorch, MONAI, License
  - Restructured with clear visual hierarchy
  - Added emoji icons for better navigation
  - Expanded Usage section with 4 notebook breakdowns
  - Added CLI interface documentation
  - Created detailed preprocessing pipeline table
  - Enhanced citation section with BibTeX
  - Professional contact & support section
- Updated pyproject.toml description to be more descriptive
- Fixed Python version requirement from >=3.13 to >=3.9 for better compatibility
- Enhanced src/__init__.py with proper module exports and __all__
- Corrected dataset size information to ~11.4GB (compressed) across all files
- Fixed GitHub repository owner in README.md (ihatesea69)

### Fixed
- Corrected model.py line count in CLAUDE.md (18K → 533 lines)
- Standardized dataset size documentation
- Added known issues section to CLAUDE.md
- Fixed inconsistent formatting in documentation

### Project Structure
```
├── 📓 01-04_*.ipynb          # Educational notebooks
├── 📦 src/                   # Core implementation
│   ├── model.py              # TransUNet (533 lines)
│   ├── dataset.py            # SlicingDataset
│   ├── transforms.py         # MONAI pipeline
│   ├── loss.py               # HybridLoss
│   └── utils.py              # Visualization
├── 📁 assets/                # Documentation visuals
├── 💾 checkpoints/           # Model weights
├── 📊 outputs/               # Results
├── main.py                   # CLI entry point
├── pyproject.toml            # UV config
├── CHANGELOG.md              # Version history
└── README.md                 # Main documentation
```

### Documentation Highlights
- **Clean, professional README** inspired by NVIDIA's open-source projects
- **Comprehensive Usage guide** with 3 access methods (Notebooks, CLI, Programmatic)
- **Visual architecture diagram** placeholder in assets/
- **Detailed preprocessing pipeline** table
- **Performance benchmarks** section
- **Proper BibTeX citations** for academic reference

### Notes
- Dataset must be downloaded via notebook 01 (~11.4GB)
- Windows users should set num_workers=0 in DataLoader
- Trained model checkpoints not included (empty checkpoints/ directory)
- Architecture diagram in assets/ is currently ASCII placeholder - replace with PNG/SVG
