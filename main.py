"""Main entry point for TransUNet Pancreas Segmentation.

Provides CLI interface for training and inference.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def train(args):
    """Run training pipeline."""
    print(f"Starting training with config:")
    print(f"  Model variant: {args.variant}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print("\nFor full training, please use Notebook 03_Training_Pipeline.ipynb")


def inference(args):
    """Run inference on test data."""
    print(f"Running inference:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print("\nFor full inference, please use Notebook 04_Evaluation_and_Demo.ipynb")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TransUNet Pancreas Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python main.py train --variant small --epochs 50
  
  # Inference
  python main.py inference --checkpoint checkpoints/best_model.pth --input data/test.nii.gz
  
  # For interactive exploration, use Jupyter notebooks:
  jupyter notebook 01_Data_Exploration_and_Processing.ipynb
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--variant", default="small", choices=["small", "base", "large"],
                             help="Model variant (default: small)")
    train_parser.add_argument("--batch-size", type=int, default=8,
                             help="Batch size (default: 8)")
    train_parser.add_argument("--epochs", type=int, default=50,
                             help="Number of epochs (default: 50)")
    train_parser.add_argument("--lr", type=float, default=1e-4,
                             help="Learning rate (default: 1e-4)")
    
    # Inference arguments
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_parser.add_argument("--checkpoint", required=True,
                                 help="Path to model checkpoint")
    inference_parser.add_argument("--input", required=True,
                                 help="Path to input CT volume (.nii.gz)")
    inference_parser.add_argument("--output", default="outputs/prediction.nii.gz",
                                 help="Path to save prediction")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "inference":
        inference(args)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
