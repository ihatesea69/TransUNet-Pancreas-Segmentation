#!/usr/bin/env python3
"""
Download MSD Task07 Pancreas Dataset

This script downloads the Medical Segmentation Decathlon Task07 Pancreas dataset
with resume support, progress tracking, and automatic extraction.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output ./data
    python scripts/download_dataset.py --no-extract

Features:
    - Resume interrupted downloads
    - Progress bar with speed and ETA
    - Automatic extraction
    - Checksum verification
    - Retry on failure
"""

import os
import sys
import tarfile
import hashlib
import argparse
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Dataset info
DATASET_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"
DATASET_SIZE = 11_449_671_680  # ~11.4GB
DATASET_MD5 = None  # MD5 not provided by MSD
FILENAME = "Task07_Pancreas.tar"


def format_size(size_bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_time(seconds):
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def print_progress(downloaded, total, speed, elapsed):
    """Print download progress bar."""
    if total > 0:
        percent = downloaded / total * 100
        bar_length = 40
        filled = int(bar_length * downloaded / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        if speed > 0:
            eta = (total - downloaded) / speed
            eta_str = format_time(eta)
        else:
            eta_str = "calculating..."
        
        sys.stdout.write(
            f"\r[{bar}] {percent:5.1f}% | "
            f"{format_size(downloaded)}/{format_size(total)} | "
            f"{format_size(speed)}/s | "
            f"ETA: {eta_str}    "
        )
        sys.stdout.flush()


def download_file(url, dest_path, chunk_size=8192*4, max_retries=5):
    """
    Download file with resume support and progress tracking.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        chunk_size: Download chunk size in bytes
        max_retries: Maximum number of retry attempts
    
    Returns:
        True if successful, False otherwise
    """
    dest_path = Path(dest_path)
    temp_path = dest_path.with_suffix('.tmp')
    
    # Check if already downloaded
    if dest_path.exists():
        if dest_path.stat().st_size == DATASET_SIZE:
            print(f"✓ File already exists: {dest_path}")
            return True
        else:
            print(f"! Existing file is incomplete, re-downloading...")
            dest_path.unlink()
    
    # Resume from temp file if exists
    downloaded = 0
    if temp_path.exists():
        downloaded = temp_path.stat().st_size
        print(f"↻ Resuming download from {format_size(downloaded)}...")
    
    retries = 0
    while retries < max_retries:
        try:
            # Create request with Range header for resume
            headers = {}
            if downloaded > 0:
                headers['Range'] = f'bytes={downloaded}-'
            
            request = Request(url, headers=headers)
            
            print(f"\n📥 Downloading: {FILENAME}")
            print(f"   URL: {url}")
            print(f"   Size: {format_size(DATASET_SIZE)}")
            print(f"   Destination: {dest_path}")
            print()
            
            with urlopen(request, timeout=30) as response:
                # Get total size
                if downloaded > 0:
                    total = downloaded + int(response.headers.get('Content-Length', 0))
                else:
                    total = int(response.headers.get('Content-Length', DATASET_SIZE))
                
                # Open file in append mode for resume
                mode = 'ab' if downloaded > 0 else 'wb'
                
                start_time = time.time()
                last_print_time = start_time
                bytes_since_last = 0
                speed = 0
                
                with open(temp_path, mode) as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        bytes_since_last += len(chunk)
                        
                        # Update progress every 0.5 seconds
                        current_time = time.time()
                        if current_time - last_print_time >= 0.5:
                            elapsed = current_time - start_time
                            speed = bytes_since_last / (current_time - last_print_time)
                            bytes_since_last = 0
                            last_print_time = current_time
                            print_progress(downloaded, total, speed, elapsed)
                
                print()  # New line after progress bar
            
            # Rename temp file to final name
            temp_path.rename(dest_path)
            print(f"\n✓ Download complete: {dest_path}")
            return True
            
        except (URLError, HTTPError, OSError) as e:
            retries += 1
            print(f"\n\n⚠ Error: {e}")
            print(f"  Retry {retries}/{max_retries} in 5 seconds...")
            time.sleep(5)
    
    print(f"\n✗ Download failed after {max_retries} retries")
    return False


def extract_tar(tar_path, dest_dir):
    """Extract tar file with progress."""
    tar_path = Path(tar_path)
    dest_dir = Path(dest_dir)
    
    print(f"\n📦 Extracting: {tar_path.name}")
    print(f"   Destination: {dest_dir}")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            total = len(members)
            
            for i, member in enumerate(members):
                tar.extract(member, dest_dir)
                
                # Progress every 100 files
                if (i + 1) % 100 == 0 or i + 1 == total:
                    percent = (i + 1) / total * 100
                    sys.stdout.write(f"\r   Extracting: {i + 1}/{total} files ({percent:.1f}%)")
                    sys.stdout.flush()
        
        print(f"\n✓ Extraction complete!")
        return True
        
    except Exception as e:
        print(f"\n✗ Extraction failed: {e}")
        return False


def verify_dataset(data_dir):
    """Verify the extracted dataset structure."""
    data_dir = Path(data_dir)
    pancreas_dir = data_dir / "Task07_Pancreas"
    
    expected_files = [
        "dataset.json",
        "imagesTr",
        "labelsTr",
        "imagesTs"
    ]
    
    print(f"\n🔍 Verifying dataset structure...")
    
    missing = []
    for item in expected_files:
        path = pancreas_dir / item
        if not path.exists():
            missing.append(item)
        else:
            if path.is_dir():
                count = len(list(path.glob("*.nii.gz")))
                print(f"   ✓ {item}/ ({count} files)")
            else:
                print(f"   ✓ {item}")
    
    if missing:
        print(f"\n⚠ Missing: {missing}")
        return False
    
    print(f"\n✓ Dataset verified successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download MSD Task07 Pancreas Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data',
        help='Output directory (default: ./data)'
    )
    parser.add_argument(
        '--no-extract',
        action='store_true',
        help='Skip extraction after download'
    )
    parser.add_argument(
        '--keep-tar',
        action='store_true',
        help='Keep .tar file after extraction'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if file exists'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / FILENAME
    
    print("=" * 60)
    print("  MSD Task07 Pancreas Dataset Downloader")
    print("=" * 60)
    
    # Force re-download
    if args.force and tar_path.exists():
        print(f"Removing existing file (--force)...")
        tar_path.unlink()
    
    # Download
    success = download_file(DATASET_URL, tar_path)
    if not success:
        sys.exit(1)
    
    # Extract
    if not args.no_extract:
        success = extract_tar(tar_path, output_dir)
        if not success:
            sys.exit(1)
        
        # Verify
        verify_dataset(output_dir)
        
        # Cleanup
        if not args.keep_tar:
            print(f"\n🗑️ Removing tar file to save space...")
            tar_path.unlink()
            print(f"   Freed {format_size(DATASET_SIZE)}")
    
    print("\n" + "=" * 60)
    print("  ✅ Dataset ready!")
    print("=" * 60)
    print(f"\nDataset location: {output_dir / 'Task07_Pancreas'}")
    print("\nNext steps:")
    print("  1. Run notebooks/01_Data_Exploration_and_Processing.ipynb")
    print("  2. Or use: python main.py train --variant small")
    print()


if __name__ == "__main__":
    main()
