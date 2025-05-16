#!/usr/bin/env python3
"""
Script to check for matching PPTX and MOV files in a dataset.

This script scans a directory for PPTX files and checks if there are
corresponding MOV files with the same stem name. It creates a manifest
of the pairs and their status.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Check for matching PPTX and MOV files in a dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory containing the dataset files (default: data/)",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Output file for the manifest (default: prints to stdout)",
    )
    parser.add_argument(
        "--step_id",
        type=str,
        default="check_datasets",
        help="Identifier for the current step (default: check_datasets)",
    )
    return parser.parse_args()


def check_datasets(data_dir, step_id):
    """
    Check for matching PPTX and MOV files in the data directory.
    
    Args:
        data_dir (str): Path to the data directory
        step_id (str): Identifier for the current step
        
    Returns:
        dict: A manifest containing the pairs and status
    """
    data_path = Path(data_dir)
    pairs = []
    all_matched = True
    
    # Find all PPTX files
    for pptx_file in data_path.glob("*.pptx"):
        stem = pptx_file.stem
        mov_file = data_path / f"{stem}.mov"
        
        pair = {
            "stem": stem,
            "pptx": str(pptx_file),
            "mov": str(mov_file) if mov_file.exists() else None
        }
        
        if not mov_file.exists():
            all_matched = False
            
        pairs.append(pair)
    
    # Create the manifest
    manifest = {
        "step_id": step_id,
        "pairs": pairs,
        "status": "success" if all_matched else "failed"
    }
    
    return manifest


def main():
    """Main function."""
    args = parse_args()
    
    # Check datasets
    manifest = check_datasets(args.data, args.step_id)
    
    # Convert to JSON
    manifest_json = json.dumps(manifest, indent=2)
    
    # Print to stdout
    print(manifest_json)
    
    # Write to file if specified
    if args.out:
        with open(args.out, "w") as f:
            f.write(manifest_json)
    
    # Create the workspace/00_pairs directory if it doesn't exist
    pairs_dir = Path("workspace/00_pairs")
    pairs_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to the default manifest file
    with open(pairs_dir / "pairs_manifest.json", "w") as f:
        f.write(manifest_json)
    
    # Exit with code 1 if any matches are missing
    if manifest["status"] == "failed":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
