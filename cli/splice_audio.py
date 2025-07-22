#!/usr/bin/env python3
"""
Command-line interface for audio splicing.

Concatenates audio chunks into continuous narration with crossfade and outputs a JSON manifest.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio_splicing import (
    concatenate_audio_chunks,
    calculate_timing_metadata,
    save_audio_file,
    validate_audio_chunks,
    get_audio_splicing_config
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Splice audio chunks into continuous narration and output a manifest."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input RVC audio manifest file path."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for spliced audio."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to the pipeline configuration YAML file."
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="final_narration.wav",
        help="Output filename for spliced audio."
    )
    return parser.parse_args()


def load_rvc_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Load RVC audio manifest.
    
    Args:
        manifest_path: Path to RVC manifest file
        
    Returns:
        Dictionary containing manifest data
    """
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"RVC manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    return manifest


def extract_audio_paths(manifest: Dict[str, Any]) -> List[str]:
    """
    Extract audio file paths from RVC or TTS manifest.
    
    Args:
        manifest: RVC or TTS manifest data
        
    Returns:
        List of audio file paths in order
    """
    audio_paths = []
    
    # Try RVC manifest format first
    chunks = manifest.get("chunks", [])
    if chunks:
        successful_chunks = [chunk for chunk in chunks if chunk.get("status") == "success"]
        successful_chunks.sort(key=lambda x: x.get("id", ""))
        
        for chunk in successful_chunks:
            audio_path = chunk.get("audio_path")
            if audio_path and Path(audio_path).exists():
                audio_paths.append(audio_path)
    
    # Try TTS manifest format (tts_results)
    elif "tts_results" in manifest:
        tts_results = manifest.get("tts_results", [])
        successful_results = [result for result in tts_results if result.get("status") == "success"]
        
        for result in successful_results:
            audio_path = result.get("wav_path")
            if audio_path and Path(audio_path).exists():
                audio_paths.append(audio_path)
    
    return audio_paths


def create_splice_manifest(
    input_manifest: Dict[str, Any],
    output_file: str,
    timing_metadata: Dict[str, Any],
    splice_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create audio splicing manifest.
    
    Args:
        input_manifest: Original RVC manifest
        output_file: Output audio file path
        timing_metadata: Timing information
        splice_config: Splicing configuration
        
    Returns:
        Dictionary containing manifest data
    """
    successful_chunks = [chunk for chunk in input_manifest.get("chunks", []) 
                        if chunk.get("status") == "success"]
    
    manifest = {
        "step": "splice_audio",
        "timestamp": str(Path().cwd()),
        "output_file": output_file,
        "config": splice_config,
        "timing": timing_metadata,
        "summary": {
            "total_input_chunks": len(input_manifest.get("chunks", [])),
            "spliced_chunks": len(successful_chunks),
            "total_duration": timing_metadata.get("total_duration", 0),
            "crossfade_duration": timing_metadata.get("crossfade_duration", 0)
        },
        "input_chunks": successful_chunks
    }
    
    return manifest


def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        # Load splicing configuration
        splice_config = get_audio_splicing_config(args.config)
        
        # Load input manifest (RVC or TTS)
        input_manifest = load_rvc_manifest(args.input)
        
        # Extract audio paths
        audio_paths = extract_audio_paths(input_manifest)
        
        if not audio_paths:
            print("No successful audio chunks found in manifest", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(audio_paths)} audio chunks to splice")
        
        # Validate audio chunks
        is_valid, error_msg = validate_audio_chunks(audio_paths)
        if not is_valid:
            print(f"Audio chunk validation failed: {error_msg}", file=sys.stderr)
            sys.exit(1)
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output file path
        output_file = output_dir / args.output_name
        
        print(f"Splicing audio chunks with {splice_config['crossfade_duration']}s crossfade...")
        
        # Concatenate audio chunks
        spliced_audio, sample_rate = concatenate_audio_chunks(
            audio_paths,
            crossfade_duration=splice_config.get("crossfade_duration", 0.1),
            normalize=splice_config.get("normalize", True),
            target_db=splice_config.get("target_db", -20.0)
        )
        
        # Save spliced audio
        save_audio_file(spliced_audio, sample_rate, str(output_file))
        
        print(f"Spliced audio saved: {output_file}")
        
        # Calculate timing metadata
        # Handle both RVC and TTS manifest formats
        if "chunks" in input_manifest:
            chunks_for_timing = input_manifest.get("chunks", [])
        elif "tts_results" in input_manifest:
            # Convert TTS results to chunks format for timing calculation
            chunks_for_timing = []
            for i, result in enumerate(input_manifest.get("tts_results", [])):
                if result.get("status") == "success":
                    chunk = {
                        "id": f"chunk_{i:04d}",
                        "text": result.get("text", ""),
                        "duration": result.get("duration", 0)
                    }
                    chunks_for_timing.append(chunk)
        else:
            chunks_for_timing = []
        
        timing_metadata = calculate_timing_metadata(
            chunks_for_timing,
            crossfade_duration=splice_config.get("crossfade_duration", 0.1)
        )
        
        # Create manifest
        manifest = create_splice_manifest(
            input_manifest,
            str(output_file),
            timing_metadata,
            splice_config
        )
        
        # Save manifest
        manifest_path = output_dir / "splice_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Print summary
        summary = manifest["summary"]
        print(f"\nAudio Splicing Summary:")
        print(f"  Input chunks: {summary['total_input_chunks']}")
        print(f"  Spliced chunks: {summary['spliced_chunks']}")
        print(f"  Total duration: {summary['total_duration']:.2f}s")
        print(f"  Crossfade duration: {summary['crossfade_duration']}s")
        print(f"  Output file: {output_file}")
        print(f"  Manifest saved: {manifest_path}")
        
    except Exception as e:
        print(f"Audio splicing failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()