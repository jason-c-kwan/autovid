#!/usr/bin/env python3
"""
Command-line interface for RVC voice conversion.

Converts TTS audio chunks using RVC model and outputs a JSON manifest.
"""

import argparse
import json
import sys
import os
import uuid
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rvc_processing import (
    validate_rvc_model_files,
    get_rvc_config,
    create_rvc_workspace,
    validate_rvc_output,
    parse_rvc_error
)
from core.rvc_environment import (
    setup_rvc_environment,
    validate_rvc_environment,
    run_rvc_in_environment,
    get_rvc_environment_info
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert TTS audio using RVC model and output a manifest."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input TTS audio manifest file path."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for RVC converted audio."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to the pipeline configuration YAML file."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of audio files to process in parallel."
    )
    return parser.parse_args()


def load_tts_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Load TTS audio manifest.
    
    Args:
        manifest_path: Path to TTS manifest file
        
    Returns:
        Dictionary containing manifest data
    """
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"TTS manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    return manifest


def process_audio_chunk(
    chunk: Dict[str, Any],
    output_dir: str,
    rvc_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single audio chunk through RVC using isolated environment.
    
    Args:
        chunk: Audio chunk metadata
        output_dir: Output directory
        rvc_config: RVC configuration
        
    Returns:
        Dictionary with conversion results
    """
    chunk_id = chunk.get("id", str(uuid.uuid4()))
    input_audio = chunk.get("audio_path")
    
    if not input_audio or not Path(input_audio).exists():
        return {
            "id": chunk_id,
            "status": "error",
            "error": f"Input audio file not found: {input_audio}",
            "audio_path": None
        }
    
    # Generate output filename
    output_filename = f"rvc_{chunk_id}.wav"
    output_path = Path(output_dir) / output_filename
    
    try:
        # Extract RVC parameters
        rvc_params = {k: v for k, v in rvc_config.items() 
                     if k not in ["executable", "model_path", "index_path"]}
        
        # Run RVC in isolated environment
        success, error_msg = run_rvc_in_environment(
            model_path=rvc_config["model_path"],
            index_path=rvc_config["index_path"],
            input_audio=input_audio,
            output_audio=str(output_path),
            rvc_params=rvc_params,
            working_dir=output_dir
        )
        
        if not success:
            return {
                "id": chunk_id,
                "status": "error",
                "error": error_msg,
                "audio_path": None
            }
        
        # Validate output
        is_valid, error_msg = validate_rvc_output(str(output_path))
        if not is_valid:
            return {
                "id": chunk_id,
                "status": "error",
                "error": error_msg,
                "audio_path": None
            }
        
        return {
            "id": chunk_id,
            "status": "success",
            "audio_path": str(output_path),
            "original_path": input_audio,
            "text": chunk.get("text", ""),
            "duration": chunk.get("duration", 0),
            "rvc_config": rvc_config
        }
        
    except Exception as e:
        return {
            "id": chunk_id,
            "status": "error",
            "error": f"RVC conversion failed: {str(e)}",
            "audio_path": None
        }


def create_rvc_manifest(
    results: List[Dict[str, Any]],
    output_dir: str,
    rvc_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create RVC conversion manifest.
    
    Args:
        results: List of conversion results
        output_dir: Output directory
        rvc_config: RVC configuration
        
    Returns:
        Dictionary containing manifest data
    """
    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "error"]
    
    manifest = {
        "step": "rvc_convert",
        "timestamp": str(Path().cwd()),
        "output_dir": output_dir,
        "config": rvc_config,
        "summary": {
            "total_chunks": len(results),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0
        },
        "chunks": results
    }
    
    return manifest


def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        # Setup and validate RVC environment
        print("Setting up RVC environment...")
        if not setup_rvc_environment():
            print("Failed to setup RVC environment", file=sys.stderr)
            sys.exit(1)
        
        # Validate RVC environment
        is_valid, error_msg = validate_rvc_environment()
        if not is_valid:
            print(f"RVC environment validation failed: {error_msg}", file=sys.stderr)
            print("Environment info:")
            info = get_rvc_environment_info()
            for key, value in info.items():
                print(f"  {key}: {value}")
            sys.exit(1)
        
        # Load RVC configuration
        rvc_config = get_rvc_config(args.config)
        
        # Validate RVC model files
        is_valid, error_msg = validate_rvc_model_files(
            rvc_config["model_path"],
            rvc_config["index_path"]
        )
        if not is_valid:
            print(f"RVC model validation failed: {error_msg}", file=sys.stderr)
            sys.exit(1)
        
        # Load TTS manifest
        tts_manifest = load_tts_manifest(args.input)
        
        # Handle different manifest formats
        chunks = tts_manifest.get("chunks", [])
        if not chunks:
            # Try alternative format (tts_results from orpheus/piper)
            tts_results = tts_manifest.get("tts_results", [])
            if tts_results:
                # Convert tts_results format to chunks format
                chunks = []
                for i, result in enumerate(tts_results):
                    if result.get("status") == "success":
                        chunk = {
                            "id": f"chunk_{i:04d}",
                            "audio_path": result.get("wav_path"),
                            "text": result.get("text", ""),
                            "duration": result.get("duration", 0)
                        }
                        chunks.append(chunk)
        
        if not chunks:
            print("No audio chunks found in TTS manifest", file=sys.stderr)
            sys.exit(1)
        
        # Create output directory
        output_dir = create_rvc_workspace(args.output)
        
        # Process audio chunks
        print(f"Processing {len(chunks)} audio chunks through RVC...")
        results = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}: {chunk.get('id', 'unknown')}")
            
            result = process_audio_chunk(chunk, output_dir, rvc_config)
            results.append(result)
            
            if result["status"] == "error":
                print(f"  Error: {result['error']}", file=sys.stderr)
            else:
                print(f"  Success: {result['audio_path']}")
        
        # Create manifest
        manifest = create_rvc_manifest(results, output_dir, rvc_config)
        
        # Save manifest
        manifest_path = Path(output_dir) / "rvc_conversion_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Print summary
        summary = manifest["summary"]
        print(f"\nRVC Conversion Summary:")
        print(f"  Total chunks: {summary['total_chunks']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success rate: {summary['success_rate']:.2%}")
        print(f"  Manifest saved: {manifest_path}")
        
        # Exit with error code if any conversions failed
        if summary['failed'] > 0:
            sys.exit(1)
        
    except Exception as e:
        print(f"RVC conversion failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()