#!/usr/bin/env python3
"""
CLI tool for video analysis in the AutoVid pipeline.

This tool analyzes Keynote video exports to detect scene transitions and movement
ranges for synchronization with audio narration. It handles the 1-second delay
inherent in Keynote exports and integrates with the manifest-based pipeline.

Usage:
    python cli/analyze_video.py input_video.mov
    python cli/analyze_video.py input_video.mov --output analysis.json
    python cli/analyze_video.py input_video.mov --transcript transcript.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add the parent directory to the path so we can import from core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.video_analysis import (
    analyze_video, 
    VideoAnalysisError,
    probe_video_info
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_transcript_cues(transcript_path: str) -> List[str]:
    """
    Load transition cues from transcript JSON file.
    
    Args:
        transcript_path: Path to transcript JSON file
        
    Returns:
        List of transition cue strings
        
    Raises:
        ValueError: If transcript format is invalid
    """
    try:
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
        
        cues = []
        
        # Handle different transcript formats
        if 'slides' in transcript_data:
            # Standard transcript format
            for slide in transcript_data['slides']:
                if 'segments' in slide:
                    for segment in slide['segments']:
                        if segment.get('kind') == 'cue':
                            cues.append(segment.get('cue', '[transition]'))
        
        elif 'transcript' in transcript_data:
            # Alternative format
            transcript = transcript_data['transcript']
            if isinstance(transcript, list):
                for item in transcript:
                    if isinstance(item, dict) and item.get('type') == 'cue':
                        cues.append(item.get('text', '[transition]'))
        
        logger.info(f"Loaded {len(cues)} transition cues from {transcript_path}")
        return cues
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Failed to load transcript cues from {transcript_path}: {str(e)}")


def generate_output_path(video_path: str, suffix: str = "_analysis") -> str:
    """
    Generate output path for analysis manifest based on video path.
    
    Args:
        video_path: Path to input video file
        suffix: Suffix to add to filename
        
    Returns:
        Generated output path
    """
    video_path_obj = Path(video_path)
    output_dir = video_path_obj.parent / "workspace" / "video_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{video_path_obj.stem}{suffix}.json"
    return str(output_dir / output_filename)


def validate_video_file(video_path: str) -> bool:
    """
    Validate that the video file exists and is readable.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    if not os.access(video_path, os.R_OK):
        logger.error(f"Video file not readable: {video_path}")
        return False
    
    # Try to probe the video to ensure it's valid
    try:
        probe_video_info(video_path)
        return True
    except Exception as e:
        logger.error(f"Invalid video file {video_path}: {str(e)}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze Keynote video for scene transitions and movement ranges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python cli/analyze_video.py data/presentation.mov
    
    # With custom output path
    python cli/analyze_video.py data/presentation.mov --output results/analysis.json
    
    # With transcript validation
    python cli/analyze_video.py data/presentation.mov --transcript workspace/presentation_transcript.json
    
    # Custom thresholds
    python cli/analyze_video.py data/presentation.mov --scene-threshold 0.3 --movement-threshold 0.2
    
    # Disable Keynote delay compensation
    python cli/analyze_video.py data/presentation.mov --keynote-delay 0
        """
    )
    
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file to analyze"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for analysis JSON manifest (default: auto-generated)"
    )
    
    parser.add_argument(
        "--transcript", "-t",
        type=str,
        default=None,
        help="Path to transcript JSON file for transition validation"
    )
    
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.4,
        help="Scene detection threshold (0.0-1.0, higher = less sensitive, default: 0.4)"
    )
    
    parser.add_argument(
        "--movement-threshold",
        type=float,
        default=0.1,
        help="Movement detection threshold (0.0-1.0, default: 0.1)"
    )
    
    parser.add_argument(
        "--keynote-delay",
        type=float,
        default=1.0,
        help="Keynote export delay compensation in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without performing analysis"
    )
    
    parser.add_argument(
        "--step-id",
        type=str,
        default="analyze_video",
        help="Step ID for pipeline integration (default: analyze_video)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Validate input video
    logger.info(f"Validating video file: {args.video_path}")
    if not validate_video_file(args.video_path):
        logger.error("Video validation failed")
        sys.exit(1)
    
    # Load transcript cues if provided
    expected_transitions = None
    if args.transcript:
        try:
            expected_transitions = load_transcript_cues(args.transcript)
            logger.info(f"Loaded {len(expected_transitions)} expected transitions")
        except ValueError as e:
            logger.warning(f"Failed to load transcript: {e}")
            # Continue without transcript validation
    
    # Generate output path if not provided
    output_path = args.output
    if not output_path:
        output_path = generate_output_path(args.video_path)
        logger.info(f"Generated output path: {output_path}")
    
    # Validate thresholds
    if not (0.0 <= args.scene_threshold <= 1.0):
        logger.error("Scene threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not (0.0 <= args.movement_threshold <= 1.0):
        logger.error("Movement threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.keynote_delay < 0:
        logger.error("Keynote delay must be non-negative")
        sys.exit(1)
    
    # Log analysis parameters
    logger.info(f"Analysis parameters:")
    logger.info(f"  Scene threshold: {args.scene_threshold}")
    logger.info(f"  Movement threshold: {args.movement_threshold}")
    logger.info(f"  Keynote delay: {args.keynote_delay}s")
    logger.info(f"  Expected transitions: {len(expected_transitions) if expected_transitions else 'None'}")
    
    if args.dry_run:
        logger.info("Dry run completed successfully")
        sys.exit(0)
    
    # Perform video analysis
    try:
        logger.info("Starting video analysis...")
        
        manifest = analyze_video(
            video_path=args.video_path,
            expected_transitions=expected_transitions,
            scene_threshold=args.scene_threshold,
            movement_threshold=args.movement_threshold,
            keynote_delay=args.keynote_delay,
            output_path=output_path
        )
        
        # Extract key results for logging
        analysis = manifest['video_analysis']
        scene_count = analysis['total_scenes']
        movement_count = analysis['total_movements']
        validation = analysis['validation']
        
        logger.info(f"Analysis completed successfully:")
        logger.info(f"  Detected scenes: {scene_count}")
        logger.info(f"  Detected movements: {movement_count}")
        logger.info(f"  Validation status: {validation['status']}")
        logger.info(f"  Output saved to: {output_path}")
        
        # Print summary to stdout for pipeline integration
        summary = {
            'step_id': args.step_id,
            'status': 'success',
            'input_file': args.video_path,
            'output_file': output_path,
            'scene_count': scene_count,
            'movement_count': movement_count,
            'validation_status': validation['status'],
            'validation_message': validation['message']
        }
        
        print(json.dumps(summary, indent=2))
        
    except VideoAnalysisError as e:
        logger.error(f"Video analysis failed: {str(e)}")
        
        # Print error summary for pipeline integration
        error_summary = {
            'step_id': args.step_id,
            'status': 'error',
            'input_file': args.video_path,
            'error_message': str(e)
        }
        
        print(json.dumps(error_summary, indent=2))
        sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        
        error_summary = {
            'step_id': args.step_id,
            'status': 'error',
            'input_file': args.video_path,
            'error_message': f"Unexpected error: {str(e)}"
        }
        
        print(json.dumps(error_summary, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()