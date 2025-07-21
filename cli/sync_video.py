#!/usr/bin/env python3
"""
Video synchronization CLI for AutoVid pipeline.

This script provides a command-line interface for synchronizing video files
with processed audio using the AutoVid video sync engine. It supports both
basic and advanced synchronization modes with comprehensive validation.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.video_sync import (
    load_timing_manifests, 
    extract_scene_timings, 
    extract_audio_chunk_timings,
    calculate_sync_points,
    generate_timing_corrections,
    validate_scene_audio_mapping,
    create_sync_manifest,
    get_video_sync_config,
    VideoSyncError
)

from core.sync_engine import (
    synchronize_video_audio,
    create_preview_clips,
    SyncEngineError
)

from core.sync_validator import (
    validate_sync_accuracy,
    export_validation_report,
    check_timing_drift,
    SyncValidationError
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(__name__)


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate command line arguments and input files."""
    # Check required files exist
    required_files = [args.video, args.audio]
    
    if args.video_manifest:
        required_files.append(args.video_manifest)
    
    if args.audio_manifest:
        required_files.append(args.audio_manifest)
    
    for file_path in required_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Validate output directory
    output_dir = Path(args.output).parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create output directory: {output_dir}, {e}")


def load_manifests(args: argparse.Namespace, logger: logging.Logger) -> tuple:
    """Load video and audio manifests if provided."""
    video_manifest = None
    audio_manifest = None
    
    if args.video_manifest:
        logger.info(f"Loading video analysis manifest: {args.video_manifest}")
        try:
            video_manifest, audio_manifest = load_timing_manifests(
                args.video_manifest,
                args.audio_manifest or args.video_manifest  # Fallback if audio manifest not specified
            )
        except VideoSyncError as e:
            if args.audio_manifest:
                logger.error(f"Failed to load manifests: {e}")
                raise
            else:
                logger.warning(f"Failed to load video manifest, using basic sync: {e}")
    
    return video_manifest, audio_manifest


def calculate_synchronization_data(video_manifest: Optional[Dict[str, Any]], 
                                 audio_manifest: Optional[Dict[str, Any]],
                                 sync_config: Dict[str, Any],
                                 logger: logging.Logger) -> tuple:
    """Calculate synchronization points and timing corrections."""
    sync_points = []
    timing_corrections = None
    
    if video_manifest and audio_manifest:
        logger.info("Calculating synchronization points from manifests")
        
        # Extract timing data
        scene_transitions = extract_scene_timings(video_manifest)
        audio_chunks = extract_audio_chunk_timings(audio_manifest)
        
        # Calculate sync points
        crossfade_duration = sync_config.get('crossfade_duration', 0.1)
        sync_points = calculate_sync_points(
            scene_transitions, 
            audio_chunks, 
            crossfade_duration
        )
        
        # Generate timing corrections
        video_duration = video_manifest['video_analysis']['video_info']['duration']
        tolerance = sync_config.get('sync_tolerance', 0.1)
        timing_corrections = generate_timing_corrections(
            sync_points, 
            video_duration, 
            tolerance
        )
        
        logger.info(f"Calculated {len(sync_points)} sync points with timing corrections")
    
    return sync_points, timing_corrections


def perform_synchronization(args: argparse.Namespace,
                          sync_points: list,
                          timing_corrections,
                          sync_config: Dict[str, Any],
                          logger: logging.Logger) -> str:
    """Perform the actual video-audio synchronization."""
    logger.info("Starting video-audio synchronization")
    
    try:
        output_video_path = synchronize_video_audio(
            video_path=args.video,
            audio_path=args.audio,
            output_path=args.output,
            sync_points=sync_points,
            timing_corrections=timing_corrections,
            sync_config=sync_config
        )
        
        logger.info(f"Synchronization completed: {output_video_path}")
        return output_video_path
        
    except SyncEngineError as e:
        logger.error(f"Synchronization failed: {e}")
        raise


def validate_synchronization(output_video_path: str,
                           sync_points: list,
                           timing_corrections,
                           args: argparse.Namespace,
                           logger: logging.Logger) -> None:
    """Validate the synchronization quality."""
    if not args.validate:
        return
    
    logger.info("Validating synchronization quality")
    
    try:
        # Perform validation
        validation_report = validate_sync_accuracy(
            video_path=output_video_path,
            sync_points=sync_points,
            timing_corrections=timing_corrections
        )
        
        # Export validation report
        if args.validation_report:
            report_path = export_validation_report(validation_report, args.validation_report)
            logger.info(f"Validation report exported: {report_path}")
        
        # Log validation summary
        logger.info(f"Validation Grade: {validation_report.overall_grade}")
        logger.info(f"Sync Accuracy: {validation_report.sync_metrics.sync_accuracy_score:.1f}%")
        logger.info(f"Average Offset: {validation_report.sync_metrics.avg_offset*1000:.1f}ms")
        
        # Log recommendations if any
        if validation_report.recommendations:
            logger.info("Recommendations:")
            for rec in validation_report.recommendations[:3]:  # Show top 3
                logger.info(f"  - {rec}")
        
    except SyncValidationError as e:
        logger.warning(f"Validation failed: {e}")


def create_preview_if_requested(args: argparse.Namespace,
                              sync_points: list,
                              logger: logging.Logger) -> None:
    """Create preview clips if requested."""
    if not args.preview_dir:
        return
    
    logger.info(f"Creating preview clips in {args.preview_dir}")
    
    try:
        preview_clips = create_preview_clips(
            video_path=args.video,
            audio_path=args.audio,
            sync_points=sync_points,
            output_dir=args.preview_dir,
            clip_duration=args.preview_duration
        )
        
        logger.info(f"Created {len(preview_clips)} preview clips")
        
    except SyncEngineError as e:
        logger.warning(f"Preview creation failed: {e}")


def save_sync_manifest(video_manifest_path: Optional[str],
                      audio_manifest_path: Optional[str],
                      sync_points: list,
                      timing_corrections,
                      validation_result: Optional[Dict[str, Any]],
                      output_video_path: str,
                      manifest_output_path: str,
                      logger: logging.Logger) -> None:
    """Save synchronization manifest if requested."""
    if not manifest_output_path:
        return
    
    logger.info(f"Saving synchronization manifest: {manifest_output_path}")
    
    try:
        # Create validation result if not provided
        if validation_result is None and sync_points:
            # Quick validation
            drift_check = check_timing_drift(sync_points)
            validation_result = {
                'status': 'BASIC_CHECK',
                'drift_detected': drift_check['has_drift'],
                'requires_attention': drift_check['requires_attention']
            }
        elif validation_result is None:
            validation_result = {'status': 'NO_VALIDATION', 'message': 'No sync points available'}
        
        # Create sync manifest
        sync_manifest = create_sync_manifest(
            video_manifest_path=video_manifest_path or "direct_input",
            audio_manifest_path=audio_manifest_path or "direct_input",
            sync_points=sync_points,
            timing_corrections=timing_corrections,
            validation_result=validation_result,
            output_video_path=output_video_path
        )
        
        # Save manifest
        with open(manifest_output_path, 'w') as f:
            json.dump(sync_manifest, f, indent=2)
        
        logger.info(f"Sync manifest saved: {manifest_output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save sync manifest: {e}")


def main():
    """Main function for video synchronization CLI."""
    parser = argparse.ArgumentParser(
        description="Synchronize video with processed audio using AutoVid sync engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic synchronization (no timing data)
  python cli/sync_video.py input.mov final_audio.wav output.mp4

  # Advanced synchronization with manifests
  python cli/sync_video.py input.mov final_audio.wav output.mp4 \\
    --video-manifest video_analysis.json \\
    --audio-manifest splice_manifest.json \\
    --validate

  # Create preview clips for manual verification
  python cli/sync_video.py input.mov final_audio.wav output.mp4 \\
    --preview-dir preview_clips \\
    --preview-duration 5.0
        """
    )
    
    # Required arguments
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('audio', help='Path to input audio file')
    parser.add_argument('output', help='Path for output synchronized video')
    
    # Optional manifest inputs
    parser.add_argument('--video-manifest', 
                       help='Path to video analysis manifest JSON file')
    parser.add_argument('--audio-manifest',
                       help='Path to audio splice manifest JSON file')
    
    # Synchronization options
    parser.add_argument('--keynote-delay', type=float, default=None,
                       help='Keynote delay compensation in seconds (overrides config)')
    parser.add_argument('--sync-tolerance', type=float, default=None,
                       help='Sync tolerance in seconds (overrides config)')
    parser.add_argument('--video-codec', default=None,
                       help='Video codec for output (default: copy)')
    parser.add_argument('--audio-codec', default=None,
                       help='Audio codec for output (default: aac)')
    
    # Validation options
    parser.add_argument('--validate', action='store_true',
                       help='Validate synchronization quality')
    parser.add_argument('--validation-report', 
                       help='Path to save validation report JSON')
    
    # Preview options
    parser.add_argument('--preview-dir',
                       help='Directory to create preview clips')
    parser.add_argument('--preview-duration', type=float, default=10.0,
                       help='Duration of preview clips in seconds')
    
    # Output options
    parser.add_argument('--sync-manifest',
                       help='Path to save synchronization manifest')
    parser.add_argument('--config', default='config/pipeline.yaml',
                       help='Path to pipeline configuration file')
    
    # General options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    try:
        logger.info("AutoVid Video Synchronization CLI")
        logger.info(f"Video: {args.video}")
        logger.info(f"Audio: {args.audio}")
        logger.info(f"Output: {args.output}")
        
        # Validate inputs
        validate_inputs(args)
        
        # Load configuration
        sync_config = get_video_sync_config(args.config)
        
        # Override config with command line arguments
        if args.keynote_delay is not None:
            sync_config['keynote_delay'] = args.keynote_delay
        if args.sync_tolerance is not None:
            sync_config['sync_tolerance'] = args.sync_tolerance
        if args.video_codec is not None:
            sync_config['video_codec'] = args.video_codec
        if args.audio_codec is not None:
            sync_config['audio_codec'] = args.audio_codec
        
        logger.info(f"Sync config: {sync_config}")
        
        if args.dry_run:
            logger.info("DRY RUN: Would perform synchronization with above settings")
            return 0
        
        # Load manifests if provided
        video_manifest, audio_manifest = load_manifests(args, logger)
        
        # Calculate synchronization data
        sync_points, timing_corrections = calculate_synchronization_data(
            video_manifest, audio_manifest, sync_config, logger
        )
        
        # Validate scene-audio mapping if we have manifests
        validation_result = None
        if video_manifest and audio_manifest:
            validation_result = validate_scene_audio_mapping(
                video_manifest, audio_manifest, sync_points
            )
            logger.info(f"Scene-audio mapping validation: {validation_result['status']}")
        
        # Perform synchronization
        output_video_path = perform_synchronization(
            args, sync_points, timing_corrections, sync_config, logger
        )
        
        # Validate synchronization quality
        validate_synchronization(
            output_video_path, sync_points, timing_corrections, args, logger
        )
        
        # Create preview clips if requested
        create_preview_if_requested(args, sync_points, logger)
        
        # Save sync manifest if requested
        save_sync_manifest(
            args.video_manifest,
            args.audio_manifest,
            sync_points,
            timing_corrections,
            validation_result,
            output_video_path,
            args.sync_manifest,
            logger
        )
        
        logger.info("Video synchronization completed successfully")
        print(f"Synchronized video saved to: {output_video_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
        
    except Exception as e:
        logger.error(f"Video synchronization failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())