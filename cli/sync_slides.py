#!/usr/bin/env python3
"""
Slide synchronization CLI for AutoVid.

This script orchestrates the complete slide synchronization process, taking
Keynote video exports and synchronizing them with AI-generated narration
based on PowerPoint speaker notes and [transition] cues.
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.slide_sync import SlideSynchronizer, load_sync_plan
from core.gap_management import GapManager, optimize_gap_types, validate_gap_requirements
from core.video_assembly import VideoAssembler, assemble_synchronized_video, create_assembly_manifest
from core.video_analysis import probe_video_info, analyze_video

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synchronize Keynote video with AI-generated narration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create synchronization plan
  python sync_slides.py plan --video data/presentation.mov --audio workspace/final_audio.wav 
                           --video-analysis workspace/video_analysis.json 
                           --audio-manifest workspace/audio_splice.json
                           --transcript workspace/transcript.json
                           --output workspace/sync_plan.json

  # Execute synchronization from existing plan
  python sync_slides.py execute --sync-plan workspace/sync_plan.json 
                               --output workspace/synchronized_video.mp4

  # Full synchronization in one step
  python sync_slides.py sync --video data/presentation.mov --audio workspace/final_audio.wav
                            --video-analysis workspace/video_analysis.json
                            --audio-manifest workspace/audio_splice.json
                            --transcript workspace/transcript.json
                            --output workspace/synchronized_video.mp4

  # Create preview
  python sync_slides.py preview --sync-plan workspace/sync_plan.json
                               --output workspace/preview.mp4 --duration 30
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Plan command
    plan_parser = subparsers.add_parser('plan', help='Create synchronization plan')
    plan_parser.add_argument('--video', required=True, help='Path to Keynote video file')
    plan_parser.add_argument('--audio', required=True, help='Path to final narration audio')
    plan_parser.add_argument('--video-analysis', required=True, 
                            help='Path to video analysis manifest')
    plan_parser.add_argument('--audio-manifest', required=True,
                            help='Path to audio splice manifest')
    plan_parser.add_argument('--transcript', required=True,
                            help='Path to transcript manifest')
    plan_parser.add_argument('--output', required=True,
                            help='Output path for sync plan JSON')
    plan_parser.add_argument('--keynote-delay', type=float, default=1.0,
                            help='Keynote export delay compensation (default: 1.0s)')
    plan_parser.add_argument('--optimize-gaps', action='store_true',
                            help='Optimize gap types based on content analysis')
    
    # Execute command
    execute_parser = subparsers.add_parser('execute', help='Execute synchronization from plan')
    execute_parser.add_argument('--sync-plan', required=True,
                               help='Path to synchronization plan JSON')
    execute_parser.add_argument('--output', required=True,
                               help='Output path for synchronized video')
    execute_parser.add_argument('--method', choices=['standard', 'complex'], default='standard',
                               help='Assembly method (default: standard)')
    execute_parser.add_argument('--temp-dir', help='Temporary directory for processing')
    execute_parser.add_argument('--keep-temps', action='store_true',
                               help='Keep temporary files for debugging')
    
    # Full sync command
    sync_parser = subparsers.add_parser('sync', help='Full synchronization process')
    sync_parser.add_argument('--video', required=True, help='Path to Keynote video file')
    sync_parser.add_argument('--audio', required=True, help='Path to final narration audio')
    sync_parser.add_argument('--video-analysis', required=True,
                            help='Path to video analysis manifest')
    sync_parser.add_argument('--audio-manifest', required=True,
                            help='Path to audio splice manifest')
    sync_parser.add_argument('--transcript', required=True,
                            help='Path to transcript manifest')
    sync_parser.add_argument('--output', required=True,
                            help='Output path for synchronized video')
    sync_parser.add_argument('--sync-plan-output', 
                            help='Optional path to save sync plan JSON')
    sync_parser.add_argument('--assembly-manifest',
                            help='Optional path to save assembly manifest JSON')
    sync_parser.add_argument('--keynote-delay', type=float, default=1.0,
                            help='Keynote export delay compensation (default: 1.0s)')
    sync_parser.add_argument('--method', choices=['standard', 'complex'], default='standard',
                            help='Assembly method (default: standard)')
    sync_parser.add_argument('--optimize-gaps', action='store_true',
                            help='Optimize gap types based on content analysis')
    sync_parser.add_argument('--temp-dir', help='Temporary directory for processing')
    sync_parser.add_argument('--keep-temps', action='store_true',
                            help='Keep temporary files for debugging')
    
    # Preview command
    preview_parser = subparsers.add_parser('preview', help='Create preview video')
    preview_parser.add_argument('--sync-plan', required=True,
                               help='Path to synchronization plan JSON')
    preview_parser.add_argument('--output', required=True,
                               help='Output path for preview video')
    preview_parser.add_argument('--duration', type=float, default=30.0,
                               help='Preview duration in seconds (default: 30)')
    preview_parser.add_argument('--temp-dir', help='Temporary directory for processing')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate synchronization plan')
    validate_parser.add_argument('--sync-plan', required=True,
                                help='Path to synchronization plan JSON')
    validate_parser.add_argument('--detailed', action='store_true',
                                help='Show detailed validation report')
    
    # Analyze command (helper to run video analysis if needed)
    analyze_parser = subparsers.add_parser('analyze', help='Analyze video for transitions')
    analyze_parser.add_argument('--video', required=True, help='Path to Keynote video file')
    analyze_parser.add_argument('--output', required=True, help='Output path for analysis manifest')
    analyze_parser.add_argument('--transcript', help='Path to transcript for validation')
    analyze_parser.add_argument('--keynote-delay', type=float, default=1.0,
                               help='Keynote export delay compensation (default: 1.0s)')
    analyze_parser.add_argument('--scene-threshold', type=float, default=0.4,
                               help='Scene detection sensitivity (default: 0.4)')
    
    return parser.parse_args()


def create_sync_plan(args) -> str:
    """Create synchronization plan from manifests."""
    logger.info("Creating synchronization plan")
    
    # Initialize synchronizer
    synchronizer = SlideSynchronizer(keynote_delay=args.keynote_delay)
    
    # Create sync plan
    sync_plan = synchronizer.create_sync_plan(
        video_analysis_manifest=args.video_analysis,
        audio_splice_manifest=args.audio_manifest,
        transcript_manifest=args.transcript,
        output_path=args.output
    )
    
    # Optimize gap types if requested
    if args.optimize_gaps:
        logger.info("Optimizing gap types based on content analysis")
        with open(args.transcript, 'r') as f:
            transcript_data = json.load(f)
        sync_plan = optimize_gap_types(sync_plan, transcript_data)
        
        # Save optimized plan
        with open(args.output, 'w') as f:
            json.dump(sync_plan.to_dict(), f, indent=2)
    
    # Validate and report
    validation = synchronizer.validate_sync_plan(sync_plan)
    gap_analysis = validate_gap_requirements(sync_plan)
    
    logger.info(f"Sync plan created: {len(sync_plan.segments)} segments")
    logger.info(f"Total timing expansion: {validation['metrics']['timing_expansion']:.1f}%")
    logger.info(f"Gaps required: {gap_analysis['gap_segment_count']}/{len(sync_plan.segments)} segments")
    
    if validation['warnings']:
        for warning in validation['warnings']:
            logger.warning(warning)
    
    if gap_analysis['recommendations']:
        for rec in gap_analysis['recommendations']:
            logger.info(f"Recommendation: {rec}")
    
    return args.output


def execute_sync_plan(args) -> str:
    """Execute synchronization from existing plan."""
    logger.info(f"Executing synchronization plan: {args.sync_plan}")
    
    # Load sync plan
    sync_plan = load_sync_plan(args.sync_plan)
    
    # Validate inputs exist
    if not Path(sync_plan.video_path).exists():
        raise FileNotFoundError(f"Video file not found: {sync_plan.video_path}")
    if not Path(sync_plan.audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {sync_plan.audio_path}")
    
    # Assemble synchronized video
    result_path = assemble_synchronized_video(
        sync_plan=sync_plan,
        audio_path=sync_plan.audio_path,
        output_path=args.output,
        method=args.method,
        temp_dir=args.temp_dir
    )
    
    logger.info(f"Synchronized video created: {result_path}")
    return result_path


def full_sync_process(args) -> str:
    """Execute full synchronization process."""
    logger.info("Starting full synchronization process")
    
    # Create sync plan
    logger.info("Step 1: Creating synchronization plan")
    synchronizer = SlideSynchronizer(keynote_delay=args.keynote_delay)
    
    sync_plan = synchronizer.create_sync_plan(
        video_analysis_manifest=args.video_analysis,
        audio_splice_manifest=args.audio_manifest,
        transcript_manifest=args.transcript
    )
    
    # Optimize gaps if requested
    if args.optimize_gaps:
        with open(args.transcript, 'r') as f:
            transcript_data = json.load(f)
        sync_plan = optimize_gap_types(sync_plan, transcript_data)
    
    # Save sync plan if requested
    if args.sync_plan_output:
        with open(args.sync_plan_output, 'w') as f:
            json.dump(sync_plan.to_dict(), f, indent=2)
        logger.info(f"Sync plan saved: {args.sync_plan_output}")
    
    # Execute synchronization
    logger.info("Step 2: Assembling synchronized video")
    result_path = assemble_synchronized_video(
        sync_plan=sync_plan,
        audio_path=args.audio,
        output_path=args.output,
        method=args.method,
        temp_dir=args.temp_dir
    )
    
    # Create assembly manifest if requested
    if args.assembly_manifest:
        assembler = VideoAssembler(args.temp_dir)
        validation = assembler.validate_assembly(result_path, sync_plan.total_sync_duration)
        create_assembly_manifest(sync_plan, result_path, validation, args.assembly_manifest)
        logger.info(f"Assembly manifest saved: {args.assembly_manifest}")
    
    logger.info(f"Full synchronization complete: {result_path}")
    return result_path


def create_preview(args) -> str:
    """Create preview video."""
    logger.info(f"Creating preview video: {args.output}")
    
    # Load sync plan
    sync_plan = load_sync_plan(args.sync_plan)
    
    # Create preview
    assembler = VideoAssembler(args.temp_dir)
    result_path = assembler.create_preview_assembly(
        sync_plan=sync_plan,
        audio_path=sync_plan.audio_path,
        output_path=args.output,
        preview_duration=args.duration
    )
    
    logger.info(f"Preview created: {result_path}")
    return result_path


def validate_plan(args):
    """Validate synchronization plan."""
    logger.info(f"Validating synchronization plan: {args.sync_plan}")
    
    # Load sync plan
    sync_plan = load_sync_plan(args.sync_plan)
    
    # Validate plan
    synchronizer = SlideSynchronizer()
    validation = synchronizer.validate_sync_plan(sync_plan)
    gap_analysis = validate_gap_requirements(sync_plan)
    
    # Report results
    print(f"\n=== Slide Synchronization Plan Validation ===")
    print(f"Plan file: {args.sync_plan}")
    print(f"Status: {'VALID' if validation['valid'] else 'INVALID'}")
    print(f"Total segments: {validation['metrics']['total_segments']}")
    print(f"Segments requiring gaps: {gap_analysis['gap_segment_count']}")
    print(f"Timing expansion: {validation['metrics']['timing_expansion']:.1f}%")
    print(f"Maximum gap: {gap_analysis['max_gap']:.1f}s")
    print(f"Average gap: {gap_analysis['average_gap']:.1f}s")
    
    if validation['warnings']:
        print(f"\n--- Warnings ---")
        for warning in validation['warnings']:
            print(f"⚠️  {warning}")
    
    if validation['errors']:
        print(f"\n--- Errors ---")
        for error in validation['errors']:
            print(f"❌ {error}")
    
    if gap_analysis['recommendations']:
        print(f"\n--- Recommendations ---")
        for rec in gap_analysis['recommendations']:
            print(f"💡 {rec}")
    
    if args.detailed:
        print(f"\n--- Detailed Segment Analysis ---")
        for i, segment in enumerate(sync_plan.segments):
            print(f"Slide {segment.slide_number:2d}: "
                  f"{segment.keynote_duration:5.1f}s → "
                  f"{segment.keynote_duration + segment.gap_needed:5.1f}s "
                  f"(gap: {segment.gap_needed:4.1f}s, {segment.gap_type})")
    
    print(f"\n=== End Validation Report ===\n")


def analyze_video(args) -> str:
    """Analyze video for transition points."""
    logger.info(f"Analyzing video: {args.video}")
    
    # Load transcript for validation if provided
    expected_transitions = None
    if args.transcript:
        try:
            with open(args.transcript, 'r') as f:
                transcript_data = json.load(f)
            
            # Extract transition cues
            segments = transcript_data.get('segments', [])
            expected_transitions = []
            for segment in segments:
                text = segment.get('text', '')
                if '[transition]' in text.lower():
                    expected_transitions.append('[transition]')
            
            logger.info(f"Found {len(expected_transitions)} expected transitions in transcript")
            
        except Exception as e:
            logger.warning(f"Failed to load transcript for validation: {e}")
    
    # Perform video analysis
    analysis_result = analyze_video(
        video_path=args.video,
        expected_transitions=expected_transitions,
        scene_threshold=args.scene_threshold,
        keynote_delay=args.keynote_delay,
        output_path=args.output
    )
    
    # Report results
    video_info = analysis_result['video_analysis']['video_info']
    transitions = analysis_result['video_analysis']['scene_transitions']
    validation = analysis_result['video_analysis']['validation']
    
    logger.info(f"Video analyzed: {video_info['width']}x{video_info['height']}, "
               f"{video_info['duration']:.1f}s, {len(transitions)} transitions detected")
    logger.info(f"Validation status: {validation['status']}")
    
    if validation['status'] != 'PASS':
        logger.warning(f"Validation message: {validation['message']}")
    
    return args.output


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    
    if not args.command:
        print("Error: No command specified. Use --help for usage information.")
        sys.exit(1)
    
    try:
        if args.command == 'plan':
            result = create_sync_plan(args)
            print(f"Synchronization plan created: {result}")
            
        elif args.command == 'execute':
            result = execute_sync_plan(args)
            print(f"Synchronized video created: {result}")
            
        elif args.command == 'sync':
            result = full_sync_process(args)
            print(f"Full synchronization complete: {result}")
            
        elif args.command == 'preview':
            result = create_preview(args)
            print(f"Preview video created: {result}")
            
        elif args.command == 'validate':
            validate_plan(args)
            
        elif args.command == 'analyze':
            result = analyze_video(args)
            print(f"Video analysis complete: {result}")
            
        else:
            print(f"Error: Unknown command '{args.command}'")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()