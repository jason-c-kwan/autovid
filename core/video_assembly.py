#!/usr/bin/env python3
"""
Video segment assembly pipeline for AutoVid.

This module handles the extraction, processing, and reassembly of video segments
with synchronized timing gaps to create the final synchronized presentation video.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import ffmpeg

from .slide_sync import SlideSegment, SyncPlan
from .gap_management import GapManager

logger = logging.getLogger(__name__)


@dataclass
class VideoSegment:
    """Represents an extracted video segment."""
    slide_number: int
    segment_path: str
    start_time: float
    duration: float
    has_gap: bool = False
    gap_path: Optional[str] = None
    gap_duration: float = 0.0


class VideoAssembler:
    """Assembles synchronized video from segments and gaps."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the video assembler.
        
        Args:
            temp_dir: Directory for temporary files (default: system temp)
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "autovid_assembly"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.gap_manager = GapManager(str(self.temp_dir))
        
    def extract_video_segments(
        self, 
        sync_plan: SyncPlan,
        output_dir: Optional[str] = None
    ) -> List[VideoSegment]:
        """
        Extract video segments based on the synchronization plan.
        
        Args:
            sync_plan: Complete synchronization plan
            output_dir: Directory for extracted segments (default: temp)
            
        Returns:
            List of VideoSegment objects
        """
        if output_dir is None:
            output_dir = self.temp_dir / "segments"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        segments = []
        video_path = sync_plan.video_path
        
        logger.info(f"Extracting {len(sync_plan.segments)} video segments from {video_path}")
        
        for slide_segment in sync_plan.segments:
            segment_path = output_dir / f"segment_slide_{slide_segment.slide_number:03d}.mp4"
            
            try:
                # Extract segment with precise timing
                (
                    ffmpeg
                    .input(video_path, ss=slide_segment.keynote_start, t=slide_segment.keynote_duration)
                    .output(
                        str(segment_path),
                        vcodec='libx264',
                        acodec='aac',
                        pix_fmt='yuv420p',
                        preset='medium'
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                video_segment = VideoSegment(
                    slide_number=slide_segment.slide_number,
                    segment_path=str(segment_path),
                    start_time=slide_segment.keynote_start,
                    duration=slide_segment.keynote_duration,
                    has_gap=slide_segment.gap_needed > 0,
                    gap_duration=slide_segment.gap_needed
                )
                
                segments.append(video_segment)
                logger.debug(f"Extracted segment {slide_segment.slide_number}: {segment_path}")
                
            except ffmpeg.Error as e:
                logger.error(f"Failed to extract segment {slide_segment.slide_number}: {e.stderr.decode()}")
                raise
        
        logger.info(f"Successfully extracted {len(segments)} video segments")
        return segments
    
    def create_segment_gaps(
        self, 
        segments: List[VideoSegment],
        sync_plan: SyncPlan,
        video_info: Dict[str, Any]
    ) -> List[VideoSegment]:
        """
        Create gap videos for segments that require them.
        
        Args:
            segments: List of extracted video segments
            sync_plan: Synchronization plan with gap requirements
            video_info: Video metadata for format matching
            
        Returns:
            Updated list of segments with gap paths
        """
        gap_dir = self.temp_dir / "gaps"
        gap_videos = self.gap_manager.create_all_gaps(sync_plan, video_info, str(gap_dir))
        
        # Update segments with gap information
        for segment in segments:
            if segment.has_gap and segment.slide_number in gap_videos:
                segment.gap_path = gap_videos[segment.slide_number]
                logger.debug(f"Assigned gap video to segment {segment.slide_number}: {segment.gap_path}")
        
        return segments
    
    def assemble_final_video(
        self, 
        segments: List[VideoSegment],
        audio_path: str,
        output_path: str,
        video_info: Dict[str, Any]
    ) -> str:
        """
        Assemble the final synchronized video from segments and gaps.
        
        Args:
            segments: List of video segments with gaps
            audio_path: Path to synchronized audio
            output_path: Output path for final video
            video_info: Video metadata for encoding settings
            
        Returns:
            Path to the assembled video
        """
        logger.info(f"Assembling final video with {len(segments)} segments")
        
        # Create concatenation list for complex assembly
        concat_list_path = self.temp_dir / "concat_list.txt"
        
        try:
            # Build concatenation list
            self._create_concat_list(segments, concat_list_path)
            
            # Assemble video using concatenation
            video_stream = ffmpeg.input(str(concat_list_path), format='concat', safe=0)
            audio_stream = ffmpeg.input(audio_path)
            
            # Create final output with video and audio
            output = ffmpeg.output(
                video_stream,
                audio_stream,
                output_path,
                vcodec='libx264',
                acodec='aac',
                pix_fmt='yuv420p',
                preset='medium',
                crf=23,  # Good quality
                shortest=None,  # Don't truncate based on shortest stream
                map_metadata=-1  # Strip metadata for clean output
            )
            
            ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            logger.info(f"Successfully assembled video: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"Video assembly failed: {e.stderr.decode()}")
            raise
    
    def _create_concat_list(self, segments: List[VideoSegment], concat_list_path: Path):
        """Create FFmpeg concatenation list file."""
        with open(concat_list_path, 'w') as f:
            for segment in segments:
                # Add the main segment
                f.write(f"file '{segment.segment_path}'\n")
                
                # Add gap if present
                if segment.has_gap and segment.gap_path:
                    f.write(f"file '{segment.gap_path}'\n")
        
        logger.debug(f"Created concatenation list: {concat_list_path}")
    
    def assemble_with_complex_sync(
        self,
        sync_plan: SyncPlan,
        audio_path: str,
        output_path: str,
        video_info: Dict[str, Any]
    ) -> str:
        """
        Assemble video using complex FFmpeg filter graph for precise synchronization.
        
        Args:
            sync_plan: Complete synchronization plan
            audio_path: Path to synchronized audio
            output_path: Output path for final video
            video_info: Video metadata
            
        Returns:
            Path to assembled video
        """
        logger.info("Using complex filter graph for video assembly")
        
        try:
            # Create filter graph for complex assembly
            filter_complex = self._build_complex_filter(sync_plan, video_info)
            
            # Build ffmpeg command
            input_video = ffmpeg.input(sync_plan.video_path)
            input_audio = ffmpeg.input(audio_path)
            
            # Apply complex filter
            video_out = input_video.filter_complex(filter_complex, 'vout')
            
            # Combine with audio
            output = ffmpeg.output(
                video_out,
                input_audio,
                output_path,
                vcodec='libx264',
                acodec='aac',
                pix_fmt='yuv420p',
                preset='medium',
                crf=23
            )
            
            ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            logger.info(f"Complex assembly complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Complex assembly failed: {e}")
            # Fallback to standard assembly
            logger.info("Falling back to standard assembly method")
            segments = self.extract_video_segments(sync_plan)
            segments = self.create_segment_gaps(segments, sync_plan, video_info)
            return self.assemble_final_video(segments, audio_path, output_path, video_info)
    
    def _build_complex_filter(self, sync_plan: SyncPlan, video_info: Dict[str, Any]) -> str:
        """
        Build complex FFmpeg filter graph for synchronized assembly.
        
        Args:
            sync_plan: Synchronization plan
            video_info: Video metadata
            
        Returns:
            Complex filter string
        """
        filters = []
        current_time = 0.0
        
        for i, segment in enumerate(sync_plan.segments):
            # Trim segment
            segment_filter = f"[0:v]trim={segment.keynote_start}:{segment.keynote_end},setpts=PTS-STARTPTS[v{i}]"
            filters.append(segment_filter)
            
            # Add gap if needed
            if segment.gap_needed > 0:
                if segment.gap_type == "static_hold":
                    # Create static frame hold
                    gap_filter = (
                        f"[0:v]trim={segment.keynote_end-0.1}:{segment.keynote_end},"
                        f"loop=loop=-1:size=1:start=0,"
                        f"trim=duration={segment.gap_needed},"
                        f"setpts=PTS-STARTPTS[gap{i}]"
                    )
                    filters.append(gap_filter)
                else:
                    # For animated/fade holds, use simpler approach
                    gap_filter = (
                        f"[0:v]trim={max(0, segment.keynote_end-2)}:{segment.keynote_end},"
                        f"loop=loop=-1:size=30,"  # Loop last 30 frames
                        f"trim=duration={segment.gap_needed},"
                        f"setpts=PTS-STARTPTS[gap{i}]"
                    )
                    filters.append(gap_filter)
        
        # Concatenate all segments and gaps
        concat_inputs = []
        for i in range(len(sync_plan.segments)):
            concat_inputs.append(f"[v{i}]")
            if sync_plan.segments[i].gap_needed > 0:
                concat_inputs.append(f"[gap{i}]")
        
        concat_filter = f"{''.join(concat_inputs)}concat=n={len(concat_inputs)}:v=1:a=0[vout]"
        filters.append(concat_filter)
        
        return ";".join(filters)
    
    def create_preview_assembly(
        self,
        sync_plan: SyncPlan,
        audio_path: str,
        output_path: str,
        preview_duration: float = 30.0
    ) -> str:
        """
        Create a preview video showing synchronization quality.
        
        Args:
            sync_plan: Synchronization plan
            audio_path: Path to audio
            output_path: Output path for preview
            preview_duration: Maximum preview duration in seconds
            
        Returns:
            Path to preview video
        """
        logger.info(f"Creating preview assembly ({preview_duration}s)")
        
        # Select segments for preview (first few slides or critical points)
        preview_segments = []
        total_duration = 0.0
        
        for segment in sync_plan.segments:
            segment_total = segment.keynote_duration + segment.gap_needed
            if total_duration + segment_total <= preview_duration:
                preview_segments.append(segment)
                total_duration += segment_total
            else:
                break
        
        if not preview_segments:
            preview_segments = sync_plan.segments[:1]  # At least one segment
        
        # Create preview sync plan
        preview_plan = SyncPlan(
            video_path=sync_plan.video_path,
            audio_path=audio_path,
            segments=preview_segments,
            total_original_duration=sum(s.keynote_duration for s in preview_segments),
            total_sync_duration=sum(s.keynote_duration + s.gap_needed for s in preview_segments),
            keynote_delay_compensation=sync_plan.keynote_delay_compensation
        )
        
        # Assemble preview
        return self.assemble_with_complex_sync(
            preview_plan, audio_path, output_path, 
            {'frame_rate': 30, 'width': 1920, 'height': 1080}  # Default preview settings
        )
    
    def validate_assembly(self, video_path: str, expected_duration: float) -> Dict[str, Any]:
        """
        Validate the assembled video against expected parameters.
        
        Args:
            video_path: Path to assembled video
            expected_duration: Expected total duration
            
        Returns:
            Validation results
        """
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            actual_duration = float(video_stream.get('duration', 0))
            duration_diff = abs(actual_duration - expected_duration)
            
            validation = {
                'valid': True,
                'actual_duration': actual_duration,
                'expected_duration': expected_duration,
                'duration_difference': duration_diff,
                'has_video': True,
                'has_audio': audio_stream is not None,
                'video_codec': video_stream.get('codec_name'),
                'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
                'issues': []
            }
            
            # Check for issues
            if duration_diff > 1.0:  # More than 1 second difference
                validation['issues'].append(f"Duration mismatch: {duration_diff:.2f}s difference")
                validation['valid'] = False
            
            if not audio_stream:
                validation['issues'].append("No audio stream found")
                validation['valid'] = False
            
            if validation['valid']:
                logger.info(f"Assembly validation passed: {video_path}")
            else:
                logger.warning(f"Assembly validation issues: {validation['issues']}")
            
            return validation
            
        except Exception as e:
            logger.error(f"Assembly validation failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'issues': ['Validation failed due to error']
            }
    
    def cleanup_temp_files(self):
        """Clean up temporary assembly files."""
        try:
            # Clean up segments
            segments_dir = self.temp_dir / "segments"
            if segments_dir.exists():
                for segment_file in segments_dir.glob("*.mp4"):
                    segment_file.unlink()
                segments_dir.rmdir()
            
            # Clean up concatenation lists
            for concat_file in self.temp_dir.glob("concat_list*.txt"):
                concat_file.unlink()
            
            # Clean up gap manager temps
            self.gap_manager.cleanup_temp_files()
            
            logger.info("Cleaned up assembly temporary files")
            
        except Exception as e:
            logger.warning(f"Assembly cleanup failed: {e}")


def assemble_synchronized_video(
    sync_plan: SyncPlan,
    audio_path: str,
    output_path: str,
    video_info: Optional[Dict[str, Any]] = None,
    method: str = "standard",
    temp_dir: Optional[str] = None
) -> str:
    """
    Main function to assemble synchronized video from sync plan.
    
    Args:
        sync_plan: Complete synchronization plan
        audio_path: Path to synchronized audio
        output_path: Output path for final video
        video_info: Video metadata (will probe if not provided)
        method: Assembly method ('standard' or 'complex')
        temp_dir: Temporary directory for processing
        
    Returns:
        Path to assembled video
    """
    assembler = VideoAssembler(temp_dir)
    
    try:
        # Get video info if not provided
        if video_info is None:
            probe = ffmpeg.probe(sync_plan.video_path)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            video_info = {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'frame_rate': eval(video_stream.get('r_frame_rate', '30/1')),
                'duration': float(video_stream.get('duration', 0))
            }
        
        # Choose assembly method
        if method == "complex":
            result = assembler.assemble_with_complex_sync(sync_plan, audio_path, output_path, video_info)
        else:
            # Standard method: extract segments, create gaps, assemble
            segments = assembler.extract_video_segments(sync_plan)
            segments = assembler.create_segment_gaps(segments, sync_plan, video_info)
            result = assembler.assemble_final_video(segments, audio_path, output_path, video_info)
        
        # Validate assembly
        validation = assembler.validate_assembly(result, sync_plan.total_sync_duration)
        
        if not validation['valid']:
            logger.warning(f"Assembly validation issues: {validation['issues']}")
        
        return result
        
    finally:
        # Always cleanup
        assembler.cleanup_temp_files()


def create_assembly_manifest(
    sync_plan: SyncPlan,
    assembled_video_path: str,
    validation_results: Dict[str, Any],
    output_path: str
) -> str:
    """
    Create a manifest documenting the assembly process.
    
    Args:
        sync_plan: Synchronization plan used
        assembled_video_path: Path to final video
        validation_results: Assembly validation results
        output_path: Path for assembly manifest
        
    Returns:
        Path to created manifest
    """
    from datetime import datetime
    
    manifest = {
        'assembly_info': {
            'timestamp': datetime.now().isoformat(),
            'input_video': sync_plan.video_path,
            'input_audio': sync_plan.audio_path,
            'output_video': assembled_video_path,
            'total_segments': len(sync_plan.segments),
            'total_gaps': sum(1 for s in sync_plan.segments if s.gap_needed > 0),
            'original_duration': sync_plan.total_original_duration,
            'synchronized_duration': sync_plan.total_sync_duration,
            'timing_expansion': (
                sync_plan.total_sync_duration / sync_plan.total_original_duration - 1
            ) * 100 if sync_plan.total_original_duration > 0 else 0
        },
        'segments': [
            {
                'slide_number': seg.slide_number,
                'original_start': seg.keynote_start,
                'original_duration': seg.keynote_duration,
                'gap_needed': seg.gap_needed,
                'gap_type': seg.gap_type
            }
            for seg in sync_plan.segments
        ],
        'validation': validation_results,
        'sync_plan': sync_plan.to_dict()
    }
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created assembly manifest: {output_path}")
    return output_path