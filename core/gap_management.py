#!/usr/bin/env python3
"""
Timing gap management for AutoVid video synchronization.

This module handles the creation and management of timing gaps between
video segments to achieve perfect synchronization with narration while
preserving original Keynote animation timings.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import ffmpeg
import numpy as np

from .slide_sync import SlideSegment, SyncPlan

logger = logging.getLogger(__name__)


@dataclass
class GapFiller:
    """Represents a timing gap filler between video segments."""
    duration: float
    gap_type: str
    source_frame: Optional[str] = None  # Path to extracted frame for static holds
    source_segment: Optional[str] = None  # Path to source segment for animated holds
    fade_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError(f"Gap duration must be positive, got {self.duration}")


class GapManager:
    """Manages timing gaps and video segment holds for synchronization."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the gap manager.
        
        Args:
            temp_dir: Directory for temporary files (default: system temp)
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(exist_ok=True)
        
    def create_gap_filler(
        self, 
        segment: SlideSegment, 
        video_path: str,
        temp_prefix: str = "gap"
    ) -> Optional[GapFiller]:
        """
        Create a gap filler for a slide segment based on its requirements.
        
        Args:
            segment: SlideSegment requiring a gap filler
            video_path: Path to the source video
            temp_prefix: Prefix for temporary files
            
        Returns:
            GapFiller object or None if no gap needed
        """
        if segment.gap_needed <= 0:
            return None
            
        logger.info(f"Creating {segment.gap_type} gap filler for {segment.gap_needed:.2f}s")
        
        gap_filler = GapFiller(
            duration=segment.gap_needed,
            gap_type=segment.gap_type
        )
        
        try:
            if segment.gap_type == "static_hold":
                gap_filler = self._create_static_hold(gap_filler, segment, video_path, temp_prefix)
            elif segment.gap_type == "animated_hold":
                gap_filler = self._create_animated_hold(gap_filler, segment, video_path, temp_prefix)
            elif segment.gap_type == "fade_hold":
                gap_filler = self._create_fade_hold(gap_filler, segment, video_path, temp_prefix)
            else:
                logger.warning(f"Unknown gap type: {segment.gap_type}, using static hold")
                gap_filler.gap_type = "static_hold"
                gap_filler = self._create_static_hold(gap_filler, segment, video_path, temp_prefix)
                
        except Exception as e:
            logger.error(f"Failed to create gap filler: {e}")
            # Fallback to simple static hold
            gap_filler.gap_type = "static_hold"
            gap_filler = self._create_static_hold(gap_filler, segment, video_path, temp_prefix)
        
        return gap_filler
    
    def _create_static_hold(
        self, 
        gap_filler: GapFiller, 
        segment: SlideSegment,
        video_path: str,
        temp_prefix: str
    ) -> GapFiller:
        """
        Create a static frame hold by extracting the last frame of the segment.
        
        Args:
            gap_filler: GapFiller object to populate
            segment: SlideSegment to extract frame from
            video_path: Path to source video
            temp_prefix: Prefix for temp files
            
        Returns:
            Updated GapFiller object
        """
        # Extract the last frame of the segment
        frame_time = segment.keynote_end - 0.1  # Slightly before segment end
        frame_path = self.temp_dir / f"{temp_prefix}_slide_{segment.slide_number}_frame.png"
        
        try:
            # Extract frame using ffmpeg
            (
                ffmpeg
                .input(video_path, ss=frame_time)
                .output(str(frame_path), vframes=1, format='image2')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            gap_filler.source_frame = str(frame_path)
            logger.debug(f"Extracted static frame: {frame_path}")
            
        except ffmpeg.Error as e:
            logger.error(f"Failed to extract frame: {e.stderr.decode()}")
            raise
            
        return gap_filler
    
    def _create_animated_hold(
        self, 
        gap_filler: GapFiller, 
        segment: SlideSegment,
        video_path: str,
        temp_prefix: str
    ) -> GapFiller:
        """
        Create an animated hold by looping the last portion of the segment.
        
        Args:
            gap_filler: GapFiller object to populate
            segment: SlideSegment to extract animation from
            video_path: Path to source video
            temp_prefix: Prefix for temp files
            
        Returns:
            Updated GapFiller object
        """
        # Extract the last 2 seconds (or segment duration if shorter) for looping
        loop_duration = min(2.0, segment.keynote_duration * 0.5)
        loop_start = segment.keynote_end - loop_duration
        
        segment_path = self.temp_dir / f"{temp_prefix}_slide_{segment.slide_number}_loop.mp4"
        
        try:
            # Extract loop segment
            (
                ffmpeg
                .input(video_path, ss=loop_start, t=loop_duration)
                .output(str(segment_path), vcodec='libx264', pix_fmt='yuv420p')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            gap_filler.source_segment = str(segment_path)
            logger.debug(f"Extracted animation loop: {segment_path}")
            
        except ffmpeg.Error as e:
            logger.error(f"Failed to extract animation segment: {e.stderr.decode()}")
            # Fallback to static hold
            return self._create_static_hold(gap_filler, segment, video_path, temp_prefix)
            
        return gap_filler
    
    def _create_fade_hold(
        self, 
        gap_filler: GapFiller, 
        segment: SlideSegment,
        video_path: str,
        temp_prefix: str
    ) -> GapFiller:
        """
        Create a fade hold that gradually transitions to a static frame.
        
        Args:
            gap_filler: GapFiller object to populate
            segment: SlideSegment to extract frame from
            video_path: Path to source video
            temp_prefix: Prefix for temp files
            
        Returns:
            Updated GapFiller object
        """
        # First create the static frame
        gap_filler = self._create_static_hold(gap_filler, segment, video_path, temp_prefix)
        
        # Add fade parameters
        fade_duration = min(1.0, gap_filler.duration * 0.3)  # Fade over 30% of gap or 1s max
        gap_filler.fade_params = {
            'fade_duration': fade_duration,
            'fade_type': 'in',  # Fade in to static
            'fade_start': 0
        }
        
        logger.debug(f"Added fade parameters: {gap_filler.fade_params}")
        return gap_filler
    
    def create_gap_video(
        self, 
        gap_filler: GapFiller,
        output_path: str,
        video_info: Dict[str, Any]
    ) -> str:
        """
        Create the actual gap video file from a GapFiller object.
        
        Args:
            gap_filler: GapFiller object with gap specifications
            output_path: Path for the output gap video
            video_info: Video metadata for format matching
            
        Returns:
            Path to the created gap video file
        """
        logger.info(f"Creating {gap_filler.gap_type} gap video: {output_path}")
        
        try:
            if gap_filler.gap_type == "static_hold":
                return self._create_static_gap_video(gap_filler, output_path, video_info)
            elif gap_filler.gap_type == "animated_hold":
                return self._create_animated_gap_video(gap_filler, output_path, video_info)
            elif gap_filler.gap_type == "fade_hold":
                return self._create_fade_gap_video(gap_filler, output_path, video_info)
            else:
                raise ValueError(f"Unknown gap type: {gap_filler.gap_type}")
                
        except Exception as e:
            logger.error(f"Failed to create gap video: {e}")
            raise
    
    def _create_static_gap_video(
        self, 
        gap_filler: GapFiller,
        output_path: str,
        video_info: Dict[str, Any]
    ) -> str:
        """Create a static hold gap video from an extracted frame."""
        if not gap_filler.source_frame:
            raise ValueError("No source frame for static gap")
        
        # Create video from static frame
        (
            ffmpeg
            .input(gap_filler.source_frame, loop=1, t=gap_filler.duration)
            .output(
                output_path,
                vcodec='libx264',
                pix_fmt='yuv420p',
                r=video_info.get('frame_rate', 30),
                s=f"{video_info['width']}x{video_info['height']}"
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        logger.debug(f"Created static gap video: {output_path}")
        return output_path
    
    def _create_animated_gap_video(
        self, 
        gap_filler: GapFiller,
        output_path: str,
        video_info: Dict[str, Any]
    ) -> str:
        """Create an animated hold gap video by looping a segment."""
        if not gap_filler.source_segment:
            raise ValueError("No source segment for animated gap")
        
        # Get source segment info
        try:
            probe = ffmpeg.probe(gap_filler.source_segment)
            segment_duration = float(probe['streams'][0]['duration'])
        except (ffmpeg.Error, KeyError):
            segment_duration = 2.0  # Fallback duration
        
        # Calculate how many loops we need
        loops_needed = int(np.ceil(gap_filler.duration / segment_duration))
        
        # Create looped video
        input_stream = ffmpeg.input(gap_filler.source_segment)
        
        if loops_needed == 1:
            # Single loop, just trim to exact duration
            stream = input_stream.filter('trim', duration=gap_filler.duration)
        else:
            # Multiple loops needed
            streams = [input_stream] * loops_needed
            stream = ffmpeg.concat(*streams, v=1, a=0).filter('trim', duration=gap_filler.duration)
        
        (
            stream
            .output(
                output_path,
                vcodec='libx264',
                pix_fmt='yuv420p',
                r=video_info.get('frame_rate', 30)
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        logger.debug(f"Created animated gap video: {output_path}")
        return output_path
    
    def _create_fade_gap_video(
        self, 
        gap_filler: GapFiller,
        output_path: str,
        video_info: Dict[str, Any]
    ) -> str:
        """Create a fade hold gap video with gradual transition to static."""
        if not gap_filler.source_frame or not gap_filler.fade_params:
            raise ValueError("Missing source frame or fade parameters")
        
        fade_duration = gap_filler.fade_params['fade_duration']
        static_duration = gap_filler.duration - fade_duration
        
        # Create the static portion
        static_stream = (
            ffmpeg
            .input(gap_filler.source_frame, loop=1, t=gap_filler.duration)
            .filter('fps', video_info.get('frame_rate', 30))
        )
        
        # Apply fade effect
        faded_stream = static_stream.filter(
            'fade',
            type='in',
            start_time=0,
            duration=fade_duration
        )
        
        (
            faded_stream
            .output(
                output_path,
                vcodec='libx264',
                pix_fmt='yuv420p',
                s=f"{video_info['width']}x{video_info['height']}"
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        logger.debug(f"Created fade gap video: {output_path}")
        return output_path
    
    def create_all_gaps(
        self, 
        sync_plan: SyncPlan,
        video_info: Dict[str, Any],
        output_dir: str
    ) -> Dict[int, str]:
        """
        Create all required gap videos for a synchronization plan.
        
        Args:
            sync_plan: Complete synchronization plan
            video_info: Video metadata for format matching
            output_dir: Directory for gap video files
            
        Returns:
            Dictionary mapping slide numbers to gap video paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        gap_videos = {}
        
        for segment in sync_plan.segments:
            if segment.gap_needed > 0:
                # Create gap filler
                gap_filler = self.create_gap_filler(
                    segment, 
                    sync_plan.video_path,
                    temp_prefix=f"sync_gap_{segment.slide_number}"
                )
                
                if gap_filler:
                    # Create gap video
                    gap_video_path = output_dir / f"gap_slide_{segment.slide_number}.mp4"
                    self.create_gap_video(gap_filler, str(gap_video_path), video_info)
                    gap_videos[segment.slide_number] = str(gap_video_path)
                    
                    logger.info(f"Created gap video for slide {segment.slide_number}: {gap_video_path}")
        
        logger.info(f"Created {len(gap_videos)} gap videos in {output_dir}")
        return gap_videos
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during gap generation."""
        try:
            temp_files = list(self.temp_dir.glob("gap_*"))
            temp_files.extend(list(self.temp_dir.glob("sync_gap_*")))
            
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_file}")
                except OSError as e:
                    logger.warning(f"Failed to clean up {temp_file}: {e}")
                    
            logger.info(f"Cleaned up {len(temp_files)} temporary files")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def optimize_gap_types(sync_plan: SyncPlan, transcript_data: Dict[str, Any]) -> SyncPlan:
    """
    Optimize gap types based on transcript content analysis.
    
    Args:
        sync_plan: Original synchronization plan
        transcript_data: Transcript data for content analysis
        
    Returns:
        Updated synchronization plan with optimized gap types
    """
    transcript_segments = transcript_data.get('segments', [])
    
    for i, segment in enumerate(sync_plan.segments):
        if segment.gap_needed > 0 and i < len(transcript_segments):
            text = transcript_segments[i].get('text', '').lower()
            
            # Analyze text content to determine optimal gap type
            if any(keyword in text for keyword in [
                'chart', 'graph', 'animation', 'moving', 'dynamic', 'changing'
            ]):
                segment.gap_type = 'animated_hold'
            elif any(keyword in text for keyword in [
                'summary', 'conclusion', 'final', 'end', 'thank you', 'questions'
            ]):
                segment.gap_type = 'fade_hold'
            elif segment.gap_needed > 5.0:  # Long gaps get animated holds
                segment.gap_type = 'animated_hold'
            else:
                segment.gap_type = 'static_hold'
    
    logger.info("Optimized gap types based on content analysis")
    return sync_plan


def validate_gap_requirements(sync_plan: SyncPlan) -> Dict[str, Any]:
    """
    Validate gap requirements and return analysis.
    
    Args:
        sync_plan: Synchronization plan to validate
        
    Returns:
        Dictionary with gap analysis and recommendations
    """
    total_gaps = sum(seg.gap_needed for seg in sync_plan.segments)
    gap_segments = [seg for seg in sync_plan.segments if seg.gap_needed > 0]
    
    analysis = {
        'total_gap_duration': total_gaps,
        'gap_segment_count': len(gap_segments),
        'average_gap': total_gaps / len(gap_segments) if gap_segments else 0,
        'max_gap': max((seg.gap_needed for seg in gap_segments), default=0),
        'gap_distribution': {
            'static_hold': len([seg for seg in gap_segments if seg.gap_type == 'static_hold']),
            'animated_hold': len([seg for seg in gap_segments if seg.gap_type == 'animated_hold']),
            'fade_hold': len([seg for seg in gap_segments if seg.gap_type == 'fade_hold'])
        },
        'recommendations': []
    }
    
    # Add recommendations based on analysis
    if analysis['max_gap'] > 10.0:
        analysis['recommendations'].append(
            f"Consider reviewing slide {next(seg.slide_number for seg in gap_segments if seg.gap_needed == analysis['max_gap'])}: "
            f"very long gap required ({analysis['max_gap']:.1f}s)"
        )
    
    if analysis['total_gap_duration'] > sync_plan.total_original_duration * 0.5:
        analysis['recommendations'].append(
            "Total gap duration exceeds 50% of original video - consider adjusting narration pacing"
        )
    
    return analysis