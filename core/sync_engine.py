"""
Video synchronization engine for AutoVid pipeline.

This module provides the core FFmpeg-based synchronization engine that 
combines video and audio streams with precise timing alignment. It handles
complex filter graphs, timing corrections, and crossfade boundary processing.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

from .video_sync import SyncPoint, TimingCorrection, VideoSyncError

logger = logging.getLogger(__name__)


class SyncEngineError(Exception):
    """Custom exception for sync engine errors."""
    pass


def probe_media_streams(media_path: str) -> Dict[str, Any]:
    """
    Probe media file to get stream information.
    
    Args:
        media_path: Path to media file (video or audio)
        
    Returns:
        Dictionary containing stream metadata
        
    Raises:
        SyncEngineError: If media cannot be probed
    """
    if ffmpeg is None:
        raise SyncEngineError("ffmpeg-python is not available. Install with: pip install ffmpeg-python")
        
    try:
        probe = ffmpeg.probe(media_path)
        
        streams = {
            'video': [],
            'audio': [],
            'duration': 0.0,
            'format': probe.get('format', {})
        }
        
        for stream in probe['streams']:
            if stream['codec_type'] == 'video':
                streams['video'].append({
                    'index': stream['index'],
                    'codec': stream['codec_name'],
                    'width': int(stream.get('width', 0)),
                    'height': int(stream.get('height', 0)),
                    'frame_rate': eval(stream.get('r_frame_rate', '30/1')),
                    'duration': float(stream.get('duration', 0))
                })
            elif stream['codec_type'] == 'audio':
                streams['audio'].append({
                    'index': stream['index'],
                    'codec': stream['codec_name'],
                    'sample_rate': int(stream.get('sample_rate', 0)),
                    'channels': int(stream.get('channels', 0)),
                    'duration': float(stream.get('duration', 0))
                })
        
        # Get overall duration
        if 'duration' in probe['format']:
            streams['duration'] = float(probe['format']['duration'])
        elif streams['video'] and streams['video'][0]['duration'] > 0:
            streams['duration'] = streams['video'][0]['duration']
        elif streams['audio'] and streams['audio'][0]['duration'] > 0:
            streams['duration'] = streams['audio'][0]['duration']
        
        return streams
        
    except ffmpeg.Error as e:
        raise SyncEngineError(f"Failed to probe media {media_path}: {e.stderr.decode()}")
    except Exception as e:
        raise SyncEngineError(f"Unexpected error probing media {media_path}: {str(e)}")


def validate_sync_inputs(video_path: str, audio_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate and probe input video and audio files.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to input audio file
        
    Returns:
        Tuple of (video_streams, audio_streams) metadata
        
    Raises:
        SyncEngineError: If inputs are invalid
    """
    try:
        # Check if files exist
        if not Path(video_path).exists():
            raise SyncEngineError(f"Video file not found: {video_path}")
        if not Path(audio_path).exists():
            raise SyncEngineError(f"Audio file not found: {audio_path}")
        
        # Probe both files
        video_streams = probe_media_streams(video_path)
        audio_streams = probe_media_streams(audio_path)
        
        # Validate video has video stream
        if not video_streams['video']:
            raise SyncEngineError(f"No video stream found in {video_path}")
        
        # Validate audio has audio stream
        if not audio_streams['audio']:
            raise SyncEngineError(f"No audio stream found in {audio_path}")
        
        logger.info(f"Video: {video_streams['duration']:.2f}s, "
                   f"Audio: {audio_streams['duration']:.2f}s")
        
        return video_streams, audio_streams
        
    except Exception as e:
        if isinstance(e, SyncEngineError):
            raise
        raise SyncEngineError(f"Input validation failed: {str(e)}")


def create_basic_sync_filter(video_input, 
                           audio_input,
                           sync_offset: float = 0.0):
    """
    Create basic synchronization filter graph.
    
    Args:
        video_input: FFmpeg video input stream
        audio_input: FFmpeg audio input stream  
        sync_offset: Initial sync offset in seconds
        
    Returns:
        FFmpeg filter graph node
    """
    try:
        # Apply timing offset if needed
        if abs(sync_offset) > 0.01:  # Only apply if > 10ms
            if sync_offset > 0:
                # Video leads audio - delay audio
                delay_ms = int(sync_offset * 1000)
                audio_stream = audio_input.filter('adelay', f"{delay_ms}|{delay_ms}")
                video_stream = video_input
                logger.debug(f"Applied {delay_ms}ms audio delay for sync offset {sync_offset:.3f}s")
            else:
                # Audio leads video - advance audio or delay video start
                # Use setpts to adjust video timing
                pts_offset = abs(sync_offset)
                video_stream = video_input.filter('setpts', f'PTS+{pts_offset}/TB')
                audio_stream = audio_input
                logger.debug(f"Applied {pts_offset:.3f}s video PTS offset for sync offset {sync_offset:.3f}s")
        else:
            video_stream = video_input
            audio_stream = audio_input
        
        return video_stream, audio_stream
        
    except Exception as e:
        raise SyncEngineError(f"Failed to create basic sync filter: {str(e)}")


def create_complex_sync_filter(video_input,
                             audio_input,
                             sync_points: List[SyncPoint],
                             timing_corrections: TimingCorrection):
    """
    Create complex synchronization filter graph with multiple correction points.
    
    Args:
        video_input: FFmpeg video input stream
        audio_input: FFmpeg audio input stream
        sync_points: List of synchronization points
        timing_corrections: Timing correction metadata
        
    Returns:
        Tuple of (video_stream, audio_stream) with sync corrections applied
        
    Raises:
        SyncEngineError: If complex filter cannot be created
    """
    try:
        if not sync_points:
            raise SyncEngineError("No sync points provided for complex filter")
        
        video_stream = video_input
        audio_stream = audio_input
        
        # Apply global Keynote delay compensation first
        keynote_delay = timing_corrections.keynote_delay_applied
        if abs(keynote_delay) > 0.01:
            if keynote_delay > 0:
                # Apply audio delay for Keynote compensation
                delay_ms = int(abs(keynote_delay) * 1000)
                audio_stream = audio_stream.filter('adelay', f"{delay_ms}|{delay_ms}")
                logger.info(f"Applied {keynote_delay:.2f}s Keynote delay compensation")
            else:
                # Advance audio/delay video
                video_stream = video_stream.filter('setpts', f'PTS+{abs(keynote_delay)}/TB')
                logger.info(f"Applied {abs(keynote_delay):.2f}s video PTS offset for Keynote compensation")
        
        # Apply progressive timing corrections for large offsets
        correction_count = 0
        for point in timing_corrections.correction_points:
            if abs(point.sync_offset) > 0.05:  # Only correct significant offsets > 50ms
                correction_count += 1
                
                # For progressive correction, apply a fraction of the offset
                # to avoid sudden timing jumps
                correction_factor = min(0.5, 1.0 / correction_count)
                correction_amount = point.sync_offset * correction_factor
                
                if correction_amount > 0:
                    # Add audio delay
                    delay_ms = int(correction_amount * 1000)
                    audio_stream = audio_stream.filter('adelay', f"{delay_ms}|{delay_ms}")
                else:
                    # Adjust video timing slightly
                    pts_offset = abs(correction_amount)
                    video_stream = video_stream.filter('setpts', f'PTS+{pts_offset}/TB')
                
                logger.debug(f"Applied progressive correction {correction_amount:.3f}s at scene {point.scene_index}")
        
        # Handle crossfade compensations
        for comp in timing_corrections.crossfade_compensations:
            compensation = comp['compensation']
            if compensation > 0.01:  # Only apply if > 10ms
                if comp['direction'] == 'delay_audio':
                    delay_ms = int(compensation * 1000)
                    audio_stream = audio_stream.filter('adelay', f"{delay_ms}|{delay_ms}")
                    logger.debug(f"Applied crossfade audio delay {compensation:.3f}s at scene {comp['scene_index']}")
        
        logger.info(f"Created complex sync filter with {correction_count} timing corrections")
        return video_stream, audio_stream
        
    except Exception as e:
        raise SyncEngineError(f"Failed to create complex sync filter: {str(e)}")


def apply_timing_corrections(video_stream,
                           audio_stream,
                           timing_corrections: TimingCorrection):
    """
    Apply timing corrections to synchronized streams.
    
    Args:
        video_stream: Video stream with basic sync applied
        audio_stream: Audio stream with basic sync applied
        timing_corrections: Correction metadata to apply
        
    Returns:
        Tuple of (corrected_video_stream, corrected_audio_stream)
    """
    try:
        corrected_video = video_stream
        corrected_audio = audio_stream
        
        # Apply drift correction if total drift is significant
        total_drift = timing_corrections.total_drift
        if abs(total_drift) > 0.1:  # 100ms threshold
            # Apply gradual drift correction across the duration
            drift_correction = min(abs(total_drift), 0.05)  # Max 50ms correction
            
            if total_drift > 0:
                # Audio is lagging, delay it slightly less
                delay_ms = int(drift_correction * 1000)
                corrected_audio = corrected_audio.filter('adelay', f"{delay_ms}|{delay_ms}")
                logger.info(f"Applied drift correction: {drift_correction:.3f}s audio delay")
            else:
                # Audio is ahead, adjust video timing
                corrected_video = corrected_video.filter('setpts', f'PTS+{drift_correction}/TB')
                logger.info(f"Applied drift correction: {drift_correction:.3f}s video PTS offset")
        
        return corrected_video, corrected_audio
        
    except Exception as e:
        raise SyncEngineError(f"Failed to apply timing corrections: {str(e)}")


def generate_final_video(video_stream,
                        audio_stream,
                        output_path: str,
                        video_codec: str = 'copy',
                        audio_codec: str = 'aac',
                        quality_preset: str = 'medium') -> str:
    """
    Generate final synchronized video file.
    
    Args:
        video_stream: Final synchronized video stream
        audio_stream: Final synchronized audio stream
        output_path: Path for output video file
        video_codec: Video codec ('copy' to preserve original)
        audio_codec: Audio codec for output
        quality_preset: FFmpeg quality preset
        
    Returns:
        Path to generated video file
        
    Raises:
        SyncEngineError: If video generation fails
    """
    try:
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output configuration
        output_args = {
            'vcodec': video_codec,
            'acodec': audio_codec,
        }
        
        # Add quality settings if not copying video
        if video_codec != 'copy':
            output_args['preset'] = quality_preset
            output_args['crf'] = 23  # Good quality default
        
        # Add audio settings for AAC
        if audio_codec == 'aac':
            output_args['audio_bitrate'] = '128k'
            output_args['ar'] = 44100  # Sample rate
        
        # Create output
        output = ffmpeg.output(
            video_stream, 
            audio_stream,
            output_path,
            **output_args
        )
        
        # Run FFmpeg
        logger.info(f"Generating synchronized video: {output_path}")
        ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        # Verify output file was created
        if not Path(output_path).exists():
            raise SyncEngineError(f"Output video file was not created: {output_path}")
        
        output_size = Path(output_path).stat().st_size
        logger.info(f"Successfully generated synchronized video: {output_path} ({output_size} bytes)")
        
        return output_path
        
    except ffmpeg.Error as e:
        error_msg = f"FFmpeg error generating video: {e.stderr.decode()}"
        logger.error(error_msg)
        raise SyncEngineError(error_msg)
    except Exception as e:
        raise SyncEngineError(f"Failed to generate final video: {str(e)}")


def synchronize_video_audio(video_path: str,
                          audio_path: str,
                          output_path: str,
                          sync_points: Optional[List[SyncPoint]] = None,
                          timing_corrections: Optional[TimingCorrection] = None,
                          sync_config: Optional[Dict[str, Any]] = None) -> str:
    """
    Main function to synchronize video and audio files.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to input audio file
        output_path: Path for output synchronized video
        sync_points: List of synchronization points (optional)
        timing_corrections: Timing correction data (optional)
        sync_config: Synchronization configuration (optional)
        
    Returns:
        Path to generated synchronized video
        
    Raises:
        SyncEngineError: If synchronization fails
    """
    try:
        # Set default config
        if sync_config is None:
            sync_config = {
                'video_codec': 'copy',
                'audio_codec': 'aac',
                'quality_preset': 'medium'
            }
        
        # Validate inputs
        logger.info(f"Starting video-audio synchronization")
        logger.info(f"Video: {video_path}")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Output: {output_path}")
        
        video_streams, audio_streams = validate_sync_inputs(video_path, audio_path)
        
        # Create FFmpeg input streams
        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(audio_path)
        
        # Determine synchronization approach
        if sync_points and timing_corrections and len(sync_points) > 1:
            # Complex multi-point synchronization
            logger.info("Using complex multi-point synchronization")
            video_stream, audio_stream = create_complex_sync_filter(
                video_input, audio_input, sync_points, timing_corrections
            )
            
            # Apply additional timing corrections
            video_stream, audio_stream = apply_timing_corrections(
                video_stream, audio_stream, timing_corrections
            )
            
        elif sync_points and len(sync_points) > 0:
            # Basic single-point synchronization
            initial_offset = sync_points[0].sync_offset
            logger.info(f"Using basic synchronization with {initial_offset:.3f}s offset")
            video_stream, audio_stream = create_basic_sync_filter(
                video_input, audio_input, initial_offset
            )
            
        else:
            # No sync correction - direct combination
            logger.info("No sync corrections - combining streams directly")
            video_stream = video_input
            audio_stream = audio_input
        
        # Generate final video
        output_video_path = generate_final_video(
            video_stream=video_stream,
            audio_stream=audio_stream,
            output_path=output_path,
            video_codec=sync_config.get('video_codec', 'copy'),
            audio_codec=sync_config.get('audio_codec', 'aac'),
            quality_preset=sync_config.get('quality_preset', 'medium')
        )
        
        logger.info(f"Video synchronization completed successfully: {output_video_path}")
        return output_video_path
        
    except Exception as e:
        error_msg = f"Video synchronization failed: {str(e)}"
        logger.error(error_msg)
        raise SyncEngineError(error_msg)


def create_preview_clips(video_path: str,
                        audio_path: str,
                        sync_points: List[SyncPoint],
                        output_dir: str,
                        clip_duration: float = 10.0) -> List[str]:
    """
    Create preview clips at sync points for manual verification.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to input audio file
        sync_points: List of synchronization points
        output_dir: Directory for preview clips
        clip_duration: Duration of each preview clip in seconds
        
    Returns:
        List of paths to generated preview clips
        
    Raises:
        SyncEngineError: If preview generation fails
    """
    try:
        preview_clips = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, point in enumerate(sync_points[:5]):  # Limit to first 5 points
            # Calculate clip timing
            start_time = max(0, point.video_timestamp - 2.0)  # Start 2s before sync point
            
            # Create clip filename
            clip_filename = f"sync_preview_{i:02d}_scene_{point.scene_index}.mp4"
            clip_path = output_path / clip_filename
            
            try:
                # Create video input with timing
                video_input = ffmpeg.input(video_path, ss=start_time, t=clip_duration)
                
                # Create audio input with corresponding timing
                audio_start = max(0, point.audio_timestamp - 2.0 + point.sync_offset)
                audio_input = ffmpeg.input(audio_path, ss=audio_start, t=clip_duration)
                
                # Generate preview clip
                output = ffmpeg.output(
                    video_input,
                    audio_input,
                    str(clip_path),
                    vcodec='libx264',
                    acodec='aac',
                    preset='ultrafast',
                    crf=28
                )
                
                ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                
                if clip_path.exists():
                    preview_clips.append(str(clip_path))
                    logger.debug(f"Created preview clip: {clip_filename}")
                
            except ffmpeg.Error as e:
                logger.warning(f"Failed to create preview clip {i}: {e.stderr.decode()}")
                continue
        
        logger.info(f"Created {len(preview_clips)} preview clips in {output_dir}")
        return preview_clips
        
    except Exception as e:
        raise SyncEngineError(f"Failed to create preview clips: {str(e)}")