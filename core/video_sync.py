"""
Video synchronization module for AutoVid pipeline.

This module provides timing coordination between video analysis results 
and audio splice manifests to enable precise video-audio synchronization.
It handles Keynote delay compensation and crossfade boundary alignment.
"""

import os
import json
import logging
from datetime import datetime

try:
    import yaml
except ImportError:
    yaml = None
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class VideoSyncError(Exception):
    """Custom exception for video synchronization errors."""
    pass


@dataclass
class SyncPoint:
    """Represents a synchronization point between video and audio."""
    scene_index: int
    video_timestamp: float
    audio_timestamp: float
    sync_offset: float
    confidence: float
    scene_duration: Optional[float] = None
    audio_duration: Optional[float] = None


@dataclass
class TimingCorrection:
    """Represents timing correction metadata."""
    total_drift: float
    max_offset: float
    correction_points: List[SyncPoint]
    keynote_delay_applied: float
    crossfade_compensations: List[Dict[str, float]]


def load_video_analysis_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Load video analysis manifest containing scene transitions and timing data.
    
    Args:
        manifest_path: Path to video analysis manifest JSON file
        
    Returns:
        Dictionary containing video analysis data
        
    Raises:
        VideoSyncError: If manifest cannot be loaded or is invalid
    """
    try:
        if not Path(manifest_path).exists():
            raise VideoSyncError(f"Video analysis manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Validate required fields
        required_fields = ['video_analysis']
        if not all(field in manifest for field in required_fields):
            raise VideoSyncError(f"Invalid video analysis manifest: missing required fields")
        
        video_analysis = manifest['video_analysis']
        required_video_fields = ['scene_transitions', 'video_info', 'keynote_delay_compensation']
        if not all(field in video_analysis for field in required_video_fields):
            raise VideoSyncError(f"Invalid video analysis data: missing required fields")
        
        logger.info(f"Loaded video analysis manifest: {len(video_analysis['scene_transitions'])} scenes")
        return manifest
        
    except json.JSONDecodeError as e:
        raise VideoSyncError(f"Failed to parse video analysis manifest: {str(e)}")
    except Exception as e:
        raise VideoSyncError(f"Error loading video analysis manifest: {str(e)}")


def load_audio_splice_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Load audio splice manifest containing final audio timing data.
    
    Args:
        manifest_path: Path to audio splice manifest JSON file
        
    Returns:
        Dictionary containing audio splice data
        
    Raises:
        VideoSyncError: If manifest cannot be loaded or is invalid
    """
    try:
        if not Path(manifest_path).exists():
            raise VideoSyncError(f"Audio splice manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Validate structure - audio splice manifest should have timing info
        if 'audio_timing' not in manifest and 'chunks' not in manifest:
            raise VideoSyncError(f"Invalid audio splice manifest: no timing data found")
        
        logger.info(f"Loaded audio splice manifest: {manifest_path}")
        return manifest
        
    except json.JSONDecodeError as e:
        raise VideoSyncError(f"Failed to parse audio splice manifest: {str(e)}")
    except Exception as e:
        raise VideoSyncError(f"Error loading audio splice manifest: {str(e)}")


def load_timing_manifests(video_manifest_path: str, 
                         audio_manifest_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load both video analysis and audio splice manifests.
    
    Args:
        video_manifest_path: Path to video analysis manifest
        audio_manifest_path: Path to audio splice manifest
        
    Returns:
        Tuple of (video_manifest, audio_manifest)
        
    Raises:
        VideoSyncError: If either manifest cannot be loaded
    """
    try:
        video_manifest = load_video_analysis_manifest(video_manifest_path)
        audio_manifest = load_audio_splice_manifest(audio_manifest_path)
        
        logger.info("Successfully loaded both timing manifests")
        return video_manifest, audio_manifest
        
    except Exception as e:
        raise VideoSyncError(f"Failed to load timing manifests: {str(e)}")


def extract_scene_timings(video_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract scene timing information from video analysis manifest.
    
    Args:
        video_manifest: Video analysis manifest data
        
    Returns:
        List of scene timing dictionaries with compensated timestamps
    """
    video_analysis = video_manifest['video_analysis']
    scene_transitions = video_analysis['scene_transitions']
    keynote_delay = video_analysis.get('keynote_delay_compensation', 1.0)
    
    # Scene transitions should already have Keynote delay compensation applied
    # but verify and log the compensation being used
    logger.info(f"Using scene transitions with {keynote_delay}s Keynote delay compensation")
    
    return scene_transitions


def extract_audio_chunk_timings(audio_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract audio chunk timing information from splice manifest.
    
    Args:
        audio_manifest: Audio splice manifest data
        
    Returns:
        List of audio chunk timing dictionaries
    """
    # Handle different possible manifest structures
    if 'audio_timing' in audio_manifest:
        timing_data = audio_manifest['audio_timing']
        if 'chunks' in timing_data:
            chunks = timing_data['chunks']
        else:
            chunks = []
    elif 'chunks' in audio_manifest:
        chunks = audio_manifest['chunks']
    else:
        # Fallback: create chunks from available data
        chunks = []
        logger.warning("No explicit chunk timing data found in audio manifest")
    
    logger.info(f"Extracted {len(chunks)} audio chunk timings")
    return chunks


def calculate_sync_points(scene_transitions: List[Dict[str, Any]], 
                         audio_chunks: List[Dict[str, Any]],
                         crossfade_duration: float = 0.1) -> List[SyncPoint]:
    """
    Calculate synchronization points between video scenes and audio chunks.
    
    Args:
        scene_transitions: List of scene transition data from video analysis
        audio_chunks: List of audio chunk timing data from splice manifest
        crossfade_duration: Duration of crossfade between audio chunks
        
    Returns:
        List of SyncPoint objects representing video-audio alignment
        
    Raises:
        VideoSyncError: If sync points cannot be calculated
    """
    try:
        sync_points = []
        
        # Handle case where we have different numbers of scenes vs audio chunks
        min_count = min(len(scene_transitions), len(audio_chunks))
        if len(scene_transitions) != len(audio_chunks):
            logger.warning(f"Scene count ({len(scene_transitions)}) != audio chunk count ({len(audio_chunks)})")
            logger.warning(f"Will sync first {min_count} scenes/chunks")
        
        current_audio_time = 0.0
        
        for i in range(min_count):
            scene = scene_transitions[i]
            
            # Video timestamp is already delay-compensated
            video_timestamp = scene['timestamp']
            
            # Audio timestamp accounts for crossfade overlap
            if i > 0:
                # Subtract half crossfade duration to account for overlap
                current_audio_time -= (crossfade_duration / 2.0)
            
            # Calculate sync offset
            sync_offset = video_timestamp - current_audio_time
            
            # Determine confidence based on various factors
            confidence = calculate_sync_confidence(scene, i, sync_offset)
            
            # Get scene duration for next iteration
            scene_duration = None
            if i + 1 < len(scene_transitions):
                scene_duration = scene_transitions[i + 1]['timestamp'] - scene['timestamp']
            
            # Get audio duration from chunk data
            audio_duration = None
            if len(audio_chunks) > i and 'duration' in audio_chunks[i]:
                audio_duration = audio_chunks[i]['duration']
                current_audio_time += audio_duration
            elif scene_duration:
                # Fallback: use scene duration
                audio_duration = scene_duration
                current_audio_time += scene_duration
            else:
                # Last resort: estimate based on remaining chunks
                estimated_duration = 3.0  # Default 3 seconds per chunk
                audio_duration = estimated_duration
                current_audio_time += estimated_duration
            
            sync_point = SyncPoint(
                scene_index=i,
                video_timestamp=video_timestamp,
                audio_timestamp=current_audio_time - audio_duration,
                sync_offset=sync_offset,
                confidence=confidence,
                scene_duration=scene_duration,
                audio_duration=audio_duration
            )
            
            sync_points.append(sync_point)
            
            logger.debug(f"Scene {i}: video={video_timestamp:.2f}s, "
                        f"audio={sync_point.audio_timestamp:.2f}s, "
                        f"offset={sync_offset:.2f}s")
        
        logger.info(f"Calculated {len(sync_points)} sync points")
        return sync_points
        
    except Exception as e:
        raise VideoSyncError(f"Failed to calculate sync points: {str(e)}")


def calculate_sync_confidence(scene: Dict[str, Any], scene_index: int, sync_offset: float) -> float:
    """
    Calculate confidence score for a sync point based on various factors.
    
    Args:
        scene: Scene transition data
        scene_index: Index of the scene
        sync_offset: Calculated sync offset in seconds
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    confidence = 1.0
    
    # Reduce confidence for large sync offsets
    offset_penalty = min(abs(sync_offset) * 0.1, 0.5)
    confidence -= offset_penalty
    
    # First and last scenes are typically more reliable
    if scene_index == 0:
        confidence += 0.1
    
    # Check if scene has validation data
    if scene.get('confidence', 0) > 0.5:
        confidence += 0.1
    
    # Clamp to valid range
    confidence = max(0.0, min(1.0, confidence))
    
    return confidence


def generate_timing_corrections(sync_points: List[SyncPoint], 
                              video_duration: float,
                              tolerance: float = 0.1) -> TimingCorrection:
    """
    Generate timing corrections to minimize sync drift across video.
    
    Args:
        sync_points: List of calculated sync points
        video_duration: Total video duration in seconds
        tolerance: Acceptable sync offset tolerance in seconds
        
    Returns:
        TimingCorrection object with correction metadata
        
    Raises:
        VideoSyncError: If corrections cannot be generated
    """
    try:
        if not sync_points:
            raise VideoSyncError("No sync points provided for correction calculation")
        
        # Calculate drift statistics
        offsets = [point.sync_offset for point in sync_points]
        total_drift = sum(offsets)
        max_offset = max(abs(offset) for offset in offsets)
        
        # Identify points that need correction
        correction_points = []
        crossfade_compensations = []
        
        for point in sync_points:
            if abs(point.sync_offset) > tolerance:
                correction_points.append(point)
                
                # Calculate crossfade compensation if needed
                if point.scene_index > 0:  # Skip first scene
                    crossfade_comp = {
                        'scene_index': point.scene_index,
                        'compensation': min(abs(point.sync_offset), 0.05),  # Max 50ms
                        'direction': 'delay_audio' if point.sync_offset > 0 else 'advance_audio'
                    }
                    crossfade_compensations.append(crossfade_comp)
        
        # Apply progressive correction to reduce drift accumulation
        keynote_delay_applied = sync_points[0].video_timestamp - sync_points[0].audio_timestamp if sync_points else 0.0
        
        correction = TimingCorrection(
            total_drift=total_drift,
            max_offset=max_offset,
            correction_points=correction_points,
            keynote_delay_applied=keynote_delay_applied,
            crossfade_compensations=crossfade_compensations
        )
        
        logger.info(f"Generated timing corrections: {len(correction_points)} points need correction")
        logger.info(f"Total drift: {total_drift:.2f}s, Max offset: {max_offset:.2f}s")
        
        return correction
        
    except Exception as e:
        raise VideoSyncError(f"Failed to generate timing corrections: {str(e)}")


def compensate_keynote_delay(sync_points: List[SyncPoint], 
                           additional_delay: float = 0.0) -> List[SyncPoint]:
    """
    Apply additional Keynote delay compensation if needed.
    
    Args:
        sync_points: List of sync points to adjust
        additional_delay: Additional delay compensation in seconds
        
    Returns:
        List of sync points with additional delay compensation applied
    """
    if additional_delay == 0.0:
        return sync_points
    
    compensated_points = []
    
    for point in sync_points:
        compensated_point = SyncPoint(
            scene_index=point.scene_index,
            video_timestamp=point.video_timestamp - additional_delay,
            audio_timestamp=point.audio_timestamp,
            sync_offset=point.sync_offset + additional_delay,
            confidence=point.confidence * 0.9,  # Slight confidence reduction
            scene_duration=point.scene_duration,
            audio_duration=point.audio_duration
        )
        compensated_points.append(compensated_point)
    
    logger.info(f"Applied additional {additional_delay}s delay compensation")
    return compensated_points


def validate_scene_audio_mapping(video_manifest: Dict[str, Any], 
                               audio_manifest: Dict[str, Any],
                               sync_points: List[SyncPoint]) -> Dict[str, Any]:
    """
    Validate the mapping between video scenes and audio segments.
    
    Args:
        video_manifest: Video analysis manifest
        audio_manifest: Audio splice manifest
        sync_points: List of calculated sync points
        
    Returns:
        Validation result dictionary
    """
    try:
        validation_result = {
            'status': 'PASS',
            'warnings': [],
            'errors': [],
            'statistics': {
                'total_sync_points': len(sync_points),
                'high_confidence_points': 0,
                'large_offset_points': 0,
                'avg_confidence': 0.0,
                'avg_offset': 0.0
            }
        }
        
        if not sync_points:
            validation_result['status'] = 'FAIL'
            validation_result['errors'].append("No sync points calculated")
            return validation_result
        
        # Calculate statistics
        confidences = [point.confidence for point in sync_points]
        offsets = [abs(point.sync_offset) for point in sync_points]
        
        validation_result['statistics']['high_confidence_points'] = sum(1 for c in confidences if c >= 0.8)
        validation_result['statistics']['large_offset_points'] = sum(1 for o in offsets if o > 0.2)
        validation_result['statistics']['avg_confidence'] = sum(confidences) / len(confidences)
        validation_result['statistics']['avg_offset'] = sum(offsets) / len(offsets)
        
        # Generate warnings
        if validation_result['statistics']['avg_confidence'] < 0.6:
            validation_result['warnings'].append("Low average sync confidence")
            validation_result['status'] = 'WARN'
        
        if validation_result['statistics']['avg_offset'] > 0.15:
            validation_result['warnings'].append("High average sync offset")
            validation_result['status'] = 'WARN'
        
        if validation_result['statistics']['large_offset_points'] > len(sync_points) * 0.3:
            validation_result['warnings'].append("Many points with large sync offsets")
            validation_result['status'] = 'WARN'
        
        # Check scene vs audio count mismatch
        video_scenes = len(video_manifest['video_analysis']['scene_transitions'])
        audio_chunks = len(audio_manifest.get('chunks', []))
        
        if video_scenes != audio_chunks:
            validation_result['warnings'].append(
                f"Scene count ({video_scenes}) != audio chunk count ({audio_chunks})"
            )
            if validation_result['status'] == 'PASS':
                validation_result['status'] = 'WARN'
        
        logger.info(f"Scene-audio mapping validation: {validation_result['status']}")
        return validation_result
        
    except Exception as e:
        return {
            'status': 'FAIL',
            'errors': [f"Validation failed: {str(e)}"],
            'warnings': [],
            'statistics': {}
        }


def get_video_sync_config(config_path: str = "config/pipeline.yaml") -> Dict[str, Any]:
    """
    Extract video synchronization configuration from pipeline config.
    
    Args:
        config_path: Path to pipeline configuration file
        
    Returns:
        Dictionary of video sync configuration with defaults
    """
    project_root = Path(__file__).parent.parent
    config_file = project_root / config_path
    
    defaults = {
        'keynote_delay': 1.0,
        'sync_tolerance': 0.1,
        'crossfade_handling': 'smooth',
        'timing_validation': 'strict',
        'output_format': 'mp4',
        'quality_preset': 'medium',
        'audio_codec': 'aac',
        'video_codec': 'copy'
    }
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}, using defaults")
        return defaults
    
    if yaml is None:
        logger.warning("PyYAML not available, using default config")
        return defaults
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        sync_config = config.get("video_sync", {})
        
        # Merge defaults with config
        for key, value in defaults.items():
            if key not in sync_config:
                sync_config[key] = value
        
        return sync_config
        
    except Exception as e:
        logger.warning(f"Failed to load config: {str(e)}, using defaults")
        return defaults


def create_sync_manifest(video_manifest_path: str,
                        audio_manifest_path: str,
                        sync_points: List[SyncPoint],
                        timing_corrections: TimingCorrection,
                        validation_result: Dict[str, Any],
                        output_video_path: str) -> Dict[str, Any]:
    """
    Create comprehensive synchronization manifest.
    
    Args:
        video_manifest_path: Path to input video analysis manifest
        audio_manifest_path: Path to input audio splice manifest
        sync_points: List of calculated sync points
        timing_corrections: Applied timing corrections
        validation_result: Scene-audio mapping validation results
        output_video_path: Path to output synchronized video
        
    Returns:
        Complete synchronization manifest dictionary
    """
    manifest = {
        'video_sync': {
            'input_manifests': {
                'video_analysis': str(Path(video_manifest_path).resolve()),
                'audio_splice': str(Path(audio_manifest_path).resolve())
            },
            'output_video': str(Path(output_video_path).resolve()),
            'sync_timestamp': datetime.now().isoformat(),
            'sync_points': [
                {
                    'scene_index': point.scene_index,
                    'video_timestamp': point.video_timestamp,
                    'audio_timestamp': point.audio_timestamp,
                    'sync_offset': point.sync_offset,
                    'confidence': point.confidence,
                    'scene_duration': point.scene_duration,
                    'audio_duration': point.audio_duration
                }
                for point in sync_points
            ],
            'timing_corrections': {
                'total_drift': timing_corrections.total_drift,
                'max_offset': timing_corrections.max_offset,
                'keynote_delay_applied': timing_corrections.keynote_delay_applied,
                'correction_count': len(timing_corrections.correction_points),
                'crossfade_compensations': timing_corrections.crossfade_compensations
            },
            'validation': validation_result,
            'statistics': {
                'total_sync_points': len(sync_points),
                'avg_confidence': sum(p.confidence for p in sync_points) / len(sync_points) if sync_points else 0.0,
                'avg_sync_offset': sum(abs(p.sync_offset) for p in sync_points) / len(sync_points) if sync_points else 0.0
            }
        }
    }
    
    return manifest