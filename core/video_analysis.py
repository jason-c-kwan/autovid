"""
Video analysis module for AutoVid pipeline.

This module provides functions for analyzing Keynote video exports to detect
scene transitions and movement ranges for synchronization with audio narration.
It handles the 1-second delay inherent in Keynote exports and provides robust
error handling for various video formats.
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from . import config_loader

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

logger = logging.getLogger(__name__)


class VideoAnalysisError(Exception):
    """Custom exception for video analysis errors."""
    pass


def probe_video_info(video_path: str) -> Dict[str, Any]:
    """
    Extract video metadata using ffmpeg probe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video metadata
        
    Raises:
        VideoAnalysisError: If video cannot be probed
    """
    if ffmpeg is None:
        raise VideoAnalysisError("ffmpeg-python is not available. Install with: pip install ffmpeg-python")
        
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] 
                           if stream['codec_type'] == 'video'), None)
        
        if not video_stream:
            raise VideoAnalysisError(f"No video stream found in {video_path}")
            
        return {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'duration': float(video_stream.get('duration', 0)),
            'frame_rate': eval(video_stream.get('r_frame_rate', '30/1')),
            'frame_count': int(video_stream.get('nb_frames', 0)),
            'codec': video_stream.get('codec_name'),
            'pixel_format': video_stream.get('pix_fmt'),
            'bitrate': int(video_stream.get('bit_rate', 0))
        }
    except ffmpeg.Error as e:
        raise VideoAnalysisError(f"Failed to probe video {video_path}: {e.stderr.decode()}")
    except Exception as e:
        raise VideoAnalysisError(f"Unexpected error probing video {video_path}: {str(e)}")


def detect_scene_changes(video_path: str, threshold: float = 0.4, 
                        min_scene_duration: float = 0.5) -> List[Dict[str, Any]]:
    """
    Detect scene transitions in video using FFmpeg scene detection with enhanced accuracy.
    
    Args:
        video_path: Path to the video file
        threshold: Scene detection sensitivity (0.0-1.0, higher = less sensitive)
        min_scene_duration: Minimum duration between scenes in seconds
        
    Returns:
        List of scene transition dictionaries with timestamps and metadata
        
    Raises:
        VideoAnalysisError: If scene detection fails
    """
    try:
        # Get video info for context
        video_info = probe_video_info(video_path)
        frame_rate = video_info['frame_rate']
        duration = video_info['duration']
        
        scenes = []
        
        # Enhanced scene detection using multiple methods for better accuracy
        try:
            # Method 1: Use scene filter with detailed output (optimized for performance)
            result = (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=1)  # Reduce framerate aggressively for 4K video
                .filter('select', f'gt(scene,{threshold})')
                .filter('showinfo')
                .output('pipe:', format='null')
                .global_args('-loglevel', 'info', '-nostats')
                .run(capture_stdout=True, capture_stderr=True)
            )
            stderr_output = result[1].decode()
            
            # Parse enhanced scene detection output
            scene_count = 0
            last_timestamp = -1
            
            for line in stderr_output.split('\n'):
                if 'showinfo' in line and 'pts_time:' in line:
                    try:
                        # Extract timestamp more robustly
                        import re
                        timestamp_match = re.search(r'pts_time:(\d+(?:\.\d+)?)', line)
                        if not timestamp_match:
                            continue
                            
                        timestamp = float(timestamp_match.group(1))
                        frame_number = int(timestamp * frame_rate)
                        
                        # Apply minimum scene duration filter with enhanced validation
                        if (timestamp >= min_scene_duration and 
                            (last_timestamp == -1 or timestamp - last_timestamp >= min_scene_duration)):
                            
                            scene_count += 1
                            last_timestamp = timestamp
                            
                            # Extract additional scene information if available
                            scene_score = None
                            scene_match = re.search(r'scene:(\d+(?:\.\d+)?)', line)
                            if scene_match:
                                scene_score = float(scene_match.group(1))
                            
                            scenes.append({
                                'timestamp': timestamp,
                                'frame_number': frame_number,
                                'confidence': scene_score or threshold,
                                'slide_number': scene_count,
                                'scene_score': scene_score,
                                'transition_type': 'slide_change'
                            })
                            
                    except (ValueError, IndexError, AttributeError) as e:
                        logger.debug(f"Failed to parse scene detection line: {line}, error: {e}")
                        continue
                        
        except ffmpeg.Error as e:
            logger.warning(f"Primary scene detection failed: {e.stderr.decode()}")
            # Fallback to simpler detection method
            
        # If no scenes detected or very few, use fallback method
        if len(scenes) < 2:
            logger.info("Using fallback scene detection method")
            scenes = _detect_scenes_fallback(video_path, threshold, min_scene_duration, frame_rate, duration)
        
        # Ensure we have at least one scene at the beginning
        if not scenes or scenes[0]['timestamp'] > 0.5:
            scenes.insert(0, {
                'timestamp': 0.0,
                'frame_number': 0,
                'confidence': 1.0,
                'slide_number': 1,
                'scene_score': None,
                'transition_type': 'presentation_start'
            })
            # Renumber subsequent slides
            for i, scene in enumerate(scenes[1:], 2):
                scene['slide_number'] = i
                
        logger.info(f"Detected {len(scenes)} scene transitions in {video_path}")
        return scenes
        
    except Exception as e:
        raise VideoAnalysisError(f"Scene detection failed for {video_path}: {str(e)}")


def _detect_scenes_fallback(video_path: str, threshold: float, min_scene_duration: float, 
                           frame_rate: float, duration: float) -> List[Dict[str, Any]]:
    """
    Fallback scene detection method for challenging videos.
    
    Args:
        video_path: Path to the video file
        threshold: Scene detection sensitivity
        min_scene_duration: Minimum duration between scenes
        frame_rate: Video frame rate
        duration: Total video duration
        
    Returns:
        List of detected scene transitions
    """
    scenes = []
    
    try:
        # Use histogram-based detection as fallback (optimized for performance)
        result = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=1)  # Reduce framerate aggressively for 4K video
            .filter('select', f'gt(scene,{threshold * 0.7})')  # Lower threshold for fallback
            .filter('metadata', mode='print')
            .output('pipe:', format='null')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        stderr_output = result[1].decode()
        scene_count = 0
        
        # Parse metadata output for scene changes
        import re
        for line in stderr_output.split('\n'):
            timestamp_match = re.search(r'frame:\s*\d+\s+pts:\s*\d+\s+pts_time:(\d+(?:\.\d+)?)', line)
            if timestamp_match:
                timestamp = float(timestamp_match.group(1))
                
                if (not scenes or timestamp - scenes[-1]['timestamp'] >= min_scene_duration):
                    scene_count += 1
                    scenes.append({
                        'timestamp': timestamp,
                        'frame_number': int(timestamp * frame_rate),
                        'confidence': threshold * 0.7,
                        'slide_number': scene_count,
                        'scene_score': None,
                        'transition_type': 'slide_change_detected'
                    })
                    
    except ffmpeg.Error:
        # Ultimate fallback: estimate scene changes based on typical presentation timing
        logger.warning("Using estimated scene transitions based on typical presentation timing")
        estimated_scenes = max(2, int(duration / 10))  # Assume ~10 seconds per slide
        for i in range(estimated_scenes):
            timestamp = (duration / estimated_scenes) * i
            scenes.append({
                'timestamp': timestamp,
                'frame_number': int(timestamp * frame_rate),
                'confidence': 0.5,  # Low confidence for estimated scenes
                'slide_number': i + 1,
                'scene_score': None,
                'transition_type': 'estimated'
            })
    
    return scenes


def extract_movement_frames(video_path: str, scene_transitions: List[Dict[str, Any]], 
                          movement_threshold: float = 0.1) -> List[Dict[str, Any]]:
    """
    Detect movement/animation ranges within each scene.
    
    Args:
        video_path: Path to the video file
        scene_transitions: List of scene transition data
        movement_threshold: Threshold for detecting significant movement
        
    Returns:
        List of movement range dictionaries
        
    Raises:
        VideoAnalysisError: If movement detection fails
    """
    try:
        video_info = probe_video_info(video_path)
        frame_rate = video_info['frame_rate']
        duration = video_info['duration']
        
        movement_ranges = []
        
        # Process each scene individually
        for i, scene in enumerate(scene_transitions):
            start_time = scene['timestamp']
            end_time = scene_transitions[i + 1]['timestamp'] if i + 1 < len(scene_transitions) else duration
            
            # Skip very short scenes
            if end_time - start_time < 0.5:
                continue
                
            try:
                # Use motion detection filter to find movement within scene
                result = (
                    ffmpeg
                    .input(video_path, ss=start_time, t=(end_time - start_time))
                    .filter('select', f'gt(scene,{movement_threshold/10})')  # Lower threshold for movement
                    .filter('showinfo')
                    .output('pipe:', format='null')
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                stderr_output = result[1].decode()
                
                # Parse movement detection within this scene
                scene_movements = []
                for line in stderr_output.split('\n'):
                    if 'showinfo' in line and 'pts_time:' in line:
                        try:
                            pts_time_start = line.find('pts_time:') + 9
                            pts_time_end = line.find(' ', pts_time_start)
                            if pts_time_end == -1:
                                pts_time_end = len(line)
                            
                            relative_timestamp = float(line[pts_time_start:pts_time_end])
                            absolute_timestamp = start_time + relative_timestamp
                            frame_number = int(absolute_timestamp * frame_rate)
                            
                            scene_movements.append({
                                'timestamp': absolute_timestamp,
                                'frame_number': frame_number
                            })
                            
                        except (ValueError, IndexError):
                            continue
                
                # Group consecutive movement frames into ranges
                if scene_movements:
                    movement_start = scene_movements[0]
                    movement_end = scene_movements[-1]
                    
                    # Only add if there's significant movement duration
                    if movement_end['timestamp'] - movement_start['timestamp'] > 0.2:
                        movement_ranges.append({
                            'start_frame': movement_start['frame_number'],
                            'end_frame': movement_end['frame_number'],
                            'start_time': movement_start['timestamp'],
                            'end_time': movement_end['timestamp'],
                            'within_slide': scene['slide_number'],
                            'duration': movement_end['timestamp'] - movement_start['timestamp']
                        })
                        
            except ffmpeg.Error as e:
                logger.warning(f"Movement detection failed for scene {i}: {e.stderr.decode()}")
                continue
        
        logger.info(f"Detected {len(movement_ranges)} movement ranges in {video_path}")
        return movement_ranges
        
    except Exception as e:
        raise VideoAnalysisError(f"Movement detection failed for {video_path}: {str(e)}")


def compensate_keynote_delay(transitions: List[Dict[str, Any]], 
                           delay_seconds: float = 1.0) -> List[Dict[str, Any]]:
    """
    Compensate for Keynote export delay by adjusting timestamps.
    
    Args:
        transitions: List of scene transition data
        delay_seconds: Keynote export delay in seconds (default 1.0)
        
    Returns:
        List of transitions with compensated timestamps
    """
    compensated = []
    
    for transition in transitions:
        compensated_transition = transition.copy()
        
        # Adjust timestamp accounting for delay
        original_timestamp = transition['timestamp']
        compensated_timestamp = max(0, original_timestamp - delay_seconds)
        
        compensated_transition['timestamp'] = compensated_timestamp
        compensated_transition['original_timestamp'] = original_timestamp
        compensated_transition['delay_compensation'] = delay_seconds
        
        # Recalculate frame number if needed
        if 'frame_rate' in transition:
            compensated_transition['frame_number'] = int(compensated_timestamp * transition['frame_rate'])
        
        compensated.append(compensated_transition)
    
    logger.info(f"Applied {delay_seconds}s delay compensation to {len(transitions)} transitions")
    return compensated


def validate_transition_count(detected_transitions: List[Dict[str, Any]], 
                            expected_cues: List[str], 
                            tolerance: int = 1) -> Dict[str, Any]:
    """
    Validate detected transitions against expected transition cues.
    
    Args:
        detected_transitions: List of detected scene transitions
        expected_cues: List of transition cues from transcript
        tolerance: Acceptable difference in transition counts
        
    Returns:
        Validation result dictionary
    """
    detected_count = len(detected_transitions)
    expected_count = len(expected_cues)
    difference = abs(detected_count - expected_count)
    
    validation_result = {
        'detected_count': detected_count,
        'expected_count': expected_count,
        'difference': difference,
        'within_tolerance': difference <= tolerance,
        'status': 'PASS' if difference <= tolerance else 'WARN',
        'message': ''
    }
    
    if difference == 0:
        validation_result['message'] = "Transition count matches exactly"
    elif difference <= tolerance:
        validation_result['message'] = f"Transition count within tolerance ({difference} difference)"
    else:
        validation_result['message'] = (
            f"Transition count mismatch: detected {detected_count}, "
            f"expected {expected_count} (difference: {difference})"
        )
    
    logger.info(f"Transition validation: {validation_result['message']}")
    return validation_result


def generate_timing_manifest(video_path: str, 
                           scene_transitions: List[Dict[str, Any]],
                           movement_ranges: List[Dict[str, Any]],
                           video_info: Dict[str, Any],
                           validation_result: Dict[str, Any],
                           keynote_delay: float = 1.0) -> Dict[str, Any]:
    """
    Generate comprehensive timing manifest for video synchronization.
    
    Args:
        video_path: Path to the analyzed video file
        scene_transitions: List of detected scene transitions
        movement_ranges: List of detected movement ranges
        video_info: Video metadata
        validation_result: Transition validation results
        keynote_delay: Applied delay compensation
        
    Returns:
        Complete timing manifest dictionary
    """
    manifest = {
        'video_analysis': {
            'input_file': str(Path(video_path).resolve()),
            'analysis_timestamp': datetime.now().isoformat(),
            'video_info': video_info,
            'scene_transitions': scene_transitions,
            'movement_ranges': movement_ranges,
            'keynote_delay_compensation': keynote_delay,
            'total_scenes': len(scene_transitions),
            'total_movements': len(movement_ranges),
            'validation': validation_result,
            'processing_settings': {
                'scene_threshold': 0.4,
                'movement_threshold': 0.1,
                'min_scene_duration': 0.5,
                'delay_compensation': keynote_delay
            }
        }
    }
    
    return manifest


def detect_keynote_scenes(
    video_path: str,
    expected_transitions: Optional[int] = None,
    config_override: Optional[Dict] = None,
    transcript_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Keynote-optimized scene detection combining multiple algorithms.
    
    This function uses three different detection methods specifically tuned for
    slide-only videos with subtle transitions and 1-second Keynote pauses.
    
    Args:
        video_path: Path to the video file
        expected_transitions: Expected number of transitions for validation
        config_override: Override configuration parameters
        transcript_path: Optional path to transcript JSON for validation against [transition] cues
        
    Returns:
        Dictionary containing detected scenes and metadata
        
    Raises:
        VideoAnalysisError: If all detection methods fail
    """
    try:
        # Load configuration with overrides
        config = config_loader.load_config()
        video_config = config.get('video_analysis', {})
        
        # Apply overrides if provided
        if config_override:
            video_config = {**video_config, **config_override}
        
        # Extract configuration parameters with defaults
        threshold = video_config.get('scene_threshold', 0.1)
        keynote_delay = video_config.get('keynote_delay', 1.0)
        presentation_mode = video_config.get('presentation_mode', True)
        min_scene_duration = video_config.get('min_scene_duration', 0.5)
        enable_multi_algorithm = video_config.get('enable_multi_algorithm', True)
        enable_static_detection = video_config.get('enable_static_detection', True)
        enable_content_analysis = video_config.get('enable_content_analysis', True)
        
        # Multi-algorithm configuration
        algorithm_weights = video_config.get('algorithm_weights', {
            'sensitive': 1.0,
            'static': 1.2,
            'content': 0.9
        })
        confidence_threshold = video_config.get('confidence_threshold', 0.5)
        
        logger.info(f"Starting Keynote-optimized scene detection for {video_path}")
        logger.info(f"Config: threshold={threshold}, keynote_delay={keynote_delay}, multi_algorithm={enable_multi_algorithm}")
        if expected_transitions is not None:
            logger.info(f"Expected transitions: {expected_transitions}")
        
        video_info = probe_video_info(video_path)
        frame_rate = video_info['frame_rate']
        duration = video_info['duration']
        
        # Use adaptive threshold detection if enabled and expected transitions are available
        enable_adaptive = video_config.get('enable_adaptive_thresholds', True)
        max_adaptive_attempts = video_config.get('max_adaptive_attempts', 3)
        
        if enable_adaptive and expected_transitions and expected_transitions > 0:
            logger.info("Using adaptive threshold detection")
            detection_results, adaptive_validation = _detect_with_adaptive_thresholds(
                video_path, video_config, expected_transitions, duration, max_adaptive_attempts
            )
        else:
            # Fallback to manual detection
            logger.info("Using manual threshold detection")
            detection_results = []
            
            # Method 1: Ultra-sensitive FFmpeg scene detection
            if enable_multi_algorithm:
                try:
                    logger.debug("Running ultra-sensitive scene detection")
                    scenes_method1 = _detect_scenes_sensitive(video_path, threshold=threshold)
                    detection_results.append(("sensitive", scenes_method1, algorithm_weights.get('sensitive', 1.0)))
                    logger.info(f"Sensitive detection found {len(scenes_method1)} scenes")
                except Exception as e:
                    logger.warning(f"Sensitive detection failed: {e}")
            
            # Method 2: Static period detection for 1-second pauses
            if presentation_mode and enable_static_detection:
                try:
                    logger.debug("Running static period detection")
                    scenes_method2 = _detect_static_transitions(video_path, pause_duration=1.0)
                    detection_results.append(("static", scenes_method2, algorithm_weights.get('static', 1.2)))
                    logger.info(f"Static detection found {len(scenes_method2)} transitions")
                except Exception as e:
                    logger.warning(f"Static detection failed: {e}")
            
            # Method 3: Content-based detection with histogram comparison
            if enable_content_analysis:
                try:
                    logger.debug("Running content-based detection")
                    scenes_method3 = _detect_content_changes(video_path, threshold=threshold * 1.5)
                    detection_results.append(("content", scenes_method3, algorithm_weights.get('content', 0.9)))
                    logger.info(f"Content detection found {len(scenes_method3)} scenes")
                except Exception as e:
                    logger.warning(f"Content detection failed: {e}")
        
        if not detection_results:
            raise VideoAnalysisError("All scene detection methods failed")
        
        # Calculate algorithm reliability and apply fallback strategies
        reliability_scores = _calculate_algorithm_reliability(
            detection_results, expected_transitions, duration
        )
        logger.info(f"Algorithm reliability scores: {reliability_scores}")
        
        # Apply fallback strategies for unreliable algorithms
        min_reliability = video_config.get('min_algorithm_reliability', 0.3)
        detection_results = _apply_fallback_strategies(
            detection_results, reliability_scores, min_reliability
        )
        
        if not detection_results:
            raise VideoAnalysisError("All detection algorithms failed reliability checks")
        
        # Combine and validate results using weighted voting
        if enable_multi_algorithm and len(detection_results) > 1:
            combined_scenes, scene_metadata = _merge_scene_detections_weighted(
                detection_results, 
                tolerance=min_scene_duration,
                confidence_threshold=confidence_threshold
            )
        else:
            # Fallback to simple merge for single algorithm
            combined_scenes = _merge_scene_detections(
                [scenes for _, scenes, _ in detection_results], 
                tolerance=min_scene_duration
            )
            scene_metadata = [{'confidence': 0.8, 'methods': [method for method, _, _ in detection_results]} 
                             for _ in combined_scenes]
        
        # Apply Keynote delay compensation
        adjusted_scenes = []
        for i, timestamp in enumerate(combined_scenes):
            adjusted_timestamp = max(0, timestamp - keynote_delay)
            frame_number = int(adjusted_timestamp * frame_rate)
            
            # Get metadata for this scene if available
            metadata = scene_metadata[i] if i < len(scene_metadata) else {}
            
            adjusted_scenes.append({
                'timestamp': adjusted_timestamp,
                'original_timestamp': timestamp,
                'frame_number': frame_number,
                'confidence': metadata.get('confidence', 0.8),
                'slide_number': len(adjusted_scenes) + 1,
                'scene_score': None,
                'transition_type': 'keynote_optimized',
                'detection_methods': metadata.get('methods', [method for method, _, _ in detection_results]),
                'method_count': metadata.get('method_count', len(detection_results)),
                'total_weight': metadata.get('total_weight', 0)
            })
        
        # Validate against expected count and transcript cues
        validation_result = _validate_scene_count(
            adjusted_scenes, expected_transitions or 0, duration, transcript_path
        )
        
        # Apply corrections if needed
        if validation_result['status'] == 'UNDER_DETECTED' and expected_transitions and expected_transitions > 0:
            logger.info("Applying interpolation to add missing scenes")
            adjusted_scenes = _interpolate_missing_scenes(
                adjusted_scenes, expected_transitions, duration
            )
        
        # Create comprehensive diagnostic information
        diagnostic_info = {
            'algorithm_reliability': reliability_scores,
            'algorithm_weights': algorithm_weights,
            'confidence_threshold': confidence_threshold,
            'adaptive_detection_used': enable_adaptive and expected_transitions and expected_transitions > 0,
            'configuration': {
                'threshold': threshold,
                'keynote_delay': keynote_delay,
                'min_scene_duration': min_scene_duration,
                'multi_algorithm': enable_multi_algorithm,
                'static_detection': enable_static_detection,
                'content_analysis': enable_content_analysis
            }
        }
        
        # Add adaptive detection info if used
        if enable_adaptive and expected_transitions and expected_transitions > 0:
            diagnostic_info['adaptive_validation'] = locals().get('adaptive_validation', {})
        
        # Calculate scene statistics for logging
        scene_timestamps = [scene['timestamp'] for scene in adjusted_scenes]
        if scene_timestamps:
            gaps = [scene_timestamps[i] - scene_timestamps[i-1] for i in range(1, len(scene_timestamps))]
            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            min_gap = min(gaps) if gaps else 0
            max_gap = max(gaps) if gaps else 0
        else:
            avg_gap = min_gap = max_gap = 0
        
        result = {
            'scenes': adjusted_scenes,
            'detection_methods': [method for method, _, _ in detection_results],
            'method_results': {method: len(scenes) for method, scenes, _ in detection_results},
            'method_weights': {method: weight for method, _, weight in detection_results},
            'validation': validation_result,
            'video_info': {
                'duration': duration,
                'frame_rate': frame_rate,
                'keynote_delay': keynote_delay
            },
            'diagnostics': diagnostic_info,
            'statistics': {
                'total_scenes': len(adjusted_scenes),
                'average_gap': avg_gap,
                'min_gap': min_gap,
                'max_gap': max_gap,
                'scene_distribution': 'uniform' if max_gap - min_gap < avg_gap * 0.5 else 'varied'
            }
        }
        
        # Enhanced logging
        logger.info(f"Keynote scene detection complete: {len(adjusted_scenes)} scenes detected")
        logger.info(f"Scene statistics: avg_gap={avg_gap:.2f}s, min_gap={min_gap:.2f}s, max_gap={max_gap:.2f}s")
        logger.info(f"Detection methods used: {[method for method, _, _ in detection_results]}")
        logger.info(f"Algorithm reliability scores: {reliability_scores}")
        logger.info(f"Validation status: {validation_result.get('status', 'UNKNOWN')}")
        
        return result
        
    except Exception as e:
        raise VideoAnalysisError(f"Keynote scene detection failed for {video_path}: {str(e)}")


def save_analysis_manifest(
    analysis_result: Dict[str, Any],
    video_path: str,
    output_dir: str = None
) -> str:
    """
    Save comprehensive video analysis manifest for debugging and validation.
    
    Args:
        analysis_result: Result from detect_keynote_scenes()
        video_path: Path to the original video file
        output_dir: Directory to save manifest (defaults to config output_dir)
        
    Returns:
        Path to saved manifest file
    """
    if output_dir is None:
        config = config_loader.load_config()
        output_dir = config.get('video_analysis', {}).get('output_dir', 'workspace/video_analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create manifest filename based on video path
    video_stem = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = Path(output_dir) / f"scene_analysis_{video_stem}_{timestamp}.json"
    
    # Create comprehensive manifest
    manifest = {
        'metadata': {
            'video_path': video_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'autovid_version': 'Phase2-Enhanced',
            'analysis_type': 'keynote_scene_detection'
        },
        'analysis_result': analysis_result,
        'summary': {
            'total_scenes': len(analysis_result.get('scenes', [])),
            'detection_methods': analysis_result.get('detection_methods', []),
            'validation_status': analysis_result.get('validation', {}).get('status', 'UNKNOWN'),
            'reliability_scores': analysis_result.get('diagnostics', {}).get('algorithm_reliability', {}),
            'adaptive_used': analysis_result.get('diagnostics', {}).get('adaptive_detection_used', False),
            'scene_distribution': analysis_result.get('statistics', {}).get('scene_distribution', 'unknown')
        }
    }
    
    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    logger.info(f"Analysis manifest saved to: {manifest_path}")
    return str(manifest_path)


def _detect_scenes_sensitive(video_path: str, threshold: float = 0.1) -> List[float]:
    """
    Ultra-sensitive scene detection with lowered threshold for slide transitions.
    
    Args:
        video_path: Path to the video file
        threshold: Scene detection threshold (lower = more sensitive)
        
    Returns:
        List of scene transition timestamps
    """
    try:
        result = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=1)  # Reduce framerate aggressively for 4K video
            .filter('select', f'gt(scene,{threshold})')
            .filter('showinfo')
            .output('pipe:', format='null')
            .global_args('-loglevel', 'info', '-nostats')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        stderr_output = result[1].decode()
        scenes = []
        
        import re
        for line in stderr_output.split('\n'):
            if 'showinfo' in line and 'pts_time:' in line:
                timestamp_match = re.search(r'pts_time:(\d+(?:\.\d+)?)', line)
                if timestamp_match:
                    timestamp = float(timestamp_match.group(1))
                    scenes.append(timestamp)
        
        return sorted(list(set(scenes)))  # Remove duplicates and sort
        
    except ffmpeg.Error as e:
        logger.error(f"Sensitive scene detection failed: {e.stderr.decode()}")
        return []


def _detect_static_transitions(video_path: str, pause_duration: float = 1.0) -> List[float]:
    """
    Detect scene transitions by finding static frame periods.
    
    Keynote videos have 1-second pauses before slide transitions.
    This function detects when frames become nearly identical for 1+ seconds.
    
    Args:
        video_path: Path to the video file
        pause_duration: Duration of static period to detect
        
    Returns:
        List of transition timestamps after static periods
    """
    try:
        # Use framerate decimation and small scene changes to avoid processing every frame
        # This reduces processing time while still detecting transitions
        result = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=1)  # Reduce framerate aggressively for 4K video
            .filter('select', 'gt(scene,0.005)')  # Small scene changes only
            .filter('showinfo') 
            .output('pipe:', format='null')
            .global_args('-loglevel', 'info', '-nostats')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        stderr_output = result[1].decode()
        transitions = []
        
        import re
        for line in stderr_output.split('\n'):
            if 'showinfo' in line and 'pts_time:' in line:
                timestamp_match = re.search(r'pts_time:(\d+(?:\.\d+)?)', line)
                if timestamp_match:
                    timestamp = float(timestamp_match.group(1))
                    transitions.append(timestamp)
        
        # Filter transitions that are at least pause_duration apart
        filtered_transitions = []
        last_transition = -pause_duration
        
        for timestamp in sorted(transitions):
            if timestamp - last_transition >= pause_duration:
                filtered_transitions.append(timestamp)
                last_transition = timestamp
        
        return filtered_transitions
        
    except ffmpeg.Error as e:
        logger.error(f"Static transition detection failed: {e.stderr.decode()}")
        return []


def _detect_content_changes(video_path: str, threshold: float = 0.15) -> List[float]:
    """
    Histogram-based content change detection for slide transitions.
    
    Args:
        video_path: Path to the video file
        threshold: Content difference threshold
        
    Returns:
        List of scene transition timestamps
    """
    try:
        # Use histogram-based detection which is more sensitive to color/content changes
        result = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=1)  # Reduce framerate aggressively for 4K video
            .filter('select', f'gt(scene,{threshold})')
            .filter('metadata', mode='print')
            .output('pipe:', format='null')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        stderr_output = result[1].decode()
        scenes = []
        
        import re
        for line in stderr_output.split('\n'):
            # Look for frame metadata with timestamps
            timestamp_match = re.search(r'frame:\s*\d+\s+pts:\s*\d+\s+pts_time:(\d+(?:\.\d+)?)', line)
            if timestamp_match:
                timestamp = float(timestamp_match.group(1))
                scenes.append(timestamp)
        
        return sorted(list(set(scenes)))
        
    except ffmpeg.Error as e:
        logger.error(f"Content-based detection failed: {e.stderr.decode()}")
        return []


def _merge_scene_detections(detection_results: List[List[float]], tolerance: float = 0.5) -> List[float]:
    """
    Merge overlapping detections from multiple algorithms.
    
    Args:
        detection_results: List of detection results from different methods
        tolerance: Time tolerance for merging similar detections
        
    Returns:
        Merged list of scene timestamps
    """
    if not detection_results:
        return []
    
    # Flatten all detections
    all_scenes = []
    for scenes in detection_results:
        all_scenes.extend(scenes)
    
    if not all_scenes:
        return []
    
    # Sort and merge nearby detections
    all_scenes.sort()
    merged = [all_scenes[0]]
    
    for timestamp in all_scenes[1:]:
        # If this timestamp is close to the last merged one, skip it
        if timestamp - merged[-1] > tolerance:
            merged.append(timestamp)
    
    return merged


def _merge_scene_detections_weighted(
    detection_results: List[Tuple[str, List[float], float]], 
    tolerance: float = 0.5,
    confidence_threshold: float = 0.5
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Merge overlapping detections using weighted voting with confidence scores.
    
    Args:
        detection_results: List of (method_name, scenes, weight) tuples
        tolerance: Time tolerance for grouping similar detections
        confidence_threshold: Minimum confidence required for scene acceptance
        
    Returns:
        Tuple of (merged_scenes, scene_metadata)
    """
    if not detection_results:
        return [], []
    
    # Create weighted scene candidates
    scene_candidates = []
    for method_name, scenes, weight in detection_results:
        for timestamp in scenes:
            scene_candidates.append({
                'timestamp': timestamp,
                'method': method_name,
                'weight': weight,
                'confidence': weight  # Initial confidence equals weight
            })
    
    if not scene_candidates:
        return [], []
    
    # Sort by timestamp
    scene_candidates.sort(key=lambda x: x['timestamp'])
    
    # Group nearby candidates using clustering
    clusters = []
    current_cluster = [scene_candidates[0]]
    
    for candidate in scene_candidates[1:]:
        # If within tolerance of cluster center, add to current cluster
        cluster_center = sum(c['timestamp'] for c in current_cluster) / len(current_cluster)
        if abs(candidate['timestamp'] - cluster_center) <= tolerance:
            current_cluster.append(candidate)
        else:
            # Start new cluster
            clusters.append(current_cluster)
            current_cluster = [candidate]
    clusters.append(current_cluster)
    
    # Apply weighted voting within each cluster
    merged_scenes = []
    scene_metadata = []
    
    for cluster in clusters:
        # Calculate weighted average timestamp
        total_weight = sum(c['weight'] for c in cluster)
        weighted_timestamp = sum(c['timestamp'] * c['weight'] for c in cluster) / total_weight
        
        # Calculate combined confidence
        confidence = min(1.0, total_weight / len(detection_results))  # Normalize by number of methods
        
        # Only accept scenes above confidence threshold
        if confidence >= confidence_threshold:
            merged_scenes.append(weighted_timestamp)
            
            # Create metadata for this scene
            methods_used = list(set(c['method'] for c in cluster))
            scene_metadata.append({
                'timestamp': weighted_timestamp,
                'confidence': confidence,
                'methods': methods_used,
                'method_count': len(methods_used),
                'total_weight': total_weight,
                'cluster_size': len(cluster)
            })
    
    return merged_scenes, scene_metadata


def _calculate_algorithm_reliability(
    detection_results: List[Tuple[str, List[float], float]],
    expected_count: Optional[int] = None,
    video_duration: float = 0.0
) -> Dict[str, float]:
    """
    Calculate reliability scores for each detection algorithm.
    
    Args:
        detection_results: List of (method_name, scenes, weight) tuples
        expected_count: Expected number of transitions for accuracy calculation
        video_duration: Video duration for distribution analysis
        
    Returns:
        Dictionary mapping algorithm names to reliability scores (0.0-1.0)
    """
    reliability_scores = {}
    
    for method_name, scenes, base_weight in detection_results:
        score = base_weight  # Start with base weight
        
        # Factor 1: Detection count reasonableness
        if expected_count and expected_count > 0:
            count_ratio = len(scenes) / expected_count
            # Penalize extreme over/under detection
            if count_ratio < 0.5 or count_ratio > 2.0:
                score *= 0.7
            elif 0.8 <= count_ratio <= 1.2:
                score *= 1.1  # Reward accurate counts
        
        # Factor 2: Scene distribution (avoid clustering at beginning/end)
        if len(scenes) > 1 and video_duration > 0:
            # Check if scenes are well distributed
            gaps = []
            for i in range(1, len(scenes)):
                gaps.append(scenes[i] - scenes[i-1])
            
            if gaps:
                gap_variance = sum((gap - sum(gaps)/len(gaps))**2 for gap in gaps) / len(gaps)
                # Reward consistent spacing
                if gap_variance < (video_duration / len(scenes)) * 0.5:
                    score *= 1.05
        
        # Factor 3: Avoid edge clustering (first/last 10% of video)
        edge_threshold = video_duration * 0.1
        edge_scenes = sum(1 for s in scenes if s < edge_threshold or s > video_duration - edge_threshold)
        if edge_scenes > len(scenes) * 0.5:  # More than half at edges
            score *= 0.8
        
        # Normalize score to 0.0-1.0 range
        reliability_scores[method_name] = min(1.0, max(0.1, score))
    
    return reliability_scores


def _apply_fallback_strategies(
    detection_results: List[Tuple[str, List[float], float]],
    reliability_scores: Dict[str, float],
    min_reliability: float = 0.3
) -> List[Tuple[str, List[float], float]]:
    """
    Apply fallback strategies when algorithms have low reliability.
    
    Args:
        detection_results: Original detection results
        reliability_scores: Calculated reliability scores
        min_reliability: Minimum acceptable reliability score
        
    Returns:
        Filtered and adjusted detection results
    """
    # Filter out unreliable methods
    reliable_results = []
    for method_name, scenes, weight in detection_results:
        reliability = reliability_scores.get(method_name, 0.5)
        if reliability >= min_reliability:
            # Adjust weight based on reliability
            adjusted_weight = weight * reliability
            reliable_results.append((method_name, scenes, adjusted_weight))
    
    # If no methods are reliable enough, keep the best one with reduced confidence
    if not reliable_results and detection_results:
        best_method = max(detection_results, key=lambda x: reliability_scores.get(x[0], 0.0))
        method_name, scenes, weight = best_method
        # Reduce weight significantly for unreliable fallback
        fallback_weight = weight * 0.3
        reliable_results = [(method_name, scenes, fallback_weight)]
        logger.warning(f"All algorithms unreliable, using fallback: {method_name} (reliability: {reliability_scores.get(method_name, 0.0):.2f})")
    
    return reliable_results


def _adjust_thresholds_adaptively(
    initial_threshold: float,
    validation_result: Dict[str, Any],
    expected_count: Optional[int] = None,
    adjustment_factor: float = 0.1,
    max_adjustments: int = 3
) -> Tuple[float, bool]:
    """
    Adaptively adjust detection threshold based on validation results.
    
    Args:
        initial_threshold: Starting threshold value
        validation_result: Validation result from _validate_scene_count
        expected_count: Expected number of transitions
        adjustment_factor: How much to adjust threshold (0.0-1.0)
        max_adjustments: Maximum number of adjustment attempts
        
    Returns:
        Tuple of (adjusted_threshold, should_retry)
    """
    status = validation_result.get('status', 'UNKNOWN')
    detected_count = validation_result.get('detected_count', 0)
    
    if status == 'VALIDATED' or not expected_count or expected_count == 0:
        return initial_threshold, False
    
    # Calculate adjustment based on validation status
    new_threshold = initial_threshold
    should_retry = False
    
    if status == 'UNDER_DETECTED':
        # Lower threshold to increase sensitivity
        new_threshold = max(0.01, initial_threshold - adjustment_factor)
        should_retry = True
        logger.info(f"Under-detection: lowering threshold from {initial_threshold:.3f} to {new_threshold:.3f}")
    
    elif status == 'OVER_DETECTED':
        # Raise threshold to decrease sensitivity
        new_threshold = min(1.0, initial_threshold + adjustment_factor)
        should_retry = True
        logger.info(f"Over-detection: raising threshold from {initial_threshold:.3f} to {new_threshold:.3f}")
    
    # Don't retry if threshold can't be adjusted further
    if abs(new_threshold - initial_threshold) < 0.001:
        should_retry = False
    
    return new_threshold, should_retry


def _detect_with_adaptive_thresholds(
    video_path: str,
    video_config: Dict[str, Any],
    expected_transitions: Optional[int],
    duration: float,
    max_attempts: int = 3
) -> Tuple[List[Tuple[str, List[float], float]], Dict[str, Any]]:
    """
    Perform scene detection with adaptive threshold adjustment.
    
    Args:
        video_path: Path to video file
        video_config: Configuration parameters
        expected_transitions: Expected number of transitions
        duration: Video duration
        max_attempts: Maximum adjustment attempts
        
    Returns:
        Tuple of (detection_results, final_validation_result)
    """
    threshold = video_config.get('scene_threshold', 0.1)
    best_results = None
    best_validation = None
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        logger.info(f"Adaptive detection attempt {attempt}/{max_attempts} with threshold {threshold:.3f}")
        
        # Run detection with current threshold
        detection_results = []
        
        # Method 1: Ultra-sensitive FFmpeg scene detection
        if video_config.get('enable_multi_algorithm', True):
            try:
                scenes_method1 = _detect_scenes_sensitive(video_path, threshold=threshold)
                weight = video_config.get('algorithm_weights', {}).get('sensitive', 1.0)
                detection_results.append(("sensitive", scenes_method1, weight))
            except Exception as e:
                logger.warning(f"Sensitive detection failed: {e}")
        
        # Method 2: Static period detection
        if video_config.get('presentation_mode', True) and video_config.get('enable_static_detection', True):
            try:
                scenes_method2 = _detect_static_transitions(video_path, pause_duration=1.0)
                weight = video_config.get('algorithm_weights', {}).get('static', 1.2)
                detection_results.append(("static", scenes_method2, weight))
            except Exception as e:
                logger.warning(f"Static detection failed: {e}")
        
        # Method 3: Content-based detection
        if video_config.get('enable_content_analysis', True):
            try:
                scenes_method3 = _detect_content_changes(video_path, threshold=threshold * 1.5)
                weight = video_config.get('algorithm_weights', {}).get('content', 0.9)
                detection_results.append(("content", scenes_method3, weight))
            except Exception as e:
                logger.warning(f"Content detection failed: {e}")
        
        if not detection_results:
            logger.warning(f"No detection results on attempt {attempt}")
            break
        
        # Quick validation to check if we should adjust
        total_scenes = sum(len(scenes) for _, scenes, _ in detection_results)
        quick_validation = {
            'detected_count': total_scenes,
            'expected_count': expected_transitions or 0,
            'status': 'VALIDATED'
        }
        
        if expected_transitions and expected_transitions > 0:
            ratio = total_scenes / expected_transitions
            if ratio < 0.7:
                quick_validation['status'] = 'UNDER_DETECTED'
            elif ratio > 1.5:
                quick_validation['status'] = 'OVER_DETECTED'
        
        # Store best results so far
        if best_results is None or quick_validation['status'] == 'VALIDATED':
            best_results = detection_results
            best_validation = quick_validation
        
        # If validation is good or we can't adjust further, stop
        if quick_validation['status'] == 'VALIDATED':
            logger.info(f"Adaptive threshold found optimal value: {threshold:.3f}")
            break
        
        # Adjust threshold for next attempt
        adjustment_factor = 0.1 / attempt  # Smaller adjustments on later attempts
        new_threshold, should_retry = _adjust_thresholds_adaptively(
            threshold, quick_validation, expected_transitions, adjustment_factor
        )
        
        if not should_retry or abs(new_threshold - threshold) < 0.001:
            logger.info(f"Threshold adjustment complete after {attempt} attempts")
            break
        
        threshold = new_threshold
    
    return best_results or [], best_validation or {'status': 'FAILED', 'detected_count': 0}


def _validate_scene_count(
    detected_scenes: List[Dict[str, Any]], 
    expected_count: int, 
    video_duration: float,
    transcript_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhanced validation with corrective actions and transcript cue validation.
    
    Args:
        detected_scenes: List of detected scene dictionaries
        expected_count: Expected number of transitions
        video_duration: Total video duration
        transcript_path: Optional path to transcript JSON for cue validation
        
    Returns:
        Validation result with status and recommendations
    """
    detected_count = len(detected_scenes)
    
    # Try to validate against transcript cues if available
    transcript_cue_count = None
    if transcript_path and os.path.exists(transcript_path):
        try:
            with open(transcript_path, 'r') as f:
                transcript_data = json.load(f)
            
            # Count [transition] cues in transcript chunks
            transition_cues = 0
            chunks = transcript_data.get('chunks', [])
            for chunk in chunks:
                text = chunk.get('text', '')
                transition_cues += text.count('[transition]')
            
            transcript_cue_count = transition_cues
            logger.info(f"Found {transition_cues} [transition] cues in transcript")
            
            # If transcript cues are available and expected_count is 0, use transcript count
            if expected_count == 0 and transcript_cue_count > 0:
                expected_count = transcript_cue_count
                logger.info(f"Using transcript cue count as expected: {expected_count}")
                
        except Exception as e:
            logger.warning(f"Failed to validate against transcript cues: {e}")
    
    if expected_count == 0:
        return {
            'status': 'SKIP',
            'message': 'No expected transitions provided',
            'detected_count': detected_count,
            'expected_count': 0,
            'transcript_cue_count': transcript_cue_count,
            'ratio': 0
        }
    
    ratio = detected_count / expected_count if expected_count > 0 else 0
    
    # Enhanced validation with transcript cue cross-check
    validation_sources = []
    if transcript_cue_count is not None:
        cue_ratio = detected_count / transcript_cue_count if transcript_cue_count > 0 else 0
        validation_sources.append(f"transcript_cues:{transcript_cue_count}(ratio:{cue_ratio:.2f})")
    
    # Determine status based on ratio
    if 0.7 <= ratio <= 1.3:  # Within 30%
        status = 'VALIDATED'
        message = f"Good match: {detected_count}/{expected_count} scenes (ratio: {ratio:.2f})"
    elif 0.3 <= ratio < 0.7:
        status = 'UNDER_DETECTED'
        message = f"Under-detection: {detected_count}/{expected_count} scenes (ratio: {ratio:.2f})"
    elif ratio < 0.3:
        status = 'SEVERE_UNDER_DETECTED'
        message = f"Severe under-detection: {detected_count}/{expected_count} scenes (ratio: {ratio:.2f})"
    else:
        status = 'OVER_DETECTED'
        message = f"Over-detection: {detected_count}/{expected_count} scenes (ratio: {ratio:.2f})"
    
    # Add transcript validation information to message
    if validation_sources:
        message += f" [Sources: {', '.join(validation_sources)}]"
    
    return {
        'status': status,
        'message': message,
        'detected_count': detected_count,
        'expected_count': expected_count,
        'transcript_cue_count': transcript_cue_count,
        'ratio': ratio,
        'validation_sources': validation_sources
    }


def _interpolate_missing_scenes(
    detected_scenes: List[Dict[str, Any]], 
    expected_count: int, 
    duration: float
) -> List[Dict[str, Any]]:
    """
    Interpolate missing scenes when under-detection occurs.
    
    Args:
        detected_scenes: Current detected scenes
        expected_count: Expected number of scenes
        duration: Video duration
        
    Returns:
        Enhanced scene list with interpolated scenes
    """
    if len(detected_scenes) >= expected_count:
        return detected_scenes
    
    # Calculate how many scenes we need to add
    missing_count = expected_count - len(detected_scenes)
    
    # Create evenly spaced interpolated scenes
    if detected_scenes:
        # Use detected scenes as anchors and interpolate between them
        enhanced_scenes = detected_scenes.copy()
        
        # Add interpolated scenes between existing ones
        for i in range(len(detected_scenes) - 1):
            start_time = detected_scenes[i]['timestamp']
            end_time = detected_scenes[i + 1]['timestamp']
            gap_duration = end_time - start_time
            
            # If gap is large enough, add interpolated scenes
            if gap_duration > 10.0 and missing_count > 0:  # 10 second minimum gap
                interpolated_time = start_time + (gap_duration / 2)
                enhanced_scenes.append({
                    'timestamp': interpolated_time,
                    'original_timestamp': interpolated_time,
                    'frame_number': int(interpolated_time * 30),  # Assume 30fps
                    'confidence': 0.5,  # Lower confidence for interpolated
                    'slide_number': 0,  # Will be renumbered
                    'scene_score': None,
                    'transition_type': 'interpolated',
                    'detection_methods': ['interpolation']
                })
                missing_count -= 1
        
        # Sort by timestamp and renumber
        enhanced_scenes.sort(key=lambda x: x['timestamp'])
        for i, scene in enumerate(enhanced_scenes):
            scene['slide_number'] = i + 1
        
        return enhanced_scenes
    else:
        # No detected scenes, create evenly distributed scenes
        scenes = []
        for i in range(expected_count):
            timestamp = (duration / expected_count) * i
            scenes.append({
                'timestamp': timestamp,
                'original_timestamp': timestamp,
                'frame_number': int(timestamp * 30),
                'confidence': 0.3,  # Low confidence for estimated
                'slide_number': i + 1,
                'scene_score': None,
                'transition_type': 'estimated',
                'detection_methods': ['estimation']
            })
        return scenes


def analyze_video(video_path: str, 
                 expected_transitions: Optional[List[str]] = None,
                 scene_threshold: float = 0.4,
                 movement_threshold: float = 0.1,
                 keynote_delay: float = 1.0,
                 presentation_mode: bool = False,
                 output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to perform complete video analysis.
    
    Args:
        video_path: Path to the video file to analyze
        expected_transitions: List of expected transition cues for validation
        scene_threshold: Scene detection sensitivity (0.0-1.0)
        movement_threshold: Movement detection sensitivity
        keynote_delay: Keynote export delay compensation in seconds
        output_path: Optional path to save the analysis manifest
        
    Returns:
        Complete analysis manifest
        
    Raises:
        VideoAnalysisError: If analysis fails
    """
    try:
        logger.info(f"Starting video analysis for {video_path}")
        
        # Validate input file
        if not os.path.exists(video_path):
            raise VideoAnalysisError(f"Video file not found: {video_path}")
        
        # Step 1: Probe video information
        video_info = probe_video_info(video_path)
        logger.info(f"Video info: {video_info['width']}x{video_info['height']}, "
                   f"{video_info['duration']:.2f}s, {video_info['frame_rate']:.2f}fps")
        
        # Step 2: Detect scene transitions
        if presentation_mode and expected_transitions:
            # Use Keynote-optimized detection
            expected_count = len(expected_transitions) if expected_transitions else 0
            keynote_result = detect_keynote_scenes(
                video_path=video_path,
                expected_transitions=expected_count,
                threshold=scene_threshold,
                keynote_delay=keynote_delay,
                presentation_mode=True
            )
            scene_transitions = keynote_result['scenes']
            logger.info(f"Using Keynote-optimized detection: {keynote_result['validation']['message']}")
        else:
            # Use traditional scene detection
            scene_transitions = detect_scene_changes(video_path, scene_threshold)
        
        # Step 3: Apply Keynote delay compensation (if not already applied)
        if presentation_mode and expected_transitions:
            # Keynote detection already applies delay compensation
            compensated_transitions = scene_transitions
        else:
            # Apply delay compensation for traditional detection
            compensated_transitions = compensate_keynote_delay(scene_transitions, keynote_delay)
        
        # Step 4: Extract movement ranges
        movement_ranges = extract_movement_frames(video_path, compensated_transitions, movement_threshold)
        
        # Step 5: Validate transition count
        validation_result = {'status': 'SKIP', 'message': 'No expected transitions provided'}
        if expected_transitions:
            validation_result = validate_transition_count(compensated_transitions, expected_transitions)
        
        # Step 6: Generate comprehensive manifest
        manifest = generate_timing_manifest(
            video_path, compensated_transitions, movement_ranges, 
            video_info, validation_result, keynote_delay
        )
        
        # Step 7: Save manifest if output path specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Analysis manifest saved to {output_path}")
        
        logger.info(f"Video analysis complete: {len(compensated_transitions)} scenes, "
                   f"{len(movement_ranges)} movements")
        return manifest
        
    except Exception as e:
        error_msg = f"Video analysis failed for {video_path}: {str(e)}"
        logger.error(error_msg)
        raise VideoAnalysisError(error_msg)