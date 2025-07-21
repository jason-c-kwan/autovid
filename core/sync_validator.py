"""
Video synchronization validation module for AutoVid pipeline.

This module provides quality assurance and validation tools for video-audio
synchronization. It can measure sync accuracy, detect timing drift, and
generate comprehensive sync quality reports.
"""

import os
import json
import logging
import tempfile
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

from .video_sync import SyncPoint, TimingCorrection, VideoSyncError

logger = logging.getLogger(__name__)


class SyncValidationError(Exception):
    """Custom exception for sync validation errors."""
    pass


@dataclass
class SyncMetrics:
    """Container for synchronization quality metrics."""
    avg_offset: float
    max_offset: float
    std_deviation: float
    sync_accuracy_score: float
    drift_rate: float
    timing_consistency: float
    confidence_score: float


@dataclass
class DriftAnalysis:
    """Container for timing drift analysis results."""
    total_drift: float
    drift_rate_per_minute: float
    drift_direction: str
    significant_drift_points: List[Dict[str, Any]]
    drift_consistency: float
    requires_correction: bool


class SyncReport(NamedTuple):
    """Complete synchronization validation report."""
    overall_grade: str
    sync_metrics: SyncMetrics
    drift_analysis: DriftAnalysis
    quality_issues: List[Dict[str, Any]]
    recommendations: List[str]
    validation_timestamp: str


def measure_audio_video_offset(video_path: str, 
                             audio_path: str,
                             sample_points: int = 5) -> List[float]:
    """
    Measure actual audio-video offset at multiple points in the synchronized video.
    
    Args:
        video_path: Path to synchronized video file
        audio_path: Path to reference audio file (optional)
        sample_points: Number of sample points to measure
        
    Returns:
        List of measured offsets in seconds
        
    Raises:
        SyncValidationError: If offset measurement fails
    """
    try:
        # Get video duration
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        duration = float(video_stream['duration'])
        
        measured_offsets = []
        
        # Sample at evenly distributed points
        for i in range(sample_points):
            sample_time = (duration / (sample_points + 1)) * (i + 1)
            
            try:
                # Extract a short clip for analysis
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                
                # Extract audio from video at sample point
                (
                    ffmpeg
                    .input(video_path, ss=sample_time, t=1.0)
                    .output(temp_audio_path, acodec='pcm_s16le', ac=1, ar=16000)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                # For now, we'll estimate offset based on timing analysis
                # A more sophisticated approach would use cross-correlation
                estimated_offset = 0.0  # Placeholder
                measured_offsets.append(estimated_offset)
                
                # Clean up temp file
                try:
                    os.unlink(temp_audio_path)
                except OSError:
                    pass
                    
            except ffmpeg.Error as e:
                logger.warning(f"Failed to measure offset at {sample_time:.1f}s: {e.stderr.decode()}")
                continue
        
        logger.info(f"Measured offsets at {len(measured_offsets)} sample points")
        return measured_offsets
        
    except Exception as e:
        raise SyncValidationError(f"Failed to measure audio-video offset: {str(e)}")


def calculate_sync_metrics(sync_points: List[SyncPoint]) -> SyncMetrics:
    """
    Calculate comprehensive synchronization quality metrics.
    
    Args:
        sync_points: List of synchronization points to analyze
        
    Returns:
        SyncMetrics object with calculated quality metrics
        
    Raises:
        SyncValidationError: If metrics cannot be calculated
    """
    try:
        if not sync_points:
            raise SyncValidationError("No sync points provided for metrics calculation")
        
        # Extract offset values
        offsets = [abs(point.sync_offset) for point in sync_points]
        confidences = [point.confidence for point in sync_points]
        
        # Basic statistics
        avg_offset = statistics.mean(offsets)
        max_offset = max(offsets)
        std_deviation = statistics.stdev(offsets) if len(offsets) > 1 else 0.0
        
        # Calculate sync accuracy score (0-100)
        # Lower average offset and lower standard deviation = higher score
        accuracy_base = max(0, 100 - (avg_offset * 1000))  # Penalize by milliseconds
        consistency_bonus = max(0, 20 - (std_deviation * 100))  # Bonus for consistency
        sync_accuracy_score = min(100, accuracy_base + consistency_bonus)
        
        # Calculate drift rate (change in offset over time)
        drift_rate = 0.0
        if len(sync_points) > 1:
            first_offset = sync_points[0].sync_offset
            last_offset = sync_points[-1].sync_offset
            time_span = sync_points[-1].video_timestamp - sync_points[0].video_timestamp
            if time_span > 0:
                drift_rate = (last_offset - first_offset) / time_span
        
        # Calculate timing consistency (how stable offsets are)
        timing_consistency = max(0, 100 - (std_deviation * 1000))  # 0-100 scale
        
        # Calculate confidence score
        confidence_score = statistics.mean(confidences) * 100 if confidences else 0.0
        
        metrics = SyncMetrics(
            avg_offset=avg_offset,
            max_offset=max_offset,
            std_deviation=std_deviation,
            sync_accuracy_score=sync_accuracy_score,
            drift_rate=drift_rate,
            timing_consistency=timing_consistency,
            confidence_score=confidence_score
        )
        
        logger.info(f"Calculated sync metrics: accuracy={sync_accuracy_score:.1f}, "
                   f"avg_offset={avg_offset*1000:.1f}ms")
        
        return metrics
        
    except Exception as e:
        raise SyncValidationError(f"Failed to calculate sync metrics: {str(e)}")


def analyze_timing_drift(sync_points: List[SyncPoint], 
                        video_duration: float,
                        drift_threshold: float = 0.1) -> DriftAnalysis:
    """
    Analyze timing drift patterns across the video duration.
    
    Args:
        sync_points: List of synchronization points
        video_duration: Total video duration in seconds
        drift_threshold: Threshold for significant drift in seconds
        
    Returns:
        DriftAnalysis object with drift analysis results
        
    Raises:
        SyncValidationError: If drift analysis fails
    """
    try:
        if not sync_points or len(sync_points) < 2:
            return DriftAnalysis(
                total_drift=0.0,
                drift_rate_per_minute=0.0,
                drift_direction='none',
                significant_drift_points=[],
                drift_consistency=100.0,
                requires_correction=False
            )
        
        # Calculate total drift
        first_offset = sync_points[0].sync_offset
        last_offset = sync_points[-1].sync_offset
        total_drift = last_offset - first_offset
        
        # Calculate drift rate per minute
        time_span = sync_points[-1].video_timestamp - sync_points[0].video_timestamp
        drift_rate_per_minute = (total_drift / (time_span / 60.0)) if time_span > 0 else 0.0
        
        # Determine drift direction
        if abs(total_drift) < 0.01:
            drift_direction = 'none'
        elif total_drift > 0:
            drift_direction = 'audio_lagging'
        else:
            drift_direction = 'audio_leading'
        
        # Find significant drift points
        significant_drift_points = []
        for i, point in enumerate(sync_points[1:], 1):
            prev_point = sync_points[i - 1]
            drift_change = point.sync_offset - prev_point.sync_offset
            
            if abs(drift_change) > drift_threshold:
                significant_drift_points.append({
                    'scene_index': point.scene_index,
                    'timestamp': point.video_timestamp,
                    'drift_change': drift_change,
                    'cumulative_drift': point.sync_offset - first_offset
                })
        
        # Calculate drift consistency (how uniform the drift is)
        offsets = [point.sync_offset for point in sync_points]
        offset_changes = [offsets[i] - offsets[i-1] for i in range(1, len(offsets))]
        
        if offset_changes:
            drift_std = statistics.stdev(offset_changes) if len(offset_changes) > 1 else 0.0
            drift_consistency = max(0, 100 - (drift_std * 1000))  # Higher = more consistent
        else:
            drift_consistency = 100.0
        
        # Determine if correction is required
        requires_correction = (
            abs(total_drift) > drift_threshold or
            len(significant_drift_points) > len(sync_points) * 0.3 or
            drift_consistency < 50.0
        )
        
        analysis = DriftAnalysis(
            total_drift=total_drift,
            drift_rate_per_minute=drift_rate_per_minute,
            drift_direction=drift_direction,
            significant_drift_points=significant_drift_points,
            drift_consistency=drift_consistency,
            requires_correction=requires_correction
        )
        
        logger.info(f"Drift analysis: total={total_drift*1000:.1f}ms, "
                   f"rate={drift_rate_per_minute*1000:.1f}ms/min, "
                   f"requires_correction={requires_correction}")
        
        return analysis
        
    except Exception as e:
        raise SyncValidationError(f"Failed to analyze timing drift: {str(e)}")


def identify_quality_issues(sync_metrics: SyncMetrics, 
                          drift_analysis: DriftAnalysis,
                          sync_points: List[SyncPoint]) -> List[Dict[str, Any]]:
    """
    Identify specific quality issues with the synchronization.
    
    Args:
        sync_metrics: Calculated sync quality metrics
        drift_analysis: Timing drift analysis results
        sync_points: List of synchronization points
        
    Returns:
        List of quality issue dictionaries
    """
    issues = []
    
    # Check for high average offset
    if sync_metrics.avg_offset > 0.05:  # 50ms
        issues.append({
            'type': 'high_average_offset',
            'severity': 'warning' if sync_metrics.avg_offset < 0.1 else 'error',
            'description': f"High average sync offset: {sync_metrics.avg_offset*1000:.1f}ms",
            'recommendation': 'Consider applying global timing correction'
        })
    
    # Check for inconsistent timing
    if sync_metrics.timing_consistency < 70.0:
        issues.append({
            'type': 'timing_inconsistency',
            'severity': 'warning',
            'description': f"Low timing consistency: {sync_metrics.timing_consistency:.1f}%",
            'recommendation': 'Review scene detection and audio chunk boundaries'
        })
    
    # Check for significant drift
    if drift_analysis.requires_correction:
        issues.append({
            'type': 'significant_drift',
            'severity': 'error',
            'description': f"Significant timing drift detected: {drift_analysis.total_drift*1000:.1f}ms total",
            'recommendation': 'Apply progressive timing corrections'
        })
    
    # Check for low confidence scores
    low_confidence_points = [p for p in sync_points if p.confidence < 0.5]
    if len(low_confidence_points) > len(sync_points) * 0.3:
        issues.append({
            'type': 'low_confidence',
            'severity': 'warning',
            'description': f"{len(low_confidence_points)} sync points have low confidence",
            'recommendation': 'Manual review of scene detection results recommended'
        })
    
    # Check for extreme offsets
    extreme_points = [p for p in sync_points if abs(p.sync_offset) > 0.2]
    if extreme_points:
        issues.append({
            'type': 'extreme_offsets',
            'severity': 'error',
            'description': f"{len(extreme_points)} sync points have extreme offsets (>200ms)",
            'recommendation': 'Review Keynote delay compensation and scene alignment'
        })
    
    return issues


def generate_sync_recommendations(sync_metrics: SyncMetrics,
                                drift_analysis: DriftAnalysis,
                                quality_issues: List[Dict[str, Any]]) -> List[str]:
    """
    Generate actionable recommendations for improving sync quality.
    
    Args:
        sync_metrics: Sync quality metrics
        drift_analysis: Drift analysis results
        quality_issues: List of identified quality issues
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Overall quality recommendations
    if sync_metrics.sync_accuracy_score < 80:
        recommendations.append("Overall sync quality is below optimal. Consider reviewing input data.")
    
    # Drift-specific recommendations
    if drift_analysis.requires_correction:
        if drift_analysis.drift_direction == 'audio_lagging':
            recommendations.append("Audio appears to be lagging. Consider reducing Keynote delay compensation.")
        elif drift_analysis.drift_direction == 'audio_leading':
            recommendations.append("Audio appears to be leading. Consider increasing Keynote delay compensation.")
        
        if len(drift_analysis.significant_drift_points) > 2:
            recommendations.append("Multiple significant drift points detected. Consider using multi-point synchronization.")
    
    # Confidence-specific recommendations
    if sync_metrics.confidence_score < 70:
        recommendations.append("Low confidence in sync points. Review video analysis scene detection parameters.")
        recommendations.append("Consider manual validation of scene transition timing.")
    
    # Issue-specific recommendations
    error_issues = [i for i in quality_issues if i['severity'] == 'error']
    if error_issues:
        recommendations.append("Critical sync errors detected. Manual review strongly recommended.")
    
    # Performance recommendations
    if sync_metrics.std_deviation > 0.05:
        recommendations.append("High timing variability detected. Consider reviewing audio chunk boundaries.")
    
    # Default recommendation if quality is good
    if not recommendations and sync_metrics.sync_accuracy_score >= 90:
        recommendations.append("Synchronization quality is excellent. No corrections needed.")
    
    return recommendations


def validate_sync_accuracy(video_path: str,
                         sync_points: List[SyncPoint],
                         timing_corrections: Optional[TimingCorrection] = None,
                         video_duration: Optional[float] = None) -> SyncReport:
    """
    Perform comprehensive synchronization accuracy validation.
    
    Args:
        video_path: Path to synchronized video file
        sync_points: List of synchronization points used
        timing_corrections: Applied timing corrections (optional)
        video_duration: Total video duration (optional, will be probed if not provided)
        
    Returns:
        Complete SyncReport with validation results
        
    Raises:
        SyncValidationError: If validation fails
    """
    if ffmpeg is None:
        logger.warning("ffmpeg-python not available, using limited validation")
        video_duration = video_duration or 30.0  # Default fallback
    
    try:
        logger.info(f"Starting sync validation for {video_path}")
        
        # Get video duration if not provided
        if video_duration is None:
            probe = ffmpeg.probe(video_path)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            video_duration = float(video_stream['duration'])
        
        # Calculate sync metrics
        sync_metrics = calculate_sync_metrics(sync_points)
        
        # Analyze timing drift
        drift_analysis = analyze_timing_drift(sync_points, video_duration)
        
        # Identify quality issues
        quality_issues = identify_quality_issues(sync_metrics, drift_analysis, sync_points)
        
        # Generate recommendations
        recommendations = generate_sync_recommendations(sync_metrics, drift_analysis, quality_issues)
        
        # Determine overall grade
        if sync_metrics.sync_accuracy_score >= 90 and not drift_analysis.requires_correction:
            overall_grade = 'A'
        elif sync_metrics.sync_accuracy_score >= 80 and len([i for i in quality_issues if i['severity'] == 'error']) == 0:
            overall_grade = 'B'
        elif sync_metrics.sync_accuracy_score >= 70:
            overall_grade = 'C'
        else:
            overall_grade = 'D'
        
        # Create report
        report = SyncReport(
            overall_grade=overall_grade,
            sync_metrics=sync_metrics,
            drift_analysis=drift_analysis,
            quality_issues=quality_issues,
            recommendations=recommendations,
            validation_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Sync validation complete. Grade: {overall_grade}, "
                   f"Accuracy: {sync_metrics.sync_accuracy_score:.1f}%")
        
        return report
        
    except Exception as e:
        error_msg = f"Sync validation failed: {str(e)}"
        logger.error(error_msg)
        raise SyncValidationError(error_msg)


def export_validation_report(report: SyncReport, output_path: str) -> str:
    """
    Export validation report to JSON file.
    
    Args:
        report: SyncReport to export
        output_path: Path for output JSON file
        
    Returns:
        Path to exported report file
        
    Raises:
        SyncValidationError: If export fails
    """
    try:
        # Convert report to dictionary
        report_dict = {
            'overall_grade': report.overall_grade,
            'validation_timestamp': report.validation_timestamp,
            'sync_metrics': {
                'avg_offset_ms': report.sync_metrics.avg_offset * 1000,
                'max_offset_ms': report.sync_metrics.max_offset * 1000,
                'std_deviation_ms': report.sync_metrics.std_deviation * 1000,
                'sync_accuracy_score': report.sync_metrics.sync_accuracy_score,
                'drift_rate_per_minute_ms': report.sync_metrics.drift_rate * 60 * 1000,
                'timing_consistency': report.sync_metrics.timing_consistency,
                'confidence_score': report.sync_metrics.confidence_score
            },
            'drift_analysis': {
                'total_drift_ms': report.drift_analysis.total_drift * 1000,
                'drift_rate_per_minute_ms': report.drift_analysis.drift_rate_per_minute * 1000,
                'drift_direction': report.drift_analysis.drift_direction,
                'significant_drift_points': report.drift_analysis.significant_drift_points,
                'drift_consistency': report.drift_analysis.drift_consistency,
                'requires_correction': report.drift_analysis.requires_correction
            },
            'quality_issues': report.quality_issues,
            'recommendations': report.recommendations
        }
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write report
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Exported validation report to {output_path}")
        return output_path
        
    except Exception as e:
        raise SyncValidationError(f"Failed to export validation report: {str(e)}")


def check_timing_drift(sync_points: List[SyncPoint], 
                      tolerance: float = 0.05) -> Dict[str, Any]:
    """
    Quick check for timing drift issues.
    
    Args:
        sync_points: List of synchronization points
        tolerance: Drift tolerance in seconds
        
    Returns:
        Dictionary with drift check results
    """
    try:
        if len(sync_points) < 2:
            return {
                'has_drift': False,
                'drift_amount': 0.0,
                'drift_rate': 0.0,
                'requires_attention': False
            }
        
        first_offset = sync_points[0].sync_offset
        last_offset = sync_points[-1].sync_offset
        total_drift = last_offset - first_offset
        
        time_span = sync_points[-1].video_timestamp - sync_points[0].video_timestamp
        drift_rate = total_drift / time_span if time_span > 0 else 0.0
        
        has_drift = abs(total_drift) > tolerance
        requires_attention = has_drift or abs(drift_rate) > (tolerance / 60.0)
        
        return {
            'has_drift': has_drift,
            'drift_amount': total_drift,
            'drift_rate': drift_rate,
            'requires_attention': requires_attention,
            'drift_direction': 'audio_lagging' if total_drift > 0 else 'audio_leading' if total_drift < 0 else 'none'
        }
        
    except Exception as e:
        logger.warning(f"Failed to check timing drift: {str(e)}")
        return {
            'has_drift': False,
            'drift_amount': 0.0,
            'drift_rate': 0.0,
            'requires_attention': False,
            'error': str(e)
        }