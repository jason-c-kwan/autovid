#!/usr/bin/env python3
"""
Slide synchronization engine for AutoVid.

This module provides functionality to synchronize Keynote slide transitions
with AI-generated narration by preserving original animation timings and only
adjusting when transitions should start based on narration completion or
[transition] markers in speaker notes.
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import ffmpeg

logger = logging.getLogger(__name__)


@dataclass
class MappingConfidence:
    """Quality metrics for sync mapping strategy selection."""
    scene_audio_alignment: float  # How well scene count matches audio count (0-1)
    transcript_quality: float     # Quality of transcript data (0-1)
    timing_consistency: float     # Consistency of detected timings (0-1)
    content_alignment: float      # How well content aligns across sources (0-1)
    overall_confidence: float     # Combined confidence score (0-1)
    recommended_strategy: str     # Best strategy for this data
    fallback_strategies: List[str] # Ordered list of fallback strategies


@dataclass
class SlideSegment:
    """Represents a video segment for a single slide."""
    slide_number: int
    keynote_start: float
    keynote_end: float
    keynote_duration: float
    narration_duration: float
    gap_needed: float
    gap_type: str = "static_hold"
    transition_cue: Optional[str] = None
    mapping_confidence: float = 0.0  # Confidence in this segment's sync accuracy
    strategy_used: Optional[str] = None  # Strategy used to create this segment
    
    def __post_init__(self):
        if self.keynote_duration <= 0:
            self.keynote_duration = self.keynote_end - self.keynote_start
        self.gap_needed = max(0, self.narration_duration - self.keynote_duration)


@dataclass
class SyncPlan:
    """Complete synchronization plan for a presentation."""
    video_path: str
    audio_path: str
    segments: List[SlideSegment]
    total_original_duration: float
    total_sync_duration: float
    keynote_delay_compensation: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncPlan':
        segments = [SlideSegment(**seg) for seg in data['segments']]
        return cls(
            video_path=data['video_path'],
            audio_path=data['audio_path'],
            segments=segments,
            total_original_duration=data['total_original_duration'],
            total_sync_duration=data['total_sync_duration'],
            keynote_delay_compensation=data.get('keynote_delay_compensation', 1.0)
        )


class StrategySelector:
    """Intelligent strategy selection for slide synchronization mapping."""
    
    def __init__(self):
        """Initialize the strategy selector."""
        self.strategy_weights = {
            'scene_audio_alignment': 0.35,
            'transcript_quality': 0.25,
            'timing_consistency': 0.25,
            'content_alignment': 0.15
        }
    
    def calculate_mapping_confidence(
        self,
        transition_points: List[float],
        narration_segments: List[Dict[str, Any]],
        transcript_cues: List[Dict[str, Any]],
        video_data: Dict[str, Any]
    ) -> MappingConfidence:
        """
        Calculate comprehensive confidence metrics for mapping strategy selection.
        
        Args:
            transition_points: Video transition timestamps
            narration_segments: Audio segment information
            transcript_cues: Transcript cues and metadata
            video_data: Video analysis data
            
        Returns:
            MappingConfidence object with detailed quality metrics
        """
        # Calculate scene-audio alignment score
        scene_count = len(transition_points) - 1 if len(transition_points) > 1 else 0
        audio_count = len(narration_segments)
        
        if audio_count == 0:
            scene_audio_alignment = 0.0
        elif scene_count == 0:
            scene_audio_alignment = 0.1  # Minimal score for no scene detection
        else:
            # Perfect match = 1.0, decreasing as difference increases
            alignment_ratio = min(scene_count, audio_count) / max(scene_count, audio_count)
            difference_penalty = abs(scene_count - audio_count) * 0.1
            scene_audio_alignment = max(0.0, alignment_ratio - difference_penalty)
        
        # Calculate transcript quality score
        transcript_quality = self._assess_transcript_quality(transcript_cues, narration_segments)
        
        # Calculate timing consistency score
        timing_consistency = self._assess_timing_consistency(transition_points, video_data)
        
        # Calculate content alignment score
        content_alignment = self._assess_content_alignment(
            narration_segments, transcript_cues, transition_points
        )
        
        # Calculate overall confidence using weighted average
        overall_confidence = (
            scene_audio_alignment * self.strategy_weights['scene_audio_alignment'] +
            transcript_quality * self.strategy_weights['transcript_quality'] +
            timing_consistency * self.strategy_weights['timing_consistency'] +
            content_alignment * self.strategy_weights['content_alignment']
        )
        
        # Select recommended strategy and fallbacks
        recommended_strategy, fallback_strategies = self._select_optimal_strategy(
            scene_audio_alignment, transcript_quality, timing_consistency, 
            content_alignment, overall_confidence
        )
        
        return MappingConfidence(
            scene_audio_alignment=scene_audio_alignment,
            transcript_quality=transcript_quality,
            timing_consistency=timing_consistency,
            content_alignment=content_alignment,
            overall_confidence=overall_confidence,
            recommended_strategy=recommended_strategy,
            fallback_strategies=fallback_strategies
        )
    
    def _assess_transcript_quality(
        self, 
        transcript_cues: List[Dict[str, Any]], 
        narration_segments: List[Dict[str, Any]]
    ) -> float:
        """Assess the quality and completeness of transcript data."""
        if not transcript_cues and not narration_segments:
            return 0.0
        
        score = 0.5  # Base score for having some transcript data
        
        # Check for transition cues
        transition_cues = sum(1 for cue in transcript_cues if cue.get('has_transition', False))
        if transition_cues > 0:
            score += min(0.3, transition_cues / len(narration_segments) * 0.3)
        
        # Check for slide numbers
        slides_with_numbers = sum(1 for cue in transcript_cues if cue.get('slide_number', 0) > 0)
        if slides_with_numbers > 0:
            score += min(0.2, slides_with_numbers / len(narration_segments) * 0.2)
        
        return min(1.0, score)
    
    def _assess_timing_consistency(
        self, 
        transition_points: List[float], 
        video_data: Dict[str, Any]
    ) -> float:
        """Assess consistency of detected transition timings."""
        if len(transition_points) < 2:
            return 0.3  # Low score for insufficient data
        
        # Check for confidence scores in video data
        scene_transitions = video_data.get('scene_transitions', [])
        if scene_transitions:
            confidences = [t.get('confidence', 0.5) for t in scene_transitions]
            avg_confidence = statistics.mean(confidences) if confidences else 0.5
            
            # Check timing regularity (consistent intervals suggest better detection)
            intervals = [transition_points[i+1] - transition_points[i] 
                        for i in range(len(transition_points)-1)]
            if len(intervals) > 1:
                interval_consistency = 1.0 - (statistics.stdev(intervals) / statistics.mean(intervals))
                interval_consistency = max(0.0, min(1.0, interval_consistency))
            else:
                interval_consistency = 0.5
            
            return (avg_confidence + interval_consistency) / 2.0
        
        return 0.5  # Default moderate score
    
    def _assess_content_alignment(
        self,
        narration_segments: List[Dict[str, Any]],
        transcript_cues: List[Dict[str, Any]],
        transition_points: List[float]
    ) -> float:
        """Assess how well content aligns across different data sources."""
        score = 0.5  # Base alignment score
        
        # Check if narration text matches transcript text
        text_matches = 0
        for i, segment in enumerate(narration_segments):
            if i < len(transcript_cues):
                narration_text = segment.get('text', '').lower().strip()
                transcript_text = transcript_cues[i].get('text', '').lower().strip()
                
                if narration_text and transcript_text:
                    # Simple text similarity check
                    common_words = set(narration_text.split()) & set(transcript_text.split())
                    if len(common_words) > 0:
                        text_matches += 1
        
        if narration_segments:
            text_alignment = text_matches / len(narration_segments)
            score = (score + text_alignment) / 2.0
        
        return min(1.0, score)
    
    def _select_optimal_strategy(
        self,
        scene_audio_alignment: float,
        transcript_quality: float,
        timing_consistency: float,
        content_alignment: float,
        overall_confidence: float
    ) -> Tuple[str, List[str]]:
        """Select the best strategy and ordered fallbacks based on confidence metrics."""
        
        # Strategy selection logic with confidence thresholds
        if scene_audio_alignment >= 0.8 and timing_consistency >= 0.7:
            recommended = "direct"
            fallbacks = ["transcript_guided", "duration_based", "interpolated"]
        elif transcript_quality >= 0.7 and content_alignment >= 0.6:
            recommended = "transcript_guided"
            fallbacks = ["duration_based", "direct", "interpolated"]
        elif overall_confidence >= 0.6:
            recommended = "duration_based"
            fallbacks = ["transcript_guided", "interpolated", "direct"]
        else:
            recommended = "interpolated"
            fallbacks = ["duration_based", "transcript_guided", "direct"]
        
        # Adjust based on specific conditions
        scene_count_good = scene_audio_alignment >= 0.6
        transcript_good = transcript_quality >= 0.6
        
        if not scene_count_good and transcript_good:
            recommended = "transcript_guided"
            fallbacks = ["duration_based", "interpolated", "direct"]
        elif scene_count_good and not transcript_good:
            recommended = "direct"
            fallbacks = ["duration_based", "interpolated", "transcript_guided"]
        
        return recommended, fallbacks


class AdvancedSyncValidator:
    """Advanced validation system for sync plans with comprehensive quality assessment."""
    
    def __init__(self):
        """Initialize the advanced validator with quality thresholds."""
        self.quality_thresholds = {
            'excellent': 90,
            'good': 80,
            'acceptable': 70,
            'poor': 50,
            'failing': 0
        }
        
        self.validation_weights = {
            'timing_accuracy': 0.3,
            'gap_quality': 0.25,
            'content_alignment': 0.2,
            'strategy_confidence': 0.15,
            'consistency': 0.1
        }
    
    def validate_comprehensive(
        self, 
        sync_plan: SyncPlan, 
        mapping_confidence: Optional[MappingConfidence] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation with advanced quality assessment.
        
        Args:
            sync_plan: SyncPlan to validate
            mapping_confidence: Optional confidence metrics from strategy selection
            
        Returns:
            Comprehensive validation results with actionable feedback
        """
        results = {
            'overall_quality': 'unknown',
            'quality_score': 0.0,
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'detailed_metrics': {},
            'quality_breakdown': {}
        }
        
        # Perform individual validation checks
        timing_results = self._validate_timing_accuracy(sync_plan)
        gap_results = self._validate_gap_quality(sync_plan)
        consistency_results = self._validate_consistency(sync_plan)
        drift_results = self._detect_sync_drift(sync_plan)
        strategy_results = self._validate_strategy_effectiveness(sync_plan, mapping_confidence)
        
        # Combine results
        results['detailed_metrics'] = {
            'timing_accuracy': timing_results,
            'gap_quality': gap_results,
            'consistency': consistency_results,
            'sync_drift': drift_results,
            'strategy_effectiveness': strategy_results
        }
        
        # Calculate overall quality score
        results['quality_score'] = self._calculate_overall_quality_score(results['detailed_metrics'])
        results['overall_quality'] = self._determine_quality_rating(results['quality_score'])
        
        # Generate quality breakdown
        results['quality_breakdown'] = self._generate_quality_breakdown(results['detailed_metrics'])
        
        # Collect issues and recommendations
        results['warnings'].extend(timing_results.get('warnings', []))
        results['warnings'].extend(gap_results.get('warnings', []))
        results['warnings'].extend(consistency_results.get('warnings', []))
        results['warnings'].extend(drift_results.get('warnings', []))
        
        results['errors'].extend(timing_results.get('errors', []))
        results['errors'].extend(gap_results.get('errors', []))
        results['errors'].extend(consistency_results.get('errors', []))
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Determine overall validity
        results['valid'] = len(results['errors']) == 0
        
        return results
    
    def _validate_timing_accuracy(self, sync_plan: SyncPlan) -> Dict[str, Any]:
        """Validate timing accuracy and expansion characteristics."""
        segments = sync_plan.segments
        
        if not segments:
            return {'score': 0, 'warnings': ['No segments to validate'], 'errors': []}
        
        # Calculate timing metrics
        timing_expansion = (sync_plan.total_sync_duration / sync_plan.total_original_duration - 1) * 100 if sync_plan.total_original_duration > 0 else 0
        
        # Check for reasonable timing expansion
        expansion_score = 100
        if timing_expansion > 100:  # More than double original duration
            expansion_score = 20
        elif timing_expansion > 50:  # 50-100% expansion
            expansion_score = 50
        elif timing_expansion > 25:  # 25-50% expansion
            expansion_score = 75
        elif timing_expansion < -10:  # Compression (unusual)
            expansion_score = 60
        
        # Check segment duration reasonableness
        duration_issues = []
        very_short_segments = sum(1 for seg in segments if seg.keynote_duration < 1.0)
        very_long_segments = sum(1 for seg in segments if seg.keynote_duration > 30.0)
        
        duration_score = 100 - (very_short_segments * 10) - (very_long_segments * 15)
        duration_score = max(0, duration_score)
        
        if very_short_segments > 0:
            duration_issues.append(f"{very_short_segments} segments are very short (< 1s)")
        if very_long_segments > 0:
            duration_issues.append(f"{very_long_segments} segments are very long (> 30s)")
        
        overall_score = (expansion_score + duration_score) / 2
        
        return {
            'score': overall_score,
            'timing_expansion': timing_expansion,
            'very_short_segments': very_short_segments,
            'very_long_segments': very_long_segments,
            'warnings': duration_issues,
            'errors': []
        }
    
    def _validate_gap_quality(self, sync_plan: SyncPlan) -> Dict[str, Any]:
        """Validate gap distribution and quality."""
        segments = sync_plan.segments
        
        if not segments:
            return {'score': 0, 'warnings': [], 'errors': []}
        
        gaps = [seg.gap_needed for seg in segments]
        
        # Calculate gap metrics
        total_gaps = sum(1 for g in gaps if g > 0)
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        
        # Categorize gaps
        extreme_gaps = sum(1 for g in gaps if g > 10.0)
        large_gaps = sum(1 for g in gaps if 5.0 < g <= 10.0)
        medium_gaps = sum(1 for g in gaps if 2.0 < g <= 5.0)
        small_gaps = sum(1 for g in gaps if 0 < g <= 2.0)
        
        # Calculate gap quality score
        gap_score = 100
        gap_score -= extreme_gaps * 25  # Heavy penalty for extreme gaps
        gap_score -= large_gaps * 10    # Moderate penalty for large gaps
        gap_score -= medium_gaps * 3    # Small penalty for medium gaps
        gap_score = max(0, gap_score)
        
        # Generate warnings
        warnings = []
        if extreme_gaps > 0:
            warnings.append(f"{extreme_gaps} segments have extreme gaps (>10s)")
        if avg_gap > 3.0:
            warnings.append(f"Average gap is high: {avg_gap:.1f}s")
        if max_gap > 15.0:
            warnings.append(f"Maximum gap is very large: {max_gap:.1f}s")
        
        return {
            'score': gap_score,
            'total_gaps': total_gaps,
            'avg_gap': avg_gap,
            'max_gap': max_gap,
            'extreme_gaps': extreme_gaps,
            'large_gaps': large_gaps,
            'warnings': warnings,
            'errors': []
        }
    
    def _validate_consistency(self, sync_plan: SyncPlan) -> Dict[str, Any]:
        """Validate consistency of sync plan."""
        segments = sync_plan.segments
        
        if len(segments) < 2:
            return {'score': 100, 'warnings': [], 'errors': []}
        
        # Check timing consistency
        gaps = [seg.gap_needed for seg in segments]
        durations = [seg.keynote_duration for seg in segments]
        
        # Calculate coefficient of variation for gaps and durations
        gap_cv = statistics.stdev(gaps) / statistics.mean(gaps) if statistics.mean(gaps) > 0 else 0
        duration_cv = statistics.stdev(durations) / statistics.mean(durations) if statistics.mean(durations) > 0 else 0
        
        # Lower CV indicates better consistency
        consistency_score = 100
        if gap_cv > 1.5:  # Very inconsistent gaps
            consistency_score -= 30
        elif gap_cv > 1.0:  # Somewhat inconsistent gaps
            consistency_score -= 15
        
        if duration_cv > 0.8:  # Very inconsistent durations
            consistency_score -= 20
        elif duration_cv > 0.5:  # Somewhat inconsistent durations
            consistency_score -= 10
        
        consistency_score = max(0, consistency_score)
        
        warnings = []
        if gap_cv > 1.0:
            warnings.append(f"Gap distribution is inconsistent (CV: {gap_cv:.2f})")
        if duration_cv > 0.5:
            warnings.append(f"Duration distribution is inconsistent (CV: {duration_cv:.2f})")
        
        return {
            'score': consistency_score,
            'gap_cv': gap_cv,
            'duration_cv': duration_cv,
            'warnings': warnings,
            'errors': []
        }
    
    def _detect_sync_drift(self, sync_plan: SyncPlan) -> Dict[str, Any]:
        """Detect cumulative sync drift across segments."""
        segments = sync_plan.segments
        
        if len(segments) < 3:
            return {'score': 100, 'drift_detected': False, 'warnings': [], 'errors': []}
        
        # Calculate cumulative timing difference
        cumulative_drift = 0
        max_drift = 0
        drift_points = []
        
        for i, segment in enumerate(segments):
            cumulative_drift += segment.gap_needed
            max_drift = max(max_drift, abs(cumulative_drift))
            drift_points.append(cumulative_drift)
        
        # Assess drift severity
        drift_score = 100
        drift_detected = False
        
        if max_drift > 10.0:
            drift_score = 30
            drift_detected = True
        elif max_drift > 5.0:
            drift_score = 60
            drift_detected = True
        elif max_drift > 2.0:
            drift_score = 80
        
        warnings = []
        if drift_detected:
            warnings.append(f"Sync drift detected: max cumulative drift {max_drift:.1f}s")
        
        return {
            'score': drift_score,
            'drift_detected': drift_detected,
            'max_drift': max_drift,
            'final_drift': cumulative_drift,
            'warnings': warnings,
            'errors': []
        }
    
    def _validate_strategy_effectiveness(
        self, 
        sync_plan: SyncPlan, 
        mapping_confidence: Optional[MappingConfidence]
    ) -> Dict[str, Any]:
        """Validate effectiveness of the chosen mapping strategy."""
        if not mapping_confidence:
            return {'score': 70, 'warnings': ['No mapping confidence data available'], 'errors': []}
        
        # Base score on overall confidence
        confidence_score = mapping_confidence.overall_confidence * 100
        
        # Check if segments have consistent strategy usage
        strategies_used = [seg.strategy_used for seg in sync_plan.segments if hasattr(seg, 'strategy_used') and seg.strategy_used]
        strategy_consistency = len(set(strategies_used)) <= 2 if strategies_used else True
        
        if not strategy_consistency:
            confidence_score *= 0.8  # Reduce score for inconsistent strategy usage
        
        warnings = []
        if mapping_confidence.overall_confidence < 0.6:
            warnings.append(f"Low mapping confidence: {mapping_confidence.overall_confidence:.2f}")
        if not strategy_consistency:
            warnings.append("Multiple fallback strategies were used, indicating data quality issues")
        
        return {
            'score': confidence_score,
            'mapping_confidence': mapping_confidence.overall_confidence,
            'recommended_strategy': mapping_confidence.recommended_strategy,
            'strategy_consistency': strategy_consistency,
            'warnings': warnings,
            'errors': []
        }
    
    def _calculate_overall_quality_score(self, detailed_metrics: Dict[str, Any]) -> float:
        """Calculate weighted overall quality score."""
        weighted_score = 0.0
        
        for metric_name, weight in self.validation_weights.items():
            if metric_name == 'timing_accuracy':
                score = detailed_metrics.get('timing_accuracy', {}).get('score', 0)
            elif metric_name == 'gap_quality':
                score = detailed_metrics.get('gap_quality', {}).get('score', 0)
            elif metric_name == 'content_alignment':
                score = 70  # Placeholder - would need content analysis
            elif metric_name == 'strategy_confidence':
                score = detailed_metrics.get('strategy_effectiveness', {}).get('score', 0)
            elif metric_name == 'consistency':
                score = detailed_metrics.get('consistency', {}).get('score', 0)
            else:
                score = 0
            
            weighted_score += score * weight
        
        return weighted_score
    
    def _determine_quality_rating(self, quality_score: float) -> str:
        """Determine quality rating based on score."""
        if quality_score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif quality_score >= self.quality_thresholds['good']:
            return 'good'
        elif quality_score >= self.quality_thresholds['acceptable']:
            return 'acceptable'
        elif quality_score >= self.quality_thresholds['poor']:
            return 'poor'
        else:
            return 'failing'
    
    def _generate_quality_breakdown(self, detailed_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Generate quality breakdown by component."""
        return {
            'timing_accuracy': detailed_metrics.get('timing_accuracy', {}).get('score', 0),
            'gap_quality': detailed_metrics.get('gap_quality', {}).get('score', 0),
            'consistency': detailed_metrics.get('consistency', {}).get('score', 0),
            'sync_drift': detailed_metrics.get('sync_drift', {}).get('score', 0),
            'strategy_effectiveness': detailed_metrics.get('strategy_effectiveness', {}).get('score', 0)
        }
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        quality_score = validation_results['quality_score']
        detailed_metrics = validation_results['detailed_metrics']
        
        # Overall quality recommendations
        if quality_score < 50:
            recommendations.append("Overall sync quality is poor. Consider reviewing input data and re-running with different strategy.")
        elif quality_score < 70:
            recommendations.append("Sync quality is below optimal. Review specific issues and consider adjustments.")
        
        # Timing-specific recommendations
        timing_metrics = detailed_metrics.get('timing_accuracy', {})
        if timing_metrics.get('timing_expansion', 0) > 50:
            recommendations.append("High timing expansion detected. Consider using more restrictive gap optimization.")
        
        # Gap-specific recommendations
        gap_metrics = detailed_metrics.get('gap_quality', {})
        if gap_metrics.get('extreme_gaps', 0) > 0:
            recommendations.append("Extreme gaps detected. Consider redistributing gaps or reviewing narration pacing.")
        
        # Drift-specific recommendations
        drift_metrics = detailed_metrics.get('sync_drift', {})
        if drift_metrics.get('drift_detected', False):
            recommendations.append("Sync drift detected. Consider using transcript-guided mapping or reviewing transition points.")
        
        # Strategy-specific recommendations
        strategy_metrics = detailed_metrics.get('strategy_effectiveness', {})
        if strategy_metrics.get('mapping_confidence', 1.0) < 0.6:
            recommendations.append("Low mapping confidence. Consider improving input data quality or using manual corrections.")
        
        return recommendations


class GapOptimizer:
    """Advanced gap handling and optimization for slide synchronization."""
    
    def __init__(self):
        """Initialize the gap optimizer with default thresholds."""
        self.gap_thresholds = {
            'micro': 0.5,      # < 0.5s: micro-adjustments
            'short': 2.0,      # 0.5-2s: short gaps 
            'medium': 5.0,     # 2-5s: medium gaps
            'long': 10.0,      # 5-10s: long gaps
            'extreme': 15.0    # > 10s: extreme gaps needing special handling
        }
        
    def optimize_gaps(self, segments: List[SlideSegment]) -> List[SlideSegment]:
        """
        Apply comprehensive gap optimization across all segments.
        
        Args:
            segments: List of slide segments to optimize
            
        Returns:
            Optimized list of slide segments
        """
        if not segments:
            return segments
        
        logger.info(f"Optimizing gaps for {len(segments)} segments")
        
        # Analyze gap distribution
        gap_analysis = self._analyze_gap_distribution(segments)
        
        # Apply progressive gap strategies
        optimized_segments = self._apply_progressive_gap_strategies(segments, gap_analysis)
        
        # Distribute large gaps if beneficial
        optimized_segments = self._distribute_large_gaps(optimized_segments)
        
        # Apply micro-timing adjustments
        optimized_segments = self._apply_micro_timing_adjustments(optimized_segments)
        
        # Validate gap quality
        final_analysis = self._analyze_gap_distribution(optimized_segments)
        logger.info(f"Gap optimization complete. Quality improvement: {self._calculate_quality_improvement(gap_analysis, final_analysis):.1f}%")
        
        return optimized_segments
    
    def _analyze_gap_distribution(self, segments: List[SlideSegment]) -> Dict[str, Any]:
        """Analyze the distribution and characteristics of gaps."""
        gaps = [seg.gap_needed for seg in segments]
        
        if not gaps:
            return {'total_gaps': 0, 'max_gap': 0, 'avg_gap': 0, 'gap_distribution': {}}
        
        gap_distribution = {
            'micro': sum(1 for g in gaps if 0 < g <= self.gap_thresholds['micro']),
            'short': sum(1 for g in gaps if self.gap_thresholds['micro'] < g <= self.gap_thresholds['short']),
            'medium': sum(1 for g in gaps if self.gap_thresholds['short'] < g <= self.gap_thresholds['medium']),
            'long': sum(1 for g in gaps if self.gap_thresholds['medium'] < g <= self.gap_thresholds['long']),
            'extreme': sum(1 for g in gaps if g > self.gap_thresholds['long'])
        }
        
        return {
            'total_gaps': sum(1 for g in gaps if g > 0),
            'max_gap': max(gaps),
            'avg_gap': sum(gaps) / len(gaps),
            'gap_distribution': gap_distribution,
            'problematic_gaps': sum(1 for g in gaps if g > self.gap_thresholds['medium'])
        }
    
    def _apply_progressive_gap_strategies(
        self, 
        segments: List[SlideSegment], 
        gap_analysis: Dict[str, Any]
    ) -> List[SlideSegment]:
        """Apply different strategies based on gap size categories."""
        
        for segment in segments:
            gap_size = segment.gap_needed
            
            if gap_size <= self.gap_thresholds['micro']:
                # Micro gaps: Use micro-hold strategy
                segment.gap_type = self._select_micro_gap_strategy(segment)
            elif gap_size <= self.gap_thresholds['short']:
                # Short gaps: Enhanced static/animated holds
                segment.gap_type = self._select_short_gap_strategy(segment)
            elif gap_size <= self.gap_thresholds['medium']:
                # Medium gaps: Complex animated strategies
                segment.gap_type = self._select_medium_gap_strategy(segment)
            elif gap_size <= self.gap_thresholds['long']:
                # Long gaps: Multi-phase strategies
                segment.gap_type = self._select_long_gap_strategy(segment)
            else:
                # Extreme gaps: Special handling required
                segment.gap_type = self._select_extreme_gap_strategy(segment)
        
        return segments
    
    def _select_micro_gap_strategy(self, segment: SlideSegment) -> str:
        """Select strategy for micro gaps (< 0.5s)."""
        # For very small gaps, extend the video slightly rather than adding gap
        if segment.gap_needed < 0.2:
            return 'extend_video'  # New gap type for video extension
        else:
            return 'micro_hold'   # New gap type for minimal holds
    
    def _select_short_gap_strategy(self, segment: SlideSegment) -> str:
        """Select strategy for short gaps (0.5-2s)."""
        text = getattr(segment, 'text', '').lower() if hasattr(segment, 'text') else ''
        
        if any(keyword in text for keyword in ['pause', 'moment', 'think']):
            return 'natural_pause'  # New gap type for natural pauses
        elif segment.slide_number == 1:
            return 'title_hold'     # New gap type for title slides
        else:
            return 'static_hold'    # Standard static hold
    
    def _select_medium_gap_strategy(self, segment: SlideSegment) -> str:
        """Select strategy for medium gaps (2-5s)."""
        text = getattr(segment, 'text', '').lower() if hasattr(segment, 'text') else ''
        
        if any(keyword in text for keyword in ['chart', 'graph', 'data', 'visualization']):
            return 'progressive_reveal'  # New gap type for progressive content revelation
        elif any(keyword in text for keyword in ['example', 'demonstration']):
            return 'demonstration_hold'  # New gap type for demonstrations
        else:
            return 'animated_hold'      # Enhanced animated hold
    
    def _select_long_gap_strategy(self, segment: SlideSegment) -> str:
        """Select strategy for long gaps (5-10s)."""
        return 'multi_phase_hold'  # New gap type for complex multi-phase holds
    
    def _select_extreme_gap_strategy(self, segment: SlideSegment) -> str:
        """Select strategy for extreme gaps (>10s)."""
        # Extreme gaps need to be distributed or broken down
        return 'distributed_gap'  # New gap type that will be broken into smaller parts
    
    def _distribute_large_gaps(self, segments: List[SlideSegment]) -> List[SlideSegment]:
        """Distribute excessively large gaps across multiple segments."""
        extreme_threshold = self.gap_thresholds['long']
        
        for i, segment in enumerate(segments):
            if segment.gap_needed > extreme_threshold:
                # Calculate how much gap we can redistribute
                redistributable = segment.gap_needed - extreme_threshold
                distributed = 0
                
                # Try to distribute to neighboring segments
                neighbors = []
                if i > 0:
                    neighbors.append(i - 1)
                if i < len(segments) - 1:
                    neighbors.append(i + 1)
                
                for neighbor_idx in neighbors:
                    if distributed >= redistributable:
                        break
                        
                    neighbor = segments[neighbor_idx]
                    # Only redistribute if neighbor has small gap
                    if neighbor.gap_needed < 2.0:
                        redistribution = min(redistributable - distributed, 2.0)
                        
                        # Add gap to neighbor
                        neighbor.gap_needed += redistribution
                        neighbor.keynote_duration -= redistribution  # Adjust timing
                        
                        distributed += redistribution
                        logger.debug(f"Redistributed {redistribution:.2f}s from segment {segment.slide_number} to {neighbor.slide_number}")
                
                # Reduce original segment's gap
                segment.gap_needed -= distributed
                segment.keynote_duration += distributed
        
        return segments
    
    def _apply_micro_timing_adjustments(self, segments: List[SlideSegment]) -> List[SlideSegment]:
        """Apply micro-adjustments to improve overall timing flow."""
        if len(segments) < 2:
            return segments
        
        # Look for opportunities to make small timing adjustments
        for i in range(len(segments) - 1):
            current = segments[i]
            next_segment = segments[i + 1]
            
            # If current segment has excess gap and next has deficit, balance them
            if current.gap_needed > 1.0 and next_segment.gap_needed < 0.5:
                adjustment = min(current.gap_needed - 1.0, 1.0 - next_segment.gap_needed) / 2
                
                if adjustment > 0.1:  # Only make significant adjustments
                    current.gap_needed -= adjustment
                    current.keynote_duration += adjustment
                    current.keynote_end += adjustment
                    
                    next_segment.keynote_start += adjustment
                    next_segment.gap_needed += adjustment
                    
                    logger.debug(f"Applied micro-adjustment of {adjustment:.2f}s between segments {current.slide_number} and {next_segment.slide_number}")
        
        return segments
    
    def _calculate_quality_improvement(
        self, 
        before_analysis: Dict[str, Any], 
        after_analysis: Dict[str, Any]
    ) -> float:
        """Calculate the percentage improvement in gap quality."""
        before_problematic = before_analysis.get('problematic_gaps', 0)
        after_problematic = after_analysis.get('problematic_gaps', 0)
        
        if before_problematic == 0:
            return 0.0
        
        improvement = (before_problematic - after_problematic) / before_problematic * 100
        return max(0.0, improvement)


class SlideSynchronizer:
    """Core engine for synchronizing Keynote slides with narration."""
    
    def __init__(self, keynote_delay: float = 1.0):
        """
        Initialize the slide synchronizer.
        
        Args:
            keynote_delay: Keynote export delay to compensate for (default: 1.0s)
        """
        self.keynote_delay = keynote_delay
        self.strategy_selector = StrategySelector()
        self.gap_optimizer = GapOptimizer()
        self.advanced_validator = AdvancedSyncValidator()
        
        # Initialize visual validator (optional - depends on dependencies)
        try:
            from .visual_validation import VisualSyncValidator
            self.visual_validator = VisualSyncValidator()
            self.visual_validation_available = True
        except ImportError:
            self.visual_validator = None
            self.visual_validation_available = False
            logger.warning("Visual validation not available - install dependencies: pip install Pillow ffmpeg-python")
        
    def create_sync_plan(
        self,
        video_analysis_manifest: str,
        audio_splice_manifest: str,
        transcript_manifest: str,
        output_path: Optional[str] = None,
        assembly_method: str = "intelligent"
    ) -> SyncPlan:
        """
        Create a synchronization plan based on video analysis and audio timing.
        
        Args:
            video_analysis_manifest: Path to video analysis manifest
            audio_splice_manifest: Path to audio splice manifest  
            transcript_manifest: Path to transcript manifest with transition cues
            output_path: Optional path to save the sync plan
            assembly_method: Strategy for sync mapping ("intelligent", "direct", "transcript_guided", "interpolated")
            
        Returns:
            SyncPlan object with complete synchronization instructions
        """
        logger.info(f"Creating slide synchronization plan using {assembly_method} method")
        
        # Load manifests
        video_data = self._load_manifest(video_analysis_manifest)
        audio_data = self._load_manifest(audio_splice_manifest)
        transcript_data = self._load_manifest(transcript_manifest)
        
        # Validate input data
        is_valid, error_msg = self._validate_sync_inputs(video_data, audio_data, transcript_data)
        if not is_valid:
            raise ValueError(f"Invalid sync inputs: {error_msg}")
        
        # Extract transition points and timing information
        transition_points = self._extract_transition_points(video_data)
        narration_segments = self._extract_narration_segments(audio_data, transcript_data)
        transcript_cues = self._extract_transcript_cues(transcript_data)
        
        # Calculate mapping confidence and select optimal strategy
        mapping_confidence = self.strategy_selector.calculate_mapping_confidence(
            transition_points, narration_segments, transcript_cues, video_data
        )
        
        # Use manual method if specified, otherwise use recommended strategy
        if assembly_method != "intelligent":
            strategy = assembly_method
            logger.info(f"Using manually specified strategy: {strategy}")
        else:
            strategy = mapping_confidence.recommended_strategy
            logger.info(f"Selected optimal strategy: {strategy} (confidence: {mapping_confidence.overall_confidence:.2f})")
            logger.info(f"Fallback strategies: {mapping_confidence.fallback_strategies}")
        
        # Create slide segments with enhanced strategy handling
        segments = self._create_segments_with_strategy(
            strategy, transition_points, narration_segments, transcript_cues, mapping_confidence
        )
        
        # Apply advanced gap optimization
        segments = self.gap_optimizer.optimize_gaps(segments)
        
        # Calculate overall timing after optimization
        total_original = transition_points[-1] if transition_points else 0
        total_sync = sum(seg.keynote_duration + seg.gap_needed for seg in segments)
        
        sync_plan = SyncPlan(
            video_path=video_data.get('video_path', ''),
            audio_path=audio_data.get('audio_path', ''),
            segments=segments,
            total_original_duration=total_original,
            total_sync_duration=total_sync,
            keynote_delay_compensation=self.keynote_delay
        )
        
        # Save sync plan if output path provided
        if output_path:
            self._save_sync_plan(sync_plan, output_path)
            
        logger.info(f"Created sync plan with {len(segments)} segments")
        logger.info(f"Original duration: {total_original:.2f}s, Sync duration: {total_sync:.2f}s")
        
        return sync_plan
    
    def _load_manifest(self, manifest_path: str) -> Dict[str, Any]:
        """Load and parse a manifest file."""
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest {manifest_path}: {e}")
            raise
    
    def _validate_sync_inputs(
        self,
        video_data: Dict[str, Any],
        audio_data: Dict[str, Any],
        transcript_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validate input manifests have required data for synchronization.
        
        Returns:
            (is_valid, error_message)
        """
        # Check if audio data has chunks
        audio_chunks = audio_data.get('chunks', [])
        if not audio_chunks and 'timing' in audio_data:
            audio_chunks = audio_data['timing'].get('chunks', [])
        
        if not audio_chunks:
            logger.debug(f"Audio data keys: {list(audio_data.keys())}")
            if 'timing' in audio_data:
                logger.debug(f"Timing keys: {list(audio_data['timing'].keys())}")
            return False, "Audio splice manifest contains no chunks"
        
        # Check if video data exists (scene transitions are optional)
        if not video_data:
            return False, "Video analysis manifest is empty"
        
        # Check if transcript data exists (optional but helpful)
        if not transcript_data:
            logger.warning("Transcript manifest is empty - will use fallback strategies")
        
        # Validate audio chunks have required fields
        for i, chunk in enumerate(audio_chunks):
            if 'duration' not in chunk:
                return False, f"Audio chunk {i} missing duration field"
        
        return True, ""
    
    def _select_sync_strategy(
        self,
        detected_scenes: List[float],
        narration_segments: List[Dict],
        transcript_cues: List[Dict],
        assembly_method: str
    ) -> str:
        """
        Intelligently select the best synchronization strategy.
        
        Returns: "direct", "transcript_guided", or "interpolated"
        """
        if assembly_method != "intelligent":
            return assembly_method
        
        scene_count = len(detected_scenes) - 1  # N+1 transitions define N scenes
        audio_count = len(narration_segments)
        
        logger.info(f"Strategy selection: {scene_count} scenes, {audio_count} audio segments")
        
        # Perfect or near-perfect match - use direct mapping
        if scene_count > 0 and abs(scene_count - audio_count) <= 2:
            return "direct"
        
        # Good transcript data available - use transcript-guided
        if transcript_cues and len(transcript_cues) >= audio_count * 0.5:
            return "transcript_guided"
        
        # Scene detection failed badly - use interpolation
        if scene_count < audio_count * 0.3:
            return "interpolated"
        
        # Default fallback
        return "direct"
    
    def _create_segments_with_strategy(
        self,
        strategy: str,
        transition_points: List[float],
        narration_segments: List[Dict[str, Any]],
        transcript_cues: List[Dict[str, Any]],
        mapping_confidence: MappingConfidence
    ) -> List[SlideSegment]:
        """
        Create slide segments using the specified strategy with fallback handling.
        
        Args:
            strategy: Primary strategy to use
            transition_points: Video transition timestamps
            narration_segments: Audio segment information
            transcript_cues: Transcript cues and metadata
            mapping_confidence: Confidence metrics for fallback decisions
            
        Returns:
            List of SlideSegment objects with confidence scoring
        """
        strategies_to_try = [strategy] + mapping_confidence.fallback_strategies
        
        for attempt_strategy in strategies_to_try:
            try:
                logger.info(f"Attempting {attempt_strategy} mapping strategy")
                
                if attempt_strategy == "direct":
                    segments = self._create_direct_mapping(transition_points, narration_segments)
                elif attempt_strategy == "transcript_guided":
                    segments = self._create_transcript_guided_mapping(transition_points, narration_segments, transcript_cues)
                elif attempt_strategy == "duration_based":
                    segments = self._create_duration_based_mapping(transition_points, narration_segments, transcript_cues)
                elif attempt_strategy == "interpolated":
                    segments = self._create_interpolated_mapping(transition_points, narration_segments)
                else:
                    # Fallback to basic method
                    segments = self._create_slide_segments(transition_points, narration_segments)
                
                # Add confidence and strategy metadata to segments
                for segment in segments:
                    segment.strategy_used = attempt_strategy
                    segment.mapping_confidence = self._calculate_segment_confidence(
                        segment, attempt_strategy, mapping_confidence
                    )
                
                logger.info(f"Successfully created {len(segments)} segments using {attempt_strategy} strategy")
                return segments
                
            except Exception as e:
                logger.warning(f"Strategy {attempt_strategy} failed: {e}")
                continue
        
        # Final fallback - should not reach here but safety net
        logger.error("All strategies failed, using basic slide segments")
        segments = self._create_slide_segments(transition_points, narration_segments)
        for segment in segments:
            segment.strategy_used = "fallback"
            segment.mapping_confidence = 0.3
        return segments
    
    def _calculate_segment_confidence(
        self,
        segment: SlideSegment,
        strategy_used: str,
        mapping_confidence: MappingConfidence
    ) -> float:
        """Calculate confidence score for an individual segment."""
        base_confidence = mapping_confidence.overall_confidence
        
        # Adjust based on segment characteristics
        if segment.gap_needed > 5.0:  # Large gap reduces confidence
            base_confidence *= 0.8
        elif segment.gap_needed < 0.5:  # Good timing match increases confidence
            base_confidence *= 1.1
        
        # Adjust based on strategy used
        if strategy_used == mapping_confidence.recommended_strategy:
            base_confidence *= 1.0  # No change for recommended strategy
        elif strategy_used in mapping_confidence.fallback_strategies[:2]:
            base_confidence *= 0.9  # Slight reduction for early fallbacks
        else:
            base_confidence *= 0.7  # Larger reduction for later fallbacks
        
        return max(0.0, min(1.0, base_confidence))
    
    def _extract_transition_points(self, video_data: Dict[str, Any]) -> List[float]:
        """
        Extract slide transition points from video analysis data.
        
        Args:
            video_data: Video analysis manifest data
            
        Returns:
            List of transition timestamps (compensated for Keynote delay)
        """
        transition_points = []
        
        # Get scene transitions from video analysis
        scene_transitions = video_data.get('scene_transitions', [])
        
        for transition in scene_transitions:
            timestamp = float(transition.get('timestamp', 0))
            # Compensate for Keynote delay
            compensated_timestamp = max(0, timestamp - self.keynote_delay)
            transition_points.append(compensated_timestamp)
        
        # Ensure we have at least one transition point at the start
        if not transition_points or transition_points[0] > 0:
            transition_points.insert(0, 0.0)
            
        # Sort to ensure chronological order
        transition_points.sort()
        
        logger.debug(f"Extracted {len(transition_points)} transition points: {transition_points}")
        return transition_points
    
    def _extract_narration_segments(
        self, 
        audio_data: Dict[str, Any], 
        transcript_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract narration segment timing information.
        
        Args:
            audio_data: Audio splice manifest data
            transcript_data: Transcript manifest data
            
        Returns:
            List of narration segment information
        """
        segments = []
        
        # Get audio chunks from splice manifest
        audio_chunks = audio_data.get('chunks', [])
        if not audio_chunks and 'timing' in audio_data:
            audio_chunks = audio_data['timing'].get('chunks', [])
        
        # Get transcript segments with transition cues - handle different structures
        transcript_segments = []
        if 'slides' in transcript_data:
            # New structure with slides array - convert to segments
            slides = transcript_data.get('slides', [])
            for slide in slides:
                slide_number = slide.get('index', 0) + 1
                segments = slide.get('segments', [])
                
                # Combine text segments for this slide
                slide_text = ""
                has_transition = False
                for segment in segments:
                    if segment.get('kind') == 'text':
                        slide_text += segment.get('text', '') + " "
                    elif segment.get('kind') == 'cue' and '[transition]' in segment.get('cue', '').lower():
                        has_transition = True
                
                # Create a transcript segment entry
                transcript_segment = {
                    'slide_number': slide_number,
                    'text': slide_text.strip(),
                    'transition_cue': '[transition]' if has_transition else None
                }
                transcript_segments.append(transcript_segment)
        else:
            # Legacy structure
            transcript_segments = transcript_data.get('segments', [])
        
        # Correlate audio chunks with transcript segments
        for i, chunk in enumerate(audio_chunks):
            segment_info = {
                'segment_number': i + 1,
                'start_time': float(chunk.get('start_time', 0)),
                'end_time': float(chunk.get('end_time', 0)),
                'duration': float(chunk.get('duration', 0)),
                'text': chunk.get('text', ''),
                'has_transition_cue': '[transition]' in chunk.get('text', '').lower(),
                'audio_path': chunk.get('audio_path', ''),
                'slide_number': i + 1,  # Default slide number
                'transition_cue': None
            }
            
            # Add transcript context if available
            if i < len(transcript_segments):
                transcript_seg = transcript_segments[i]
                # Override defaults with transcript data
                segment_info['slide_number'] = transcript_seg.get('slide_number', i + 1)
                segment_info['transition_cue'] = transcript_seg.get('transition_cue')
                
                # Use transcript text if audio chunk text is missing/empty
                if not segment_info['text'] and transcript_seg.get('text'):
                    segment_info['text'] = transcript_seg.get('text')
                    segment_info['has_transition_cue'] = '[transition]' in segment_info['text'].lower()
            
            # Validate essential fields
            if segment_info['duration'] <= 0:
                logger.warning(f"Segment {i+1} has invalid duration: {segment_info['duration']}")
                segment_info['duration'] = 1.0  # Fallback duration
            
            segments.append(segment_info)
        
        logger.debug(f"Extracted {len(segments)} narration segments")
        return segments
    
    def _extract_transcript_cues(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract transition cues and slide metadata from transcript.
        
        Args:
            transcript_data: Transcript manifest data
            
        Returns:
            List of structured data about expected slide transitions
        """
        cues = []
        
        # Handle different transcript structures
        if 'slides' in transcript_data:
            # New structure with slides array
            slides = transcript_data.get('slides', [])
            for slide in slides:
                slide_number = slide.get('index', 0) + 1  # Convert 0-based to 1-based
                segments = slide.get('segments', [])
                
                # Extract text and transition cues from segments
                slide_text = ""
                has_transition = False
                
                for segment in segments:
                    if segment.get('kind') == 'text':
                        slide_text += segment.get('text', '') + " "
                    elif segment.get('kind') == 'cue' and '[transition]' in segment.get('cue', '').lower():
                        has_transition = True
                
                cue_info = {
                    'segment_number': len(cues) + 1,
                    'slide_number': slide_number,
                    'transition_cue': '[transition]' if has_transition else None,
                    'has_transition': has_transition,
                    'text': slide_text.strip(),
                    'expected_duration': 0  # Not available in this structure
                }
                cues.append(cue_info)
        else:
            # Legacy structure with segments array
            transcript_segments = transcript_data.get('segments', [])
            
            for i, segment in enumerate(transcript_segments):
                cue_info = {
                    'segment_number': i + 1,
                    'slide_number': segment.get('slide_number', i + 1),
                    'transition_cue': segment.get('transition_cue'),
                    'has_transition': '[transition]' in segment.get('text', '').lower(),
                    'text': segment.get('text', ''),
                    'expected_duration': segment.get('duration', 0)
                }
                cues.append(cue_info)
        
        # Count total expected transitions
        transition_count = sum(1 for cue in cues if cue['has_transition'])
        logger.debug(f"Extracted {len(cues)} transcript cues with {transition_count} transitions")
        
        return cues
    
    def _create_direct_mapping(
        self, 
        transition_points: List[float], 
        narration_segments: List[Dict[str, Any]]
    ) -> List[SlideSegment]:
        """
        Create slide segments using direct 1:1 mapping (enhanced original logic).
        
        Args:
            transition_points: Video transition timestamps
            narration_segments: Narration segment information
            
        Returns:
            List of SlideSegment objects
        """
        return self._create_slide_segments(transition_points, narration_segments)
    
    def _create_transcript_guided_mapping(
        self,
        transition_points: List[float],
        narration_segments: List[Dict[str, Any]],
        transcript_cues: List[Dict[str, Any]]
    ) -> List[SlideSegment]:
        """
        Create sync mapping using transcript slide numbers and cues as ground truth.
        
        Enhanced version with advanced gap handling, slide boundary correction,
        and content-based timing optimization.
        
        Args:
            transition_points: Video transition timestamps (may be incomplete)
            narration_segments: Narration segment information
            transcript_cues: Transcript cues with slide numbers
            
        Returns:
            List of SlideSegment objects
        """
        segments = []
        
        # Group narration by slide number from transcript
        slide_groups = self._group_narration_by_slide(narration_segments)
        
        # Validate and correct slide boundaries if needed
        slide_groups = self._validate_slide_boundaries(slide_groups, transcript_cues, transition_points)
        
        logger.info(f"Grouped narration into {len(slide_groups)} slides after boundary validation")
        
        # Create optimized segments for each slide
        current_time = 0.0
        for slide_num in sorted(slide_groups.keys()):
            slide_segments = slide_groups[slide_num]
            slide_duration = sum(seg.get('duration', 0) for seg in slide_segments)
            
            # Enhanced timing calculation with transition cue optimization
            keynote_start, keynote_end = self._calculate_enhanced_slide_timing(
                slide_num, slide_duration, current_time, transition_points, 
                transcript_cues, slide_segments
            )
            
            # Advanced gap type determination
            gap_type = self._determine_advanced_gap_type(slide_segments, transcript_cues, slide_num)
            
            # Get optimized transition timing
            transition_cue, cue_timing = self._extract_transition_cue_timing(
                transcript_cues, slide_num, slide_segments
            )
            
            segment = SlideSegment(
                slide_number=slide_num,
                keynote_start=keynote_start,
                keynote_end=keynote_end,
                keynote_duration=keynote_end - keynote_start,
                narration_duration=slide_duration,
                gap_needed=0,  # Will be calculated in __post_init__
                gap_type=gap_type,
                transition_cue=transition_cue
            )
            
            segments.append(segment)
            current_time = keynote_end
        
        # Apply gap optimization across all segments
        segments = self._optimize_gap_distribution(segments)
        
        logger.info(f"Created {len(segments)} segments using enhanced transcript-guided mapping")
        return segments
    
    def _group_narration_by_slide(self, narration_segments: List[Dict[str, Any]]) -> Dict[int, List[Dict]]:
        """Group narration segments by slide number with validation."""
        slide_groups = {}
        
        for i, segment in enumerate(narration_segments):
            slide_num = segment.get('slide_number', i + 1)  # Fallback to sequential numbering
            
            if slide_num not in slide_groups:
                slide_groups[slide_num] = []
            slide_groups[slide_num].append(segment)
        
        return slide_groups
    
    def _validate_slide_boundaries(
        self, 
        slide_groups: Dict[int, List[Dict]], 
        transcript_cues: List[Dict[str, Any]],
        transition_points: List[float]
    ) -> Dict[int, List[Dict]]:
        """Validate and correct slide boundaries based on multiple data sources."""
        # Check for missing slide numbers
        expected_slides = set(range(1, len(slide_groups) + 1))
        actual_slides = set(slide_groups.keys())
        
        if expected_slides != actual_slides:
            logger.warning(f"Slide number mismatch. Expected: {expected_slides}, Got: {actual_slides}")
            
            # Renumber slides to be sequential if there are gaps
            corrected_groups = {}
            for new_num, old_num in enumerate(sorted(actual_slides), 1):
                corrected_groups[new_num] = slide_groups[old_num]
                # Update slide numbers in segments
                for segment in corrected_groups[new_num]:
                    segment['slide_number'] = new_num
            
            slide_groups = corrected_groups
        
        return slide_groups
    
    def _calculate_enhanced_slide_timing(
        self,
        slide_num: int,
        slide_duration: float,
        current_time: float,
        transition_points: List[float],
        transcript_cues: List[Dict[str, Any]],
        slide_segments: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Calculate optimized slide timing using multiple data sources."""
        keynote_start = current_time
        
        # Try to use actual video transitions with validation
        if len(transition_points) > slide_num:
            video_start = transition_points[slide_num - 1] if slide_num > 1 else 0.0
            video_end = transition_points[slide_num] if slide_num < len(transition_points) else None
            
            # Validate timing makes sense
            if video_end and video_end > video_start:
                video_duration = video_end - video_start
                
                # Use video timing if it's reasonable relative to narration
                if 0.3 <= video_duration / slide_duration <= 3.0:
                    keynote_start = video_start
                    keynote_end = video_end
                    return keynote_start, keynote_end
        
        # Fallback to narration-based timing with content adjustments
        base_end = keynote_start + slide_duration
        
        # Adjust based on content complexity
        complexity_multiplier = self._calculate_content_complexity_multiplier(slide_segments)
        adjusted_duration = slide_duration * complexity_multiplier
        
        keynote_end = keynote_start + adjusted_duration
        
        return keynote_start, keynote_end
    
    def _calculate_content_complexity_multiplier(self, slide_segments: List[Dict[str, Any]]) -> float:
        """Calculate a multiplier based on slide content complexity."""
        combined_text = ' '.join(seg.get('text', '') for seg in slide_segments).lower()
        
        multiplier = 1.0
        
        # Adjust for content types that need more time
        if any(keyword in combined_text for keyword in ['chart', 'graph', 'table', 'diagram']):
            multiplier *= 1.3
        elif any(keyword in combined_text for keyword in ['example', 'demonstration', 'detailed']):
            multiplier *= 1.2
        elif any(keyword in combined_text for keyword in ['overview', 'summary', 'introduction']):
            multiplier *= 0.9
        
        return max(0.7, min(1.5, multiplier))
    
    def _determine_advanced_gap_type(
        self, 
        slide_segments: List[Dict[str, Any]], 
        transcript_cues: List[Dict[str, Any]], 
        slide_num: int
    ) -> str:
        """Enhanced gap type determination with context analysis."""
        combined_text = ' '.join(seg.get('text', '') for seg in slide_segments).lower()
        
        # Check for specific content indicators
        if any(keyword in combined_text for keyword in ['animation', 'animated', 'moving', 'transition']):
            return 'animated_hold'
        elif any(keyword in combined_text for keyword in ['chart', 'graph', 'data', 'visualization']):
            return 'animated_hold'  # Charts often benefit from animated holds
        elif any(keyword in combined_text for keyword in ['conclusion', 'summary', 'thank you', 'questions', 'end']):
            return 'fade_hold'
        elif any(keyword in combined_text for keyword in ['pause', 'think', 'consider', 'reflect']):
            return 'static_hold'  # Contemplative content
        else:
            # Check position in presentation
            if slide_num == 1:
                return 'static_hold'  # Title slides usually static
            else:
                return 'static_hold'  # Default
    
    def _extract_transition_cue_timing(
        self, 
        transcript_cues: List[Dict[str, Any]], 
        slide_num: int, 
        slide_segments: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[float]]:
        """Extract transition cue and optimal timing within the slide."""
        transition_cue = None
        cue_timing = None
        
        # Find transition cue for this slide
        for cue in transcript_cues:
            if cue.get('slide_number') == slide_num and cue.get('has_transition'):
                transition_cue = cue.get('transition_cue')
                
                # Try to estimate when in the slide the transition should occur
                if '[transition]' in cue.get('text', ''):
                    # Estimate timing based on text position
                    text = cue.get('text', '')
                    transition_pos = text.lower().find('[transition]')
                    if transition_pos >= 0:
                        # Rough estimate: position in text correlates to time
                        relative_position = transition_pos / len(text) if text else 0.5
                        total_duration = sum(seg.get('duration', 0) for seg in slide_segments)
                        cue_timing = total_duration * relative_position
                
                break
        
        return transition_cue, cue_timing
    
    def _optimize_gap_distribution(self, segments: List[SlideSegment]) -> List[SlideSegment]:
        """Optimize gap distribution to minimize jarring transitions."""
        if len(segments) <= 1:
            return segments
        
        # Calculate gaps (will be done in __post_init__ but we need estimates)
        for segment in segments:
            segment.gap_needed = max(0, segment.narration_duration - segment.keynote_duration)
        
        # Find segments with large gaps that could be redistributed
        large_gap_threshold = 3.0
        large_gap_segments = [s for s in segments if s.gap_needed > large_gap_threshold]
        
        if not large_gap_segments:
            return segments
        
        logger.info(f"Optimizing {len(large_gap_segments)} segments with large gaps")
        
        # For segments with very large gaps, try to distribute some gap to adjacent segments
        for i, segment in enumerate(segments):
            if segment.gap_needed > large_gap_threshold:
                excess_gap = segment.gap_needed - large_gap_threshold
                redistributed = 0
                
                # Try to extend previous segment slightly
                if i > 0 and excess_gap > 0 and segments[i-1].gap_needed < 1.0:
                    extension = min(excess_gap * 0.3, 1.0)
                    segments[i-1].keynote_duration += extension
                    segments[i-1].keynote_end += extension
                    redistributed += extension
                
                # Try to extend next segment slightly
                if i < len(segments) - 1 and excess_gap > redistributed and segments[i+1].gap_needed < 1.0:
                    extension = min((excess_gap - redistributed) * 0.3, 1.0)
                    segments[i+1].keynote_duration += extension
                    segments[i+1].keynote_end += extension
                    redistributed += extension
                
                # Reduce the current segment's gap accordingly
                if redistributed > 0:
                    segment.keynote_duration += redistributed
                    segment.keynote_end += redistributed
                    logger.debug(f"Redistributed {redistributed:.2f}s gap from segment {segment.slide_number}")
        
        return segments
    
    def _create_duration_based_mapping(
        self,
        transition_points: List[float],
        narration_segments: List[Dict[str, Any]],
        transcript_cues: List[Dict[str, Any]]
    ) -> List[SlideSegment]:
        """
        Create sync mapping using intelligent duration estimation and content analysis.
        
        This strategy analyzes content complexity and pacing patterns to predict
        realistic slide durations when scene detection is unreliable.
        
        Args:
            transition_points: Video transition timestamps (may be unreliable)
            narration_segments: Narration segment information
            transcript_cues: Transcript cues for content analysis
            
        Returns:
            List of SlideSegment objects
        """
        segments = []
        
        # Analyze content complexity for each segment
        duration_estimates = self._estimate_slide_durations(narration_segments, transcript_cues)
        
        # Detect presentation pacing patterns
        pacing_multiplier = self._detect_pacing_pattern(narration_segments, duration_estimates)
        
        current_time = 0.0
        
        for i, (narration_seg, base_duration) in enumerate(zip(narration_segments, duration_estimates)):
            # Apply pacing adjustment
            estimated_video_duration = base_duration * pacing_multiplier
            
            # Use actual transition points if available and reasonable
            keynote_start = current_time
            if len(transition_points) > i and len(transition_points) > i + 1:
                actual_duration = transition_points[i + 1] - transition_points[i]
                # Use actual duration if it's reasonable, otherwise use estimate
                if 0.5 <= actual_duration <= estimated_video_duration * 3:
                    keynote_end = keynote_start + actual_duration
                else:
                    keynote_end = keynote_start + estimated_video_duration
            else:
                keynote_end = keynote_start + estimated_video_duration
            
            # Determine gap type based on content analysis
            gap_type = self._analyze_content_for_gap_type(narration_seg, transcript_cues, i)
            
            segment = SlideSegment(
                slide_number=narration_seg.get('slide_number', i + 1),
                keynote_start=keynote_start,
                keynote_end=keynote_end,
                keynote_duration=keynote_end - keynote_start,
                narration_duration=narration_seg.get('duration', 0),
                gap_needed=0,  # Will be calculated in __post_init__
                gap_type=gap_type,
                transition_cue=narration_seg.get('transition_cue')
            )
            
            segments.append(segment)
            current_time = keynote_end
        
        logger.info(f"Created {len(segments)} segments using duration-based mapping with {pacing_multiplier:.2f}x pacing")
        return segments
    
    def _estimate_slide_durations(
        self, 
        narration_segments: List[Dict[str, Any]], 
        transcript_cues: List[Dict[str, Any]]
    ) -> List[float]:
        """Estimate realistic slide durations based on content complexity."""
        durations = []
        
        for i, segment in enumerate(narration_segments):
            text = segment.get('text', '')
            narration_duration = segment.get('duration', 3.0)
            
            # Base duration on narration length
            base_duration = narration_duration
            
            # Adjust for content complexity
            word_count = len(text.split()) if text else 10
            if word_count > 50:  # Complex slide
                base_duration *= 1.3
            elif word_count < 10:  # Simple slide
                base_duration *= 0.8
            
            # Check for technical content
            technical_keywords = ['algorithm', 'equation', 'formula', 'data', 'chart', 'graph', 'analysis']
            if any(keyword in text.lower() for keyword in technical_keywords):
                base_duration *= 1.2
            
            # Check for transition cues that might affect pacing
            if i < len(transcript_cues) and transcript_cues[i].get('has_transition'):
                base_duration *= 1.1  # Slightly longer for slides before transitions
            
            # Ensure reasonable bounds
            base_duration = max(2.0, min(15.0, base_duration))
            durations.append(base_duration)
        
        return durations
    
    def _detect_pacing_pattern(
        self, 
        narration_segments: List[Dict[str, Any]], 
        base_durations: List[float]
    ) -> float:
        """Detect overall presentation pacing to adjust slide durations."""
        if not narration_segments:
            return 1.0
        
        # Calculate average speaking rate (words per minute)
        total_words = sum(len(seg.get('text', '').split()) for seg in narration_segments)
        total_duration = sum(seg.get('duration', 0) for seg in narration_segments)
        
        if total_duration > 0:
            speaking_rate = (total_words / total_duration) * 60  # WPM
            
            # Adjust pacing based on speaking rate
            if speaking_rate > 180:  # Fast speaker
                return 0.9
            elif speaking_rate < 120:  # Slow speaker
                return 1.2
            else:  # Normal pace
                return 1.0
        
        return 1.0
    
    def _analyze_content_for_gap_type(
        self, 
        narration_seg: Dict[str, Any], 
        transcript_cues: List[Dict[str, Any]], 
        segment_index: int
    ) -> str:
        """Analyze content to determine optimal gap type for duration-based mapping."""
        text = narration_seg.get('text', '').lower()
        
        # Enhanced gap type detection with content analysis
        if any(keyword in text for keyword in ['chart', 'graph', 'data', 'visualization', 'diagram']):
            return 'animated_hold'
        elif any(keyword in text for keyword in ['conclusion', 'summary', 'thank you', 'questions', 'final']):
            return 'fade_hold'
        elif any(keyword in text for keyword in ['example', 'demonstration', 'show', 'look at']):
            return 'animated_hold'
        else:
            return 'static_hold'
    
    def _create_interpolated_mapping(
        self,
        transition_points: List[float],
        narration_segments: List[Dict[str, Any]]
    ) -> List[SlideSegment]:
        """
        Create sync mapping using audio duration estimates when scene detection fails.
        
        Fallback strategy that creates synthetic video segments based on 
        narration pacing and duration.
        
        Args:
            transition_points: Video transition timestamps (may be minimal/empty)
            narration_segments: Narration segment information
            
        Returns:
            List of SlideSegment objects
        """
        segments = []
        
        # Calculate total narration duration
        total_narration = sum(seg.get('duration', 0) for seg in narration_segments)
        
        # Use video duration if available, otherwise use narration duration
        total_video_duration = transition_points[-1] if transition_points else total_narration
        
        # Create evenly spaced segments based on narration
        current_time = 0.0
        
        for i, narration_seg in enumerate(narration_segments):
            narration_duration = narration_seg.get('duration', 0)
            
            # Estimate video timing proportionally
            if total_narration > 0:
                video_duration = (narration_duration / total_narration) * total_video_duration
            else:
                video_duration = 5.0  # Default fallback
            
            keynote_start = current_time
            keynote_end = current_time + video_duration
            
            gap_type = self._determine_gap_type(narration_seg.get('text', ''))
            
            segment = SlideSegment(
                slide_number=narration_seg.get('slide_number', i + 1),
                keynote_start=keynote_start,
                keynote_end=keynote_end,
                keynote_duration=video_duration,
                narration_duration=narration_duration,
                gap_needed=0,  # Will be calculated in __post_init__
                gap_type=gap_type,
                transition_cue=narration_seg.get('transition_cue')
            )
            
            segments.append(segment)
            current_time = keynote_end
        
        logger.info(f"Created {len(segments)} segments using interpolated mapping")
        return segments
    
    def _create_slide_segments(
        self, 
        transition_points: List[float], 
        narration_segments: List[Dict[str, Any]]
    ) -> List[SlideSegment]:
        """
        Create slide segments by matching transition points with narration timing.
        
        Args:
            transition_points: Video transition timestamps
            narration_segments: Narration segment information
            
        Returns:
            List of SlideSegment objects
        """
        segments = []
        
        # Ensure we have enough transition points
        while len(transition_points) < len(narration_segments) + 1:
            # Estimate next transition point based on average segment duration
            if len(transition_points) >= 2:
                avg_duration = (transition_points[-1] - transition_points[0]) / (len(transition_points) - 1)
                next_point = transition_points[-1] + avg_duration
            else:
                next_point = transition_points[-1] + 5.0  # Default 5-second segments
            transition_points.append(next_point)
        
        # Create segments
        for i, narration_seg in enumerate(narration_segments):
            slide_start = transition_points[i]
            slide_end = transition_points[i + 1] if i + 1 < len(transition_points) else slide_start + 5.0
            
            # Determine gap type based on content
            gap_type = self._determine_gap_type(narration_seg.get('text', ''))
            
            segment = SlideSegment(
                slide_number=narration_seg.get('slide_number', i + 1),
                keynote_start=slide_start,
                keynote_end=slide_end,
                keynote_duration=slide_end - slide_start,
                narration_duration=narration_seg.get('duration', 0),
                gap_needed=0,  # Will be calculated in __post_init__
                gap_type=gap_type,
                transition_cue=narration_seg.get('transition_cue')
            )
            
            segments.append(segment)
        
        return segments
    
    def _determine_gap_type(self, text: str) -> str:
        """
        Determine the appropriate gap type based on slide content.
        
        Args:
            text: Narration text for the slide
            
        Returns:
            Gap type string ('static_hold', 'animated_hold', 'fade_hold')
        """
        text_lower = text.lower()
        
        # Use animated hold for slides with dynamic content
        if any(keyword in text_lower for keyword in [
            'animation', 'moving', 'transition', 'chart', 'graph', 'data'
        ]):
            return 'animated_hold'
        
        # Use fade hold for concluding slides
        if any(keyword in text_lower for keyword in [
            'conclusion', 'summary', 'thank you', 'questions', 'end'
        ]):
            return 'fade_hold'
        
        # Default to static hold
        return 'static_hold'
    
    def _save_sync_plan(self, sync_plan: SyncPlan, output_path: str):
        """Save synchronization plan to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(sync_plan.to_dict(), f, indent=2)
            logger.info(f"Saved sync plan to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save sync plan: {e}")
            raise
    
    def validate_sync_plan(self, sync_plan: SyncPlan, mapping_confidence: Optional[MappingConfidence] = None) -> Dict[str, Any]:
        """
        Validate a synchronization plan using advanced validation logic.
        
        Args:
            sync_plan: SyncPlan to validate
            mapping_confidence: Optional confidence metrics from strategy selection
            
        Returns:
            Dictionary with comprehensive validation results and metrics
        """
        # Use advanced validator for comprehensive analysis
        return self.advanced_validator.validate_comprehensive(sync_plan, mapping_confidence)
    
    def validate_sync_plan_with_visual(
        self,
        sync_plan: SyncPlan,
        video_path: str,
        pptx_path: str,
        transcript_data: Optional[Dict[str, Any]] = None,
        mapping_confidence: Optional[MappingConfidence] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation including visual validation if available.
        
        Args:
            sync_plan: SyncPlan to validate
            video_path: Path to the presentation video
            pptx_path: Path to the original PowerPoint file
            transcript_data: Optional transcript data for context
            mapping_confidence: Optional confidence metrics from strategy selection
            
        Returns:
            Comprehensive validation results including visual analysis
        """
        logger.info("Performing comprehensive sync plan validation")
        
        # Start with advanced validation
        validation_results = self.advanced_validator.validate_comprehensive(sync_plan, mapping_confidence)
        
        # Add visual validation if available
        if self.visual_validation_available and self.visual_validator:
            try:
                logger.info("Adding visual validation layer")
                visual_results = self.visual_validator.validate_sync_plan_visually(
                    sync_plan, video_path, pptx_path, transcript_data
                )
                
                # Integrate visual results
                validation_results['visual_validation'] = visual_results
                
                # Adjust overall quality score based on visual validation
                if visual_results.get('overall_score', 0) > 0:
                    original_score = validation_results.get('quality_score', 0)
                    visual_score = visual_results['overall_score'] * 100  # Convert to 0-100 scale
                    
                    # Weighted combination: 70% original validation, 30% visual validation
                    combined_score = (original_score * 0.7) + (visual_score * 0.3)
                    validation_results['quality_score'] = combined_score
                    validation_results['overall_quality'] = self.advanced_validator._determine_quality_rating(combined_score)
                
                # Add visual issues and recommendations
                validation_results['warnings'].extend(visual_results.get('issues', []))
                validation_results['recommendations'].extend(visual_results.get('recommendations', []))
                
                logger.info(f"Visual validation complete. Combined quality score: {validation_results['quality_score']:.1f}")
                
            except Exception as e:
                logger.error(f"Visual validation failed: {e}")
                validation_results['warnings'].append(f"Visual validation error: {str(e)}")
        else:
            validation_results['visual_validation'] = {
                'available': False,
                'reason': 'Visual validation dependencies not installed'
            }
            if not self.visual_validation_available:
                validation_results['recommendations'].append(
                    "Install visual validation dependencies for enhanced accuracy: pip install Pillow ffmpeg-python"
                )
        
        return validation_results


def load_sync_plan(file_path: str) -> SyncPlan:
    """Load a synchronization plan from file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return SyncPlan.from_dict(data)