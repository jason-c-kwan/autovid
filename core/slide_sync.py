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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import ffmpeg

logger = logging.getLogger(__name__)


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


class SlideSynchronizer:
    """Core engine for synchronizing Keynote slides with narration."""
    
    def __init__(self, keynote_delay: float = 1.0):
        """
        Initialize the slide synchronizer.
        
        Args:
            keynote_delay: Keynote export delay to compensate for (default: 1.0s)
        """
        self.keynote_delay = keynote_delay
        
    def create_sync_plan(
        self,
        video_analysis_manifest: str,
        audio_splice_manifest: str,
        transcript_manifest: str,
        output_path: Optional[str] = None
    ) -> SyncPlan:
        """
        Create a synchronization plan based on video analysis and audio timing.
        
        Args:
            video_analysis_manifest: Path to video analysis manifest
            audio_splice_manifest: Path to audio splice manifest  
            transcript_manifest: Path to transcript manifest with transition cues
            output_path: Optional path to save the sync plan
            
        Returns:
            SyncPlan object with complete synchronization instructions
        """
        logger.info("Creating slide synchronization plan")
        
        # Load manifests
        video_data = self._load_manifest(video_analysis_manifest)
        audio_data = self._load_manifest(audio_splice_manifest)
        transcript_data = self._load_manifest(transcript_manifest)
        
        # Extract transition points and timing information
        transition_points = self._extract_transition_points(video_data)
        narration_segments = self._extract_narration_segments(audio_data, transcript_data)
        
        # Create slide segments with timing analysis
        segments = self._create_slide_segments(transition_points, narration_segments)
        
        # Calculate overall timing
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
        
        # Get transcript segments with transition cues
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
                'audio_path': chunk.get('audio_path', '')
            }
            
            # Add transcript context if available
            if i < len(transcript_segments):
                transcript_seg = transcript_segments[i]
                segment_info['slide_number'] = transcript_seg.get('slide_number', i + 1)
                segment_info['transition_cue'] = transcript_seg.get('transition_cue')
            else:
                segment_info['slide_number'] = i + 1
            
            segments.append(segment_info)
        
        logger.debug(f"Extracted {len(segments)} narration segments")
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
    
    def validate_sync_plan(self, sync_plan: SyncPlan) -> Dict[str, Any]:
        """
        Validate a synchronization plan and return quality metrics.
        
        Args:
            sync_plan: SyncPlan to validate
            
        Returns:
            Dictionary with validation results and metrics
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {
                'total_segments': len(sync_plan.segments),
                'total_gaps': sum(1 for seg in sync_plan.segments if seg.gap_needed > 0),
                'max_gap': max((seg.gap_needed for seg in sync_plan.segments), default=0),
                'avg_gap': sum(seg.gap_needed for seg in sync_plan.segments) / len(sync_plan.segments) if sync_plan.segments else 0,
                'timing_expansion': (sync_plan.total_sync_duration / sync_plan.total_original_duration - 1) * 100 if sync_plan.total_original_duration > 0 else 0
            }
        }
        
        # Check for potential issues
        for i, segment in enumerate(sync_plan.segments):
            if segment.gap_needed > 10.0:  # Very long gap
                results['warnings'].append(f"Segment {i+1}: Large gap needed ({segment.gap_needed:.1f}s)")
            
            if segment.keynote_duration < 0.5:  # Very short segment
                results['warnings'].append(f"Segment {i+1}: Very short Keynote duration ({segment.keynote_duration:.1f}s)")
        
        # Check overall timing expansion
        if results['metrics']['timing_expansion'] > 50:  # More than 50% expansion
            results['warnings'].append(f"Significant timing expansion: {results['metrics']['timing_expansion']:.1f}%")
        
        return results


def load_sync_plan(file_path: str) -> SyncPlan:
    """Load a synchronization plan from file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return SyncPlan.from_dict(data)