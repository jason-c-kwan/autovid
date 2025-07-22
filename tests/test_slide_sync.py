#!/usr/bin/env python3
"""
Tests for slide synchronization system.

This module tests the core slide synchronization functionality,
including sync plan creation, gap management, and video assembly.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.slide_sync import SlideSynchronizer, SlideSegment, SyncPlan
from core.gap_management import GapManager, GapFiller, optimize_gap_types
from core.video_assembly import VideoAssembler, VideoSegment
from core.video_analysis import detect_scene_changes


class TestSlideSegment:
    """Test SlideSegment dataclass functionality."""
    
    def test_slide_segment_creation(self):
        """Test basic slide segment creation."""
        segment = SlideSegment(
            slide_number=1,
            keynote_start=0.0,
            keynote_end=5.0,
            keynote_duration=0.0,  # Will be calculated
            narration_duration=7.5
        )
        
        assert segment.slide_number == 1
        assert segment.keynote_start == 0.0
        assert segment.keynote_end == 5.0
        assert segment.keynote_duration == 5.0
        assert segment.narration_duration == 7.5
        assert segment.gap_needed == 2.5
        assert segment.gap_type == "static_hold"
    
    def test_slide_segment_no_gap_needed(self):
        """Test segment where no gap is needed."""
        segment = SlideSegment(
            slide_number=2,
            keynote_start=5.0,
            keynote_end=10.0,
            keynote_duration=5.0,
            narration_duration=4.0
        )
        
        assert segment.gap_needed == 0.0


class TestSyncPlan:
    """Test SyncPlan functionality."""
    
    def test_sync_plan_creation(self):
        """Test sync plan creation and serialization."""
        segments = [
            SlideSegment(
                slide_number=1,
                keynote_start=0.0,
                keynote_end=5.0,
                keynote_duration=5.0,
                narration_duration=7.0
            ),
            SlideSegment(
                slide_number=2,
                keynote_start=5.0,
                keynote_end=10.0,
                keynote_duration=5.0,
                narration_duration=6.0
            )
        ]
        
        sync_plan = SyncPlan(
            video_path="test_video.mov",
            audio_path="test_audio.wav",
            segments=segments,
            total_original_duration=10.0,
            total_sync_duration=13.0
        )
        
        assert len(sync_plan.segments) == 2
        assert sync_plan.total_original_duration == 10.0
        assert sync_plan.total_sync_duration == 13.0
    
    def test_sync_plan_serialization(self):
        """Test sync plan to/from dict conversion."""
        segments = [
            SlideSegment(1, 0.0, 5.0, 5.0, 7.0)
        ]
        
        sync_plan = SyncPlan(
            video_path="test.mov",
            audio_path="test.wav",
            segments=segments,
            total_original_duration=5.0,
            total_sync_duration=7.0
        )
        
        # Test serialization
        plan_dict = sync_plan.to_dict()
        assert isinstance(plan_dict, dict)
        assert plan_dict['video_path'] == "test.mov"
        assert len(plan_dict['segments']) == 1
        
        # Test deserialization
        restored_plan = SyncPlan.from_dict(plan_dict)
        assert restored_plan.video_path == sync_plan.video_path
        assert len(restored_plan.segments) == 1
        assert restored_plan.segments[0].slide_number == 1


class TestSlideSynchronizer:
    """Test SlideSynchronizer functionality."""
    
    @pytest.fixture
    def synchronizer(self):
        """Create a SlideSynchronizer instance."""
        return SlideSynchronizer(keynote_delay=1.0)
    
    @pytest.fixture
    def mock_manifests(self):
        """Create mock manifest data."""
        video_manifest = {
            'video_analysis': {
                'video_info': {'duration': 10.0, 'frame_rate': 30.0},
                'scene_transitions': [
                    {'timestamp': 0.0, 'slide_number': 1},
                    {'timestamp': 5.0, 'slide_number': 2},
                    {'timestamp': 10.0, 'slide_number': 3}
                ]
            }
        }
        
        audio_manifest = {
            'chunks': [
                {'start_time': 0.0, 'end_time': 3.0, 'duration': 3.0, 'text': 'First slide content'},
                {'start_time': 3.0, 'end_time': 8.0, 'duration': 5.0, 'text': 'Second slide content [transition]'},
                {'start_time': 8.0, 'end_time': 12.0, 'duration': 4.0, 'text': 'Third slide content'}
            ]
        }
        
        transcript_manifest = {
            'segments': [
                {'slide_number': 1, 'text': 'First slide content'},
                {'slide_number': 2, 'text': 'Second slide content [transition]'},
                {'slide_number': 3, 'text': 'Third slide content'}
            ]
        }
        
        return video_manifest, audio_manifest, transcript_manifest
    
    def test_extract_transition_points(self, synchronizer):
        """Test transition point extraction."""
        video_data = {
            'scene_transitions': [
                {'timestamp': 1.0},
                {'timestamp': 6.0},
                {'timestamp': 11.0}
            ]
        }
        
        transitions = synchronizer._extract_transition_points(video_data)
        
        # Should compensate for 1.0s Keynote delay
        assert len(transitions) == 4  # Including 0.0 start
        assert transitions[0] == 0.0
        assert transitions[1] == 0.0  # 1.0 - 1.0
        assert transitions[2] == 5.0  # 6.0 - 1.0
        assert transitions[3] == 10.0  # 11.0 - 1.0
    
    def test_extract_narration_segments(self, synchronizer):
        """Test narration segment extraction."""
        audio_data = {
            'chunks': [
                {'start_time': 0.0, 'end_time': 3.0, 'duration': 3.0, 'text': 'First slide'},
                {'start_time': 3.0, 'end_time': 7.0, 'duration': 4.0, 'text': 'Second slide [transition]'}
            ]
        }
        
        transcript_data = {
            'segments': [
                {'slide_number': 1},
                {'slide_number': 2}
            ]
        }
        
        segments = synchronizer._extract_narration_segments(audio_data, transcript_data)
        
        assert len(segments) == 2
        assert segments[0]['duration'] == 3.0
        assert segments[0]['slide_number'] == 1
        assert segments[1]['duration'] == 4.0
        assert segments[1]['has_transition_cue'] == True
    
    def test_create_slide_segments(self, synchronizer):
        """Test slide segment creation."""
        transition_points = [0.0, 5.0, 10.0]
        narration_segments = [
            {'segment_number': 1, 'duration': 3.0, 'slide_number': 1, 'text': 'First slide'},
            {'segment_number': 2, 'duration': 7.0, 'slide_number': 2, 'text': 'Second slide'}
        ]
        
        segments = synchronizer._create_slide_segments(transition_points, narration_segments)
        
        assert len(segments) == 2
        assert segments[0].slide_number == 1
        assert segments[0].keynote_start == 0.0
        assert segments[0].keynote_end == 5.0
        assert segments[0].keynote_duration == 5.0
        assert segments[0].narration_duration == 3.0
        assert segments[0].gap_needed == 0.0  # No gap needed
        
        assert segments[1].slide_number == 2
        assert segments[1].keynote_start == 5.0
        assert segments[1].keynote_end == 10.0
        assert segments[1].narration_duration == 7.0
        assert segments[1].gap_needed == 2.0  # 7.0 - 5.0
    
    def test_validate_sync_plan(self, synchronizer):
        """Test sync plan validation."""
        segments = [
            SlideSegment(1, 0.0, 5.0, 5.0, 3.0),  # No gap
            SlideSegment(2, 5.0, 10.0, 5.0, 12.0)  # Large gap
        ]
        
        sync_plan = SyncPlan(
            video_path="test.mov",
            audio_path="test.wav", 
            segments=segments,
            total_original_duration=10.0,
            total_sync_duration=15.0
        )
        
        validation = synchronizer.validate_sync_plan(sync_plan)
        
        assert isinstance(validation, dict)
        assert 'valid' in validation
        assert 'metrics' in validation
        assert 'warnings' in validation
        assert validation['metrics']['total_segments'] == 2
        assert validation['metrics']['total_gaps'] == 1
        assert validation['metrics']['max_gap'] == 7.0
    
    @patch('builtins.open')
    @patch('json.load')
    def test_create_sync_plan_integration(self, mock_json_load, mock_open, synchronizer, mock_manifests):
        """Test complete sync plan creation."""
        video_manifest, audio_manifest, transcript_manifest = mock_manifests
        
        # Mock file loading
        mock_json_load.side_effect = [video_manifest, audio_manifest, transcript_manifest]
        
        sync_plan = synchronizer.create_sync_plan(
            video_analysis_manifest="video.json",
            audio_splice_manifest="audio.json",
            transcript_manifest="transcript.json"
        )
        
        assert isinstance(sync_plan, SyncPlan)
        assert len(sync_plan.segments) > 0
        assert sync_plan.total_original_duration > 0


class TestGapManager:
    """Test GapManager functionality."""
    
    @pytest.fixture
    def gap_manager(self):
        """Create a GapManager instance."""
        return GapManager()
    
    def test_gap_filler_creation(self):
        """Test GapFiller creation and validation."""
        gap_filler = GapFiller(
            duration=2.5,
            gap_type="static_hold"
        )
        
        assert gap_filler.duration == 2.5
        assert gap_filler.gap_type == "static_hold"
        assert gap_filler.source_frame is None
    
    def test_gap_filler_invalid_duration(self):
        """Test GapFiller with invalid duration."""
        with pytest.raises(ValueError):
            GapFiller(duration=-1.0, gap_type="static_hold")
    
    def test_determine_gap_type(self, gap_manager):
        """Test gap type determination based on content."""
        # Test animated content
        text = "This chart shows the animation of data moving"
        gap_type = gap_manager._determine_gap_type(text)
        assert gap_type == "animated_hold"
        
        # Test conclusion content
        text = "In conclusion, thank you for your attention"
        gap_type = gap_manager._determine_gap_type(text)
        assert gap_type == "fade_hold"
        
        # Test default content
        text = "This is a normal slide with regular content"
        gap_type = gap_manager._determine_gap_type(text)
        assert gap_type == "static_hold"
    
    @patch('ffmpeg.run')
    @patch('ffmpeg.input')
    @patch('ffmpeg.output')
    def test_create_gap_filler_static(self, mock_output, mock_input, mock_run, gap_manager):
        """Test static gap filler creation."""
        segment = SlideSegment(
            slide_number=1,
            keynote_start=0.0,
            keynote_end=5.0,
            keynote_duration=5.0,
            narration_duration=7.0
        )
        segment.gap_type = "static_hold"
        
        # Mock ffmpeg operations
        mock_input_obj = Mock()
        mock_output_obj = Mock()
        mock_input.return_value = mock_input_obj
        mock_output.return_value = mock_output_obj
        
        gap_filler = gap_manager.create_gap_filler(segment, "test_video.mov")
        
        assert gap_filler is not None
        assert gap_filler.gap_type == "static_hold"
        assert gap_filler.duration == 2.0


class TestOptimizeGapTypes:
    """Test gap type optimization functionality."""
    
    def test_optimize_gap_types(self):
        """Test gap type optimization based on content."""
        segments = [
            SlideSegment(1, 0.0, 5.0, 5.0, 7.0),
            SlideSegment(2, 5.0, 10.0, 5.0, 12.0)
        ]
        
        sync_plan = SyncPlan(
            video_path="test.mov",
            audio_path="test.wav",
            segments=segments,
            total_original_duration=10.0,
            total_sync_duration=19.0
        )
        
        transcript_data = {
            'segments': [
                {'text': 'This chart shows animated data'},
                {'text': 'In summary, thank you for listening'}
            ]
        }
        
        optimized_plan = optimize_gap_types(sync_plan, transcript_data)
        
        assert optimized_plan.segments[0].gap_type == "animated_hold"
        assert optimized_plan.segments[1].gap_type == "fade_hold"


class TestVideoAssembler:
    """Test VideoAssembler functionality."""
    
    @pytest.fixture
    def assembler(self):
        """Create a VideoAssembler instance."""
        return VideoAssembler()
    
    def test_video_segment_creation(self):
        """Test VideoSegment creation."""
        segment = VideoSegment(
            slide_number=1,
            segment_path="/path/to/segment1.mp4",
            start_time=0.0,
            duration=5.0,
            has_gap=True,
            gap_duration=2.0
        )
        
        assert segment.slide_number == 1
        assert segment.segment_path == "/path/to/segment1.mp4"
        assert segment.has_gap == True
        assert segment.gap_duration == 2.0
    
    @patch('ffmpeg.run')
    @patch('ffmpeg.input')  
    @patch('ffmpeg.output')
    def test_extract_video_segments(self, mock_output, mock_input, mock_run, assembler):
        """Test video segment extraction."""
        segments = [
            SlideSegment(1, 0.0, 5.0, 5.0, 5.0),
            SlideSegment(2, 5.0, 10.0, 5.0, 7.0)
        ]
        
        sync_plan = SyncPlan(
            video_path="test_video.mov",
            audio_path="test_audio.wav",
            segments=segments,
            total_original_duration=10.0,
            total_sync_duration=12.0
        )
        
        # Mock ffmpeg operations
        mock_input_obj = Mock()
        mock_output_obj = Mock()
        mock_input.return_value = mock_input_obj
        mock_output.return_value = mock_output_obj
        
        extracted_segments = assembler.extract_video_segments(sync_plan)
        
        assert len(extracted_segments) == 2
        assert extracted_segments[0].slide_number == 1
        assert extracted_segments[1].slide_number == 2
        assert extracted_segments[1].has_gap == True


# Integration test
class TestSlideSync Integration:
    """Integration tests for the complete slide sync system."""
    
    @patch('core.slide_sync.SlideSynchronizer._load_manifest')
    @patch('core.video_assembly.assemble_synchronized_video')
    def test_full_sync_workflow(self, mock_assemble, mock_load):
        """Test complete synchronization workflow."""
        # Mock manifest data
        mock_load.side_effect = [
            {'video_analysis': {'video_info': {'duration': 10.0}, 'scene_transitions': []}},
            {'chunks': [{'duration': 5.0, 'text': 'test'}]},
            {'segments': [{'slide_number': 1}]}
        ]
        
        mock_assemble.return_value = "output_video.mp4"
        
        synchronizer = SlideSynchronizer()
        
        sync_plan = synchronizer.create_sync_plan(
            video_analysis_manifest="video.json",
            audio_splice_manifest="audio.json", 
            transcript_manifest="transcript.json"
        )
        
        assert isinstance(sync_plan, SyncPlan)
        
        # Test assembly
        result = mock_assemble(
            sync_plan=sync_plan,
            audio_path="test_audio.wav",
            output_path="test_output.mp4"
        )
        
        assert result == "output_video.mp4"


if __name__ == '__main__':
    pytest.main([__file__])