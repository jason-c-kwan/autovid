"""
Tests for video synchronization functionality.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.video_sync import (
    VideoSyncError,
    SyncPoint,
    TimingCorrection,
    load_video_analysis_manifest,
    load_audio_splice_manifest,
    extract_scene_timings,
    extract_audio_chunk_timings,
    calculate_sync_points,
    generate_timing_corrections,
    compensate_keynote_delay,
    validate_scene_audio_mapping,
    get_video_sync_config,
    create_sync_manifest
)

from core.sync_engine import (
    SyncEngineError,
    probe_media_streams,
    validate_sync_inputs,
    synchronize_video_audio
)

from core.sync_validator import (
    SyncValidationError,
    SyncMetrics,
    DriftAnalysis,
    calculate_sync_metrics,
    analyze_timing_drift,
    validate_sync_accuracy,
    check_timing_drift
)


class TestVideoSync:
    """Test cases for video synchronization coordination."""
    
    def test_sync_point_creation(self):
        """Test SyncPoint dataclass creation."""
        sync_point = SyncPoint(
            scene_index=0,
            video_timestamp=5.0,
            audio_timestamp=4.8,
            sync_offset=0.2,
            confidence=0.95
        )
        
        assert sync_point.scene_index == 0
        assert sync_point.video_timestamp == 5.0
        assert sync_point.audio_timestamp == 4.8
        assert sync_point.sync_offset == 0.2
        assert sync_point.confidence == 0.95
    
    def test_timing_correction_creation(self):
        """Test TimingCorrection dataclass creation."""
        correction = TimingCorrection(
            total_drift=0.15,
            max_offset=0.08,
            correction_points=[],
            keynote_delay_applied=1.0,
            crossfade_compensations=[]
        )
        
        assert correction.total_drift == 0.15
        assert correction.max_offset == 0.08
        assert correction.keynote_delay_applied == 1.0
    
    def test_load_video_analysis_manifest_success(self):
        """Test successful loading of video analysis manifest."""
        manifest_data = {
            'video_analysis': {
                'scene_transitions': [
                    {'timestamp': 0.0, 'slide_number': 1},
                    {'timestamp': 10.0, 'slide_number': 2}
                ],
                'video_info': {'duration': 30.0, 'width': 1920, 'height': 1080},
                'keynote_delay_compensation': 1.0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(manifest_data, f)
            manifest_path = f.name
        
        try:
            result = load_video_analysis_manifest(manifest_path)
            assert 'video_analysis' in result
            assert len(result['video_analysis']['scene_transitions']) == 2
            assert result['video_analysis']['keynote_delay_compensation'] == 1.0
        finally:
            os.unlink(manifest_path)
    
    def test_load_video_analysis_manifest_missing_file(self):
        """Test loading non-existent video analysis manifest."""
        with pytest.raises(VideoSyncError, match="Video analysis manifest not found"):
            load_video_analysis_manifest("/nonexistent/path.json")
    
    def test_load_video_analysis_manifest_invalid_json(self):
        """Test loading invalid JSON video analysis manifest."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            manifest_path = f.name
        
        try:
            with pytest.raises(VideoSyncError, match="Failed to parse video analysis manifest"):
                load_video_analysis_manifest(manifest_path)
        finally:
            os.unlink(manifest_path)
    
    def test_extract_scene_timings(self):
        """Test extraction of scene timing data."""
        video_manifest = {
            'video_analysis': {
                'scene_transitions': [
                    {'timestamp': 0.0, 'slide_number': 1, 'confidence': 0.9},
                    {'timestamp': 15.5, 'slide_number': 2, 'confidence': 0.8},
                    {'timestamp': 28.2, 'slide_number': 3, 'confidence': 0.95}
                ],
                'keynote_delay_compensation': 1.0
            }
        }
        
        scenes = extract_scene_timings(video_manifest)
        assert len(scenes) == 3
        assert scenes[0]['timestamp'] == 0.0
        assert scenes[1]['timestamp'] == 15.5
        assert scenes[2]['timestamp'] == 28.2
    
    def test_extract_audio_chunk_timings(self):
        """Test extraction of audio chunk timing data."""
        audio_manifest = {
            'audio_timing': {
                'chunks': [
                    {'id': 'chunk_0', 'start_time': 0.0, 'duration': 3.5},
                    {'id': 'chunk_1', 'start_time': 3.4, 'duration': 4.2},
                    {'id': 'chunk_2', 'start_time': 7.5, 'duration': 2.8}
                ]
            }
        }
        
        chunks = extract_audio_chunk_timings(audio_manifest)
        assert len(chunks) == 3
        assert chunks[0]['start_time'] == 0.0
        assert chunks[1]['duration'] == 4.2
    
    def test_calculate_sync_points(self):
        """Test calculation of synchronization points."""
        scene_transitions = [
            {'timestamp': 1.0, 'slide_number': 1},  # Already delay-compensated
            {'timestamp': 11.0, 'slide_number': 2},
            {'timestamp': 21.0, 'slide_number': 3}
        ]
        
        audio_chunks = [
            {'start_time': 0.0, 'duration': 3.0},
            {'start_time': 2.9, 'duration': 3.5},  # With crossfade overlap
            {'start_time': 6.3, 'duration': 2.8}
        ]
        
        sync_points = calculate_sync_points(scene_transitions, audio_chunks, crossfade_duration=0.1)
        
        assert len(sync_points) == 3
        assert sync_points[0].scene_index == 0
        assert sync_points[0].video_timestamp == 1.0
        assert all(isinstance(point, SyncPoint) for point in sync_points)
    
    def test_generate_timing_corrections(self):
        """Test generation of timing corrections."""
        sync_points = [
            SyncPoint(0, 1.0, 0.8, 0.2, 0.9),  # 200ms offset
            SyncPoint(1, 11.0, 10.7, 0.3, 0.8),  # 300ms offset
            SyncPoint(2, 21.0, 20.9, 0.1, 0.95)  # 100ms offset
        ]
        
        corrections = generate_timing_corrections(sync_points, video_duration=30.0, tolerance=0.15)
        
        assert isinstance(corrections, TimingCorrection)
        assert corrections.total_drift > 0
        assert len(corrections.correction_points) >= 1  # Points above tolerance
    
    def test_compensate_keynote_delay(self):
        """Test Keynote delay compensation."""
        sync_points = [
            SyncPoint(0, 2.0, 1.0, 1.0, 0.9),
            SyncPoint(1, 12.0, 11.0, 1.0, 0.8)
        ]
        
        compensated = compensate_keynote_delay(sync_points, additional_delay=0.5)
        
        assert len(compensated) == 2
        assert compensated[0].video_timestamp == 1.5  # 2.0 - 0.5
        assert compensated[1].video_timestamp == 11.5  # 12.0 - 0.5
        assert all(point.confidence < 0.9 for point in compensated)  # Reduced confidence
    
    def test_validate_scene_audio_mapping(self):
        """Test validation of scene-audio mapping."""
        video_manifest = {
            'video_analysis': {
                'scene_transitions': [
                    {'timestamp': 1.0, 'slide_number': 1},
                    {'timestamp': 11.0, 'slide_number': 2}
                ]
            }
        }
        
        audio_manifest = {
            'chunks': [
                {'start_time': 0.0, 'duration': 3.0},
                {'start_time': 2.9, 'duration': 3.5}
            ]
        }
        
        sync_points = [
            SyncPoint(0, 1.0, 0.0, 1.0, 0.9),
            SyncPoint(1, 11.0, 2.9, 8.1, 0.8)
        ]
        
        result = validate_scene_audio_mapping(video_manifest, audio_manifest, sync_points)
        
        assert 'status' in result
        assert 'statistics' in result
        assert result['statistics']['total_sync_points'] == 2
    
    def test_get_video_sync_config_defaults(self):
        """Test getting video sync config with defaults."""
        with patch('pathlib.Path.exists', return_value=False):
            config = get_video_sync_config("nonexistent.yaml")
            
            assert config['keynote_delay'] == 1.0
            assert config['sync_tolerance'] == 0.1
            assert config['output_format'] == 'mp4'
    
    def test_create_sync_manifest(self):
        """Test creation of synchronization manifest."""
        sync_points = [SyncPoint(0, 1.0, 0.8, 0.2, 0.9)]
        corrections = TimingCorrection(0.2, 0.2, sync_points, 1.0, [])
        validation = {'status': 'PASS', 'warnings': [], 'errors': []}
        
        manifest = create_sync_manifest(
            "video.json", "audio.json", sync_points, corrections, 
            validation, "output.mp4"
        )
        
        assert 'video_sync' in manifest
        assert 'sync_points' in manifest['video_sync']
        assert 'timing_corrections' in manifest['video_sync']
        assert 'validation' in manifest['video_sync']
        assert len(manifest['video_sync']['sync_points']) == 1


class TestSyncEngine:
    """Test cases for sync engine functionality."""
    
    @patch('ffmpeg.probe')
    def test_probe_media_streams_success(self, mock_probe):
        """Test successful media stream probing."""
        mock_probe.return_value = {
            'streams': [
                {
                    'index': 0,
                    'codec_type': 'video',
                    'codec_name': 'h264',
                    'width': 1920,
                    'height': 1080,
                    'r_frame_rate': '30/1',
                    'duration': '30.0'
                },
                {
                    'index': 1,
                    'codec_type': 'audio',
                    'codec_name': 'aac',
                    'sample_rate': 48000,
                    'channels': 2,
                    'duration': '30.0'
                }
            ],
            'format': {'duration': '30.0'}
        }
        
        streams = probe_media_streams("test_video.mp4")
        
        assert len(streams['video']) == 1
        assert len(streams['audio']) == 1
        assert streams['video'][0]['width'] == 1920
        assert streams['audio'][0]['channels'] == 2
        assert streams['duration'] == 30.0
    
    @patch('ffmpeg.probe')
    def test_probe_media_streams_error(self, mock_probe):
        """Test media stream probing error handling."""
        mock_error = Mock()
        mock_error.stderr.decode.return_value = "File not found"
        mock_probe.side_effect = Exception("FFmpeg error")
        
        with pytest.raises(SyncEngineError, match="Unexpected error probing media"):
            probe_media_streams("nonexistent.mp4")
    
    @patch('core.sync_engine.probe_media_streams')
    @patch('pathlib.Path.exists')
    def test_validate_sync_inputs_success(self, mock_exists, mock_probe):
        """Test successful sync input validation."""
        mock_exists.return_value = True
        mock_probe.side_effect = [
            {'video': [{'codec': 'h264'}], 'audio': [], 'duration': 30.0},
            {'video': [], 'audio': [{'codec': 'aac'}], 'duration': 30.0}
        ]
        
        video_streams, audio_streams = validate_sync_inputs("video.mp4", "audio.wav")
        
        assert len(video_streams['video']) == 1
        assert len(audio_streams['audio']) == 1
    
    @patch('pathlib.Path.exists')
    def test_validate_sync_inputs_missing_file(self, mock_exists):
        """Test sync input validation with missing file."""
        mock_exists.side_effect = [False, True]  # Video missing, audio present
        
        with pytest.raises(SyncEngineError, match="Video file not found"):
            validate_sync_inputs("missing_video.mp4", "audio.wav")
    
    @patch('core.sync_engine.validate_sync_inputs')
    @patch('core.sync_engine.generate_final_video')
    @patch('ffmpeg.input')
    def test_synchronize_video_audio_basic(self, mock_input, mock_generate, mock_validate):
        """Test basic video-audio synchronization."""
        # Mock validation
        mock_validate.return_value = (
            {'video': [{'codec': 'h264'}], 'duration': 30.0},
            {'audio': [{'codec': 'aac'}], 'duration': 30.0}
        )
        
        # Mock FFmpeg inputs
        mock_video_input = Mock()
        mock_audio_input = Mock()
        mock_input.side_effect = [mock_video_input, mock_audio_input]
        
        # Mock video generation
        mock_generate.return_value = "output.mp4"
        
        result = synchronize_video_audio("video.mp4", "audio.wav", "output.mp4")
        
        assert result == "output.mp4"
        mock_validate.assert_called_once()
        mock_generate.assert_called_once()


class TestSyncValidator:
    """Test cases for sync validation functionality."""
    
    def test_sync_metrics_creation(self):
        """Test SyncMetrics dataclass creation."""
        metrics = SyncMetrics(
            avg_offset=0.05,
            max_offset=0.12,
            std_deviation=0.03,
            sync_accuracy_score=85.0,
            drift_rate=0.02,
            timing_consistency=90.0,
            confidence_score=88.0
        )
        
        assert metrics.avg_offset == 0.05
        assert metrics.sync_accuracy_score == 85.0
    
    def test_drift_analysis_creation(self):
        """Test DriftAnalysis dataclass creation."""
        analysis = DriftAnalysis(
            total_drift=0.15,
            drift_rate_per_minute=0.03,
            drift_direction='audio_lagging',
            significant_drift_points=[],
            drift_consistency=80.0,
            requires_correction=True
        )
        
        assert analysis.total_drift == 0.15
        assert analysis.drift_direction == 'audio_lagging'
        assert analysis.requires_correction is True
    
    def test_calculate_sync_metrics(self):
        """Test calculation of sync metrics."""
        sync_points = [
            SyncPoint(0, 1.0, 0.95, 0.05, 0.9),
            SyncPoint(1, 11.0, 10.8, 0.2, 0.8),
            SyncPoint(2, 21.0, 20.9, 0.1, 0.95)
        ]
        
        metrics = calculate_sync_metrics(sync_points)
        
        assert isinstance(metrics, SyncMetrics)
        assert metrics.avg_offset > 0
        assert 0 <= metrics.sync_accuracy_score <= 100
        assert 0 <= metrics.confidence_score <= 100
    
    def test_analyze_timing_drift(self):
        """Test timing drift analysis."""
        sync_points = [
            SyncPoint(0, 1.0, 1.0, 0.0, 0.9),  # No drift initially
            SyncPoint(1, 11.0, 10.8, 0.2, 0.8),  # Audio lagging
            SyncPoint(2, 21.0, 20.6, 0.4, 0.85)  # More drift
        ]
        
        analysis = analyze_timing_drift(sync_points, video_duration=30.0)
        
        assert isinstance(analysis, DriftAnalysis)
        assert analysis.total_drift == 0.4  # From 0.0 to 0.4
        assert analysis.drift_direction == 'audio_lagging'
    
    def test_check_timing_drift_no_drift(self):
        """Test timing drift check with no drift."""
        sync_points = [
            SyncPoint(0, 1.0, 1.0, 0.0, 0.9),
            SyncPoint(1, 11.0, 11.0, 0.0, 0.9)
        ]
        
        result = check_timing_drift(sync_points, tolerance=0.05)
        
        assert result['has_drift'] is False
        assert result['drift_direction'] == 'none'
        assert result['requires_attention'] is False
    
    def test_check_timing_drift_with_drift(self):
        """Test timing drift check with significant drift."""
        sync_points = [
            SyncPoint(0, 1.0, 1.0, 0.0, 0.9),
            SyncPoint(1, 11.0, 10.8, 0.2, 0.8)  # 200ms drift
        ]
        
        result = check_timing_drift(sync_points, tolerance=0.05)
        
        assert result['has_drift'] is True
        assert result['drift_direction'] == 'audio_lagging'
        assert result['requires_attention'] is True
    
    @patch('core.sync_validator.ffmpeg.probe')
    def test_validate_sync_accuracy(self, mock_probe):
        """Test comprehensive sync accuracy validation."""
        mock_probe.return_value = {
            'streams': [
                {'codec_type': 'video', 'duration': '30.0'}
            ]
        }
        
        sync_points = [
            SyncPoint(0, 1.0, 0.95, 0.05, 0.9),
            SyncPoint(1, 11.0, 10.85, 0.15, 0.85),
            SyncPoint(2, 21.0, 20.9, 0.1, 0.95)
        ]
        
        report = validate_sync_accuracy("video.mp4", sync_points, video_duration=30.0)
        
        assert hasattr(report, 'overall_grade')
        assert hasattr(report, 'sync_metrics')
        assert hasattr(report, 'drift_analysis')
        assert report.overall_grade in ['A', 'B', 'C', 'D']


class TestIntegration:
    """Integration tests for video sync functionality."""
    
    @patch('subprocess.run')
    def test_sync_video_wrapper_integration(self, mock_run):
        """Test integration of sync_video wrapper function."""
        from core.wrappers import sync_video
        
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.stdout = '{"status": "success"}'
        mock_result.stderr = ''
        mock_run.return_value = mock_result
        
        # Create temporary manifest file content
        manifest_content = {
            'video_sync': {
                'output_video': '/path/to/output.mp4',
                'sync_points': [],
                'validation': {'status': 'PASS'}
            }
        }
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', Mock()) as mock_open, \
             patch('json.load', return_value=manifest_content):
            
            result = sync_video(
                video_path="test_video.mp4",
                audio_path="test_audio.wav",
                output_path="output.mp4"
            )
            
            assert 'video_sync' in result
            mock_run.assert_called_once()
    
    def test_end_to_end_sync_workflow(self):
        """Test complete synchronization workflow with mocked data."""
        # This would test the entire workflow from manifest loading
        # through synchronization to validation in a realistic scenario
        
        # Mock video analysis manifest
        video_manifest = {
            'video_analysis': {
                'scene_transitions': [
                    {'timestamp': 0.0, 'slide_number': 1, 'confidence': 0.9},
                    {'timestamp': 10.0, 'slide_number': 2, 'confidence': 0.85}
                ],
                'video_info': {'duration': 20.0},
                'keynote_delay_compensation': 1.0
            }
        }
        
        # Mock audio splice manifest
        audio_manifest = {
            'audio_timing': {
                'chunks': [
                    {'start_time': 0.0, 'duration': 3.0},
                    {'start_time': 2.9, 'duration': 4.0}
                ]
            }
        }
        
        # Test the workflow
        scene_timings = extract_scene_timings(video_manifest)
        audio_timings = extract_audio_chunk_timings(audio_manifest)
        sync_points = calculate_sync_points(scene_timings, audio_timings)
        corrections = generate_timing_corrections(sync_points, 20.0)
        validation = validate_scene_audio_mapping(video_manifest, audio_manifest, sync_points)
        
        # Verify workflow completed successfully
        assert len(sync_points) == 2
        assert isinstance(corrections, TimingCorrection)
        assert validation['status'] in ['PASS', 'WARN', 'FAIL']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])