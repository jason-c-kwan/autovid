"""
Test suite for video analysis functionality.

This module tests the video analysis components including scene detection,
movement analysis, and integration with the AutoVid pipeline.
"""

import unittest
import tempfile
import os
import json
import shutil
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.video_analysis import (
    VideoAnalysisError,
    probe_video_info,
    detect_scene_changes,
    extract_movement_frames,
    compensate_keynote_delay,
    validate_transition_count,
    generate_timing_manifest,
    analyze_video
)


class TestVideoAnalysis(unittest.TestCase):
    """Test core video analysis functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = os.path.join(self.temp_dir, "test_video.mov")
        self.test_transcript_path = os.path.join(self.temp_dir, "test_transcript.json")
        
        # Create mock video file
        with open(self.test_video_path, 'w') as f:
            f.write("mock video content")
        
        # Create mock transcript
        self.mock_transcript = {
            "slides": [
                {
                    "slide_number": 1,
                    "segments": [
                        {"kind": "text", "text": "Hello world"},
                        {"kind": "cue", "cue": "[transition]"},
                        {"kind": "text", "text": "Next part"}
                    ]
                },
                {
                    "slide_number": 2,
                    "segments": [
                        {"kind": "text", "text": "Second slide"},
                        {"kind": "cue", "cue": "[transition]"},
                        {"kind": "text", "text": "Animation"}
                    ]
                }
            ]
        }
        
        with open(self.test_transcript_path, 'w') as f:
            json.dump(self.mock_transcript, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('core.video_analysis.ffmpeg')
    def test_probe_video_info_success(self, mock_ffmpeg):
        """Test successful video probing."""
        # Mock ffmpeg probe response
        mock_probe_result = {
            'streams': [
                {
                    'codec_type': 'video',
                    'width': 1920,
                    'height': 1080,
                    'duration': '60.5',
                    'r_frame_rate': '30/1',
                    'nb_frames': '1815',
                    'codec_name': 'h264',
                    'pix_fmt': 'yuv420p',
                    'bit_rate': '5000000'
                }
            ]
        }
        mock_ffmpeg.probe.return_value = mock_probe_result
        
        result = probe_video_info(self.test_video_path)
        
        self.assertEqual(result['width'], 1920)
        self.assertEqual(result['height'], 1080)
        self.assertEqual(result['duration'], 60.5)
        self.assertEqual(result['frame_rate'], 30.0)
        self.assertEqual(result['frame_count'], 1815)
        self.assertEqual(result['codec'], 'h264')
        
        mock_ffmpeg.probe.assert_called_once_with(self.test_video_path)
    
    @patch('core.video_analysis.ffmpeg')
    def test_probe_video_info_no_video_stream(self, mock_ffmpeg):
        """Test probing with no video stream."""
        mock_probe_result = {
            'streams': [
                {'codec_type': 'audio'}
            ]
        }
        mock_ffmpeg.probe.return_value = mock_probe_result
        
        with self.assertRaises(VideoAnalysisError) as context:
            probe_video_info(self.test_video_path)
        
        self.assertIn("No video stream found", str(context.exception))
    
    @patch('core.video_analysis.ffmpeg')
    def test_probe_video_info_ffmpeg_error(self, mock_ffmpeg):
        """Test probing with FFmpeg error."""
        mock_error = MagicMock()
        mock_error.stderr.decode.return_value = "FFmpeg error message"
        mock_ffmpeg.probe.side_effect = mock_ffmpeg.Error(mock_error)
        mock_ffmpeg.Error = Exception  # Make Error a regular exception for testing
        
        with self.assertRaises(VideoAnalysisError) as context:
            probe_video_info(self.test_video_path)
        
        self.assertIn("Failed to probe video", str(context.exception))
    
    @patch('core.video_analysis.probe_video_info')
    @patch('core.video_analysis.ffmpeg')
    @patch('core.video_analysis.tempfile')
    def test_detect_scene_changes_success(self, mock_tempfile, mock_ffmpeg, mock_probe):
        """Test successful scene detection."""
        # Mock video info
        mock_probe.return_value = {
            'frame_rate': 30.0,
            'duration': 60.0
        }
        
        # Mock tempfile
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/scene_detect.txt"
        mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = mock_temp_file
        
        # Mock FFmpeg scene detection output
        mock_stderr = """
[Parsed_showinfo_1 @ 0x123] n:45 pts:1500 pts_time:1.5 pos:123456 fmt:yuv420p
[Parsed_showinfo_1 @ 0x456] n:90 pts:3000 pts_time:3.0 pos:234567 fmt:yuv420p
        """
        
        mock_error = MagicMock()
        mock_error.stderr.decode.return_value = mock_stderr
        mock_ffmpeg.run.side_effect = mock_ffmpeg.Error(mock_error)
        mock_ffmpeg.Error = Exception
        
        result = detect_scene_changes(self.test_video_path, threshold=0.4)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['timestamp'], 1.5)
        self.assertEqual(result[0]['frame_number'], 45)
        self.assertEqual(result[0]['slide_number'], 1)
        self.assertEqual(result[1]['timestamp'], 3.0)
        self.assertEqual(result[1]['frame_number'], 90)
        self.assertEqual(result[1]['slide_number'], 2)
    
    def test_compensate_keynote_delay(self):
        """Test Keynote delay compensation."""
        transitions = [
            {'timestamp': 2.5, 'frame_number': 75},
            {'timestamp': 5.0, 'frame_number': 150},
            {'timestamp': 0.5, 'frame_number': 15}  # Test edge case
        ]
        
        result = compensate_keynote_delay(transitions, delay_seconds=1.0)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['timestamp'], 1.5)
        self.assertEqual(result[0]['original_timestamp'], 2.5)
        self.assertEqual(result[0]['delay_compensation'], 1.0)
        
        self.assertEqual(result[1]['timestamp'], 4.0)
        self.assertEqual(result[1]['original_timestamp'], 5.0)
        
        # Test edge case - timestamp shouldn't go below 0
        self.assertEqual(result[2]['timestamp'], 0.0)
        self.assertEqual(result[2]['original_timestamp'], 0.5)
    
    def test_validate_transition_count_exact_match(self):
        """Test transition validation with exact match."""
        detected = [{'timestamp': 1.0}, {'timestamp': 2.0}]
        expected = ['[transition]', '[transition]']
        
        result = validate_transition_count(detected, expected)
        
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['detected_count'], 2)
        self.assertEqual(result['expected_count'], 2)
        self.assertEqual(result['difference'], 0)
        self.assertTrue(result['within_tolerance'])
        self.assertIn("matches exactly", result['message'])
    
    def test_validate_transition_count_within_tolerance(self):
        """Test transition validation within tolerance."""
        detected = [{'timestamp': 1.0}, {'timestamp': 2.0}]
        expected = ['[transition]']
        
        result = validate_transition_count(detected, expected, tolerance=1)
        
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['difference'], 1)
        self.assertTrue(result['within_tolerance'])
        self.assertIn("within tolerance", result['message'])
    
    def test_validate_transition_count_exceeds_tolerance(self):
        """Test transition validation exceeding tolerance."""
        detected = [{'timestamp': 1.0}, {'timestamp': 2.0}, {'timestamp': 3.0}]
        expected = ['[transition]']
        
        result = validate_transition_count(detected, expected, tolerance=1)
        
        self.assertEqual(result['status'], 'WARN')
        self.assertEqual(result['difference'], 2)
        self.assertFalse(result['within_tolerance'])
        self.assertIn("mismatch", result['message'])
    
    def test_generate_timing_manifest(self):
        """Test timing manifest generation."""
        video_path = self.test_video_path
        scene_transitions = [
            {'timestamp': 1.5, 'frame_number': 45, 'slide_number': 1},
            {'timestamp': 3.0, 'frame_number': 90, 'slide_number': 2}
        ]
        movement_ranges = [
            {
                'start_frame': 50,
                'end_frame': 75,
                'start_time': 1.67,
                'end_time': 2.5,
                'within_slide': 1
            }
        ]
        video_info = {'width': 1920, 'height': 1080, 'duration': 60.0}
        validation_result = {'status': 'PASS', 'message': 'All good'}
        
        result = generate_timing_manifest(
            video_path, scene_transitions, movement_ranges, 
            video_info, validation_result, keynote_delay=1.0
        )
        
        self.assertIn('video_analysis', result)
        analysis = result['video_analysis']
        
        self.assertEqual(analysis['input_file'], str(Path(video_path).resolve()))
        self.assertEqual(analysis['total_scenes'], 2)
        self.assertEqual(analysis['total_movements'], 1)
        self.assertEqual(analysis['keynote_delay_compensation'], 1.0)
        self.assertEqual(analysis['scene_transitions'], scene_transitions)
        self.assertEqual(analysis['movement_ranges'], movement_ranges)
        self.assertEqual(analysis['validation'], validation_result)
        self.assertIn('analysis_timestamp', analysis)
        self.assertIn('processing_settings', analysis)


class TestVideoAnalysisCLI(unittest.TestCase):
    """Test CLI tool functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = os.path.join(self.temp_dir, "test_video.mov")
        self.test_output_path = os.path.join(self.temp_dir, "analysis.json")
        
        # Create mock video file
        with open(self.test_video_path, 'w') as f:
            f.write("mock video content")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('core.video_analysis.analyze_video')
    def test_cli_basic_analysis(self, mock_analyze):
        """Test basic CLI video analysis."""
        # Mock analysis result
        mock_result = {
            'video_analysis': {
                'total_scenes': 3,
                'total_movements': 2,
                'validation': {'status': 'PASS', 'message': 'Success'}
            }
        }
        mock_analyze.return_value = mock_result
        
        # Import and test CLI module
        import cli.analyze_video as analyze_cli
        
        # Mock sys.argv for argparse
        test_args = [
            'analyze_video.py',
            self.test_video_path,
            '--output', self.test_output_path,
            '--scene-threshold', '0.3',
            '--movement-threshold', '0.2'
        ]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                with patch('sys.exit') as mock_exit:
                    analyze_cli.main()
        
        # Verify analyze_video was called with correct parameters
        mock_analyze.assert_called_once()
        call_args = mock_analyze.call_args
        self.assertEqual(call_args[1]['video_path'], self.test_video_path)
        self.assertEqual(call_args[1]['output_path'], self.test_output_path)
        self.assertEqual(call_args[1]['scene_threshold'], 0.3)
        self.assertEqual(call_args[1]['movement_threshold'], 0.2)
        
        # Verify output was printed
        mock_print.assert_called()
        print_call = mock_print.call_args[0][0]
        output_data = json.loads(print_call)
        self.assertEqual(output_data['status'], 'success')
        self.assertEqual(output_data['scene_count'], 3)
    
    def test_cli_load_transcript_cues(self):
        """Test loading transcript cues from JSON."""
        import cli.analyze_video as analyze_cli
        
        # Create test transcript
        transcript_data = {
            'slides': [
                {
                    'segments': [
                        {'kind': 'text', 'text': 'Hello'},
                        {'kind': 'cue', 'cue': '[transition]'},
                        {'kind': 'text', 'text': 'World'}
                    ]
                },
                {
                    'segments': [
                        {'kind': 'cue', 'cue': '[animation]'}
                    ]
                }
            ]
        }
        
        transcript_path = os.path.join(self.temp_dir, 'transcript.json')
        with open(transcript_path, 'w') as f:
            json.dump(transcript_data, f)
        
        result = analyze_cli.load_transcript_cues(transcript_path)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], '[transition]')
        self.assertEqual(result[1], '[animation]')
    
    def test_cli_generate_output_path(self):
        """Test output path generation."""
        import cli.analyze_video as analyze_cli
        
        result = analyze_cli.generate_output_path(self.test_video_path)
        
        self.assertTrue(result.endswith('_analysis.json'))
        self.assertIn('workspace/video_analysis', result)
    
    @patch('core.video_analysis.probe_video_info')
    def test_cli_validate_video_file(self, mock_probe):
        """Test video file validation."""
        import cli.analyze_video as analyze_cli
        
        # Test valid video
        mock_probe.return_value = {'width': 1920, 'height': 1080}
        self.assertTrue(analyze_cli.validate_video_file(self.test_video_path))
        
        # Test non-existent video
        self.assertFalse(analyze_cli.validate_video_file('/nonexistent/video.mov'))
        
        # Test invalid video
        mock_probe.side_effect = Exception("Invalid video")
        self.assertFalse(analyze_cli.validate_video_file(self.test_video_path))


class TestVideoAnalysisIntegration(unittest.TestCase):
    """Test video analysis integration with pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('core.wrappers.subprocess')
    def test_wrapper_function(self, mock_subprocess):
        """Test the wrapper function in core.wrappers."""
        from core.wrappers import analyze_video
        
        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            'step_id': 'analyze_video',
            'status': 'success',
            'scene_count': 3,
            'movement_count': 2
        })
        mock_subprocess.run.return_value = mock_result
        
        # Mock Path.exists
        with patch('core.wrappers.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            result = analyze_video(
                video_path='/path/to/video.mov',
                scene_threshold=0.3,
                movement_threshold=0.2
            )
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['scene_count'], 3)
        
        # Verify subprocess was called correctly
        mock_subprocess.run.assert_called_once()
        call_args = mock_subprocess.run.call_args[0][0]
        self.assertIn('cli/analyze_video.py', call_args[1])
        self.assertIn('/path/to/video.mov', call_args)
        self.assertIn('--scene-threshold', call_args)
        self.assertIn('0.3', call_args)
    
    @patch('core.wrappers.subprocess')
    def test_wrapper_function_with_output_manifest(self, mock_subprocess):
        """Test wrapper function loading full manifest."""
        from core.wrappers import analyze_video
        
        # Create mock manifest file
        manifest_path = os.path.join(self.temp_dir, 'manifest.json')
        manifest_data = {
            'video_analysis': {
                'total_scenes': 3,
                'total_movements': 2,
                'scene_transitions': [],
                'movement_ranges': []
            }
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f)
        
        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({'status': 'success'})
        mock_subprocess.run.return_value = mock_result
        
        result = analyze_video(
            video_path='/path/to/video.mov',
            output_path=manifest_path
        )
        
        self.assertEqual(result, manifest_data)
    
    @patch('core.wrappers.subprocess')
    def test_wrapper_function_error_handling(self, mock_subprocess):
        """Test wrapper function error handling."""
        from core.wrappers import analyze_video
        
        # Mock subprocess error
        mock_error = MagicMock()
        mock_error.stderr = "Analysis failed"
        mock_subprocess.run.side_effect = mock_subprocess.CalledProcessError(
            1, 'cmd', stderr="Analysis failed"
        )
        mock_subprocess.CalledProcessError = Exception
        
        with self.assertRaises(RuntimeError) as context:
            analyze_video('/path/to/video.mov')
        
        self.assertIn("Video analysis failed", str(context.exception))


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end video analysis workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_video = os.path.join(self.temp_dir, "test.mov")
        
        # Create mock video file
        with open(self.test_video, 'w') as f:
            f.write("mock video")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('core.video_analysis.ffmpeg')
    def test_full_analysis_workflow(self, mock_ffmpeg):
        """Test complete analysis workflow."""
        # Mock video probe
        mock_ffmpeg.probe.return_value = {
            'streams': [
                {
                    'codec_type': 'video',
                    'width': 1920,
                    'height': 1080,
                    'duration': '30.0',
                    'r_frame_rate': '30/1',
                    'nb_frames': '900',
                    'codec_name': 'h264',
                    'pix_fmt': 'yuv420p',
                    'bit_rate': '5000000'
                }
            ]
        }
        
        # Mock scene detection
        mock_stderr = """
[Parsed_showinfo_1 @ 0x123] n:30 pts:1000 pts_time:1.0 pos:123456 fmt:yuv420p
[Parsed_showinfo_1 @ 0x456] n:60 pts:2000 pts_time:2.0 pos:234567 fmt:yuv420p
        """
        mock_error = MagicMock()
        mock_error.stderr.decode.return_value = mock_stderr
        mock_ffmpeg.run.side_effect = mock_ffmpeg.Error(mock_error)
        mock_ffmpeg.Error = Exception
        
        # Mock tempfile for scene detection
        with patch('core.video_analysis.tempfile') as mock_tempfile:
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/test.txt"
            mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = mock_temp_file
            
            # Run analysis
            output_path = os.path.join(self.temp_dir, "analysis.json")
            result = analyze_video(
                video_path=self.test_video,
                output_path=output_path,
                scene_threshold=0.4,
                keynote_delay=1.0
            )
        
        # Verify results
        self.assertIn('video_analysis', result)
        analysis = result['video_analysis']
        
        self.assertEqual(analysis['total_scenes'], 2)
        self.assertEqual(len(analysis['scene_transitions']), 2)
        
        # Check delay compensation was applied
        transitions = analysis['scene_transitions']
        self.assertEqual(transitions[0]['timestamp'], 0.0)  # 1.0 - 1.0 delay
        self.assertEqual(transitions[1]['timestamp'], 1.0)  # 2.0 - 1.0 delay
        
        # Verify manifest was saved
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r') as f:
            saved_manifest = json.load(f)
        
        self.assertEqual(saved_manifest, result)


if __name__ == '__main__':
    unittest.main()