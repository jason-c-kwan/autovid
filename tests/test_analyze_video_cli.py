"""
Test suite specifically for the analyze_video CLI tool.

This module provides focused tests for the CLI interface and command-line
argument parsing functionality.
"""

import unittest
import tempfile
import os
import json
import shutil
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnalyzeVideoCLI(unittest.TestCase):
    """Test the analyze_video CLI tool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_video = os.path.join(self.temp_dir, "test_video.mov")
        self.test_output = os.path.join(self.temp_dir, "output.json")
        self.test_transcript = os.path.join(self.temp_dir, "transcript.json")
        
        # Create mock video file
        with open(self.test_video, 'w') as f:
            f.write("mock video content")
        
        # Create mock transcript
        transcript_data = {
            "slides": [
                {
                    "segments": [
                        {"kind": "cue", "cue": "[transition]"}
                    ]
                }
            ]
        }
        with open(self.test_transcript, 'w') as f:
            json.dump(transcript_data, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('core.video_analysis.analyze_video')
    @patch('core.video_analysis.probe_video_info')
    def test_cli_basic_usage(self, mock_probe, mock_analyze):
        """Test basic CLI usage with minimal arguments."""
        # Mock video probe for validation
        mock_probe.return_value = {'width': 1920, 'height': 1080}
        
        # Mock analysis result
        mock_result = {
            'video_analysis': {
                'total_scenes': 2,
                'total_movements': 1,
                'validation': {'status': 'PASS', 'message': 'Success'}
            }
        }
        mock_analyze.return_value = mock_result
        
        # Import CLI module
        import cli.analyze_video as analyze_cli
        
        # Test arguments
        test_args = ['analyze_video.py', self.test_video]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                with patch('sys.exit') as mock_exit:
                    analyze_cli.main()
        
        # Verify analysis was called
        mock_analyze.assert_called_once()
        
        # Verify success output
        mock_print.assert_called()
        output = json.loads(mock_print.call_args[0][0])
        self.assertEqual(output['status'], 'success')
        self.assertEqual(output['scene_count'], 2)
    
    @patch('core.video_analysis.analyze_video')
    @patch('core.video_analysis.probe_video_info')
    def test_cli_with_all_options(self, mock_probe, mock_analyze):
        """Test CLI with all available options."""
        # Mock video probe
        mock_probe.return_value = {'width': 1920, 'height': 1080}
        
        # Mock analysis result
        mock_result = {
            'video_analysis': {
                'total_scenes': 3,
                'total_movements': 2,
                'validation': {'status': 'WARN', 'message': 'Mismatch detected'}
            }
        }
        mock_analyze.return_value = mock_result
        
        import cli.analyze_video as analyze_cli
        
        test_args = [
            'analyze_video.py',
            self.test_video,
            '--output', self.test_output,
            '--transcript', self.test_transcript,
            '--scene-threshold', '0.3',
            '--movement-threshold', '0.2',
            '--keynote-delay', '0.5',
            '--verbose',
            '--step-id', 'custom_step'
        ]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                with patch('sys.exit') as mock_exit:
                    analyze_cli.main()
        
        # Verify analysis was called with correct parameters
        call_kwargs = mock_analyze.call_args[1]
        self.assertEqual(call_kwargs['video_path'], self.test_video)
        self.assertEqual(call_kwargs['output_path'], self.test_output)
        self.assertEqual(call_kwargs['scene_threshold'], 0.3)
        self.assertEqual(call_kwargs['movement_threshold'], 0.2)
        self.assertEqual(call_kwargs['keynote_delay'], 0.5)
        self.assertIsNotNone(call_kwargs['expected_transitions'])
        
        # Verify output
        output = json.loads(mock_print.call_args[0][0])
        self.assertEqual(output['step_id'], 'custom_step')
        self.assertEqual(output['validation_status'], 'WARN')
    
    @patch('core.video_analysis.probe_video_info')
    def test_cli_invalid_video_file(self, mock_probe):
        """Test CLI with invalid video file."""
        # Mock probe to raise exception
        mock_probe.side_effect = Exception("Invalid video")
        
        import cli.analyze_video as analyze_cli
        
        test_args = ['analyze_video.py', self.test_video]
        
        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                analyze_cli.main()
        
        # Verify exit was called with error code
        mock_exit.assert_called_with(1)
    
    def test_cli_nonexistent_video_file(self):
        """Test CLI with non-existent video file."""
        import cli.analyze_video as analyze_cli
        
        nonexistent_video = "/nonexistent/video.mov"
        test_args = ['analyze_video.py', nonexistent_video]
        
        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                analyze_cli.main()
        
        mock_exit.assert_called_with(1)
    
    @patch('core.video_analysis.analyze_video')
    @patch('core.video_analysis.probe_video_info')
    def test_cli_analysis_error(self, mock_probe, mock_analyze):
        """Test CLI handling of analysis errors."""
        # Mock video probe
        mock_probe.return_value = {'width': 1920, 'height': 1080}
        
        # Mock analysis to raise error
        from core.video_analysis import VideoAnalysisError
        mock_analyze.side_effect = VideoAnalysisError("Analysis failed")
        
        import cli.analyze_video as analyze_cli
        
        test_args = ['analyze_video.py', self.test_video]
        
        with patch('sys.argv', test_args):
            with patch('builtins.print') as mock_print:
                with patch('sys.exit') as mock_exit:
                    analyze_cli.main()
        
        # Verify error output
        output = json.loads(mock_print.call_args[0][0])
        self.assertEqual(output['status'], 'error')
        self.assertIn('Analysis failed', output['error_message'])
        
        # Verify exit with error code
        mock_exit.assert_called_with(1)
    
    def test_cli_invalid_threshold_values(self):
        """Test CLI with invalid threshold values."""
        import cli.analyze_video as analyze_cli
        
        # Test invalid scene threshold
        test_args = [
            'analyze_video.py',
            self.test_video,
            '--scene-threshold', '1.5'  # > 1.0
        ]
        
        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                analyze_cli.main()
        
        mock_exit.assert_called_with(1)
    
    def test_cli_invalid_keynote_delay(self):
        """Test CLI with invalid keynote delay."""
        import cli.analyze_video as analyze_cli
        
        test_args = [
            'analyze_video.py',
            self.test_video,
            '--keynote-delay', '-0.5'  # negative
        ]
        
        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                analyze_cli.main()
        
        mock_exit.assert_called_with(1)
    
    @patch('core.video_analysis.probe_video_info')
    def test_cli_dry_run(self, mock_probe):
        """Test CLI dry run functionality."""
        # Mock video probe
        mock_probe.return_value = {'width': 1920, 'height': 1080}
        
        import cli.analyze_video as analyze_cli
        
        test_args = [
            'analyze_video.py',
            self.test_video,
            '--dry-run'
        ]
        
        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                analyze_cli.main()
        
        # Verify successful dry run
        mock_exit.assert_called_with(0)
    
    def test_cli_transcript_loading_error(self):
        """Test CLI with invalid transcript file."""
        import cli.analyze_video as analyze_cli
        
        # Create invalid transcript file
        invalid_transcript = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_transcript, 'w') as f:
            f.write("invalid json content {")
        
        # Load transcript cues should handle the error gracefully
        result = analyze_cli.load_transcript_cues(invalid_transcript)
        
        # The function should raise ValueError for invalid JSON
        with self.assertRaises(ValueError):
            analyze_cli.load_transcript_cues(invalid_transcript)
    
    def test_cli_generate_output_path(self):
        """Test output path generation."""
        import cli.analyze_video as analyze_cli
        
        video_path = "/path/to/video.mov"
        result = analyze_cli.generate_output_path(video_path)
        
        self.assertTrue(result.endswith("_analysis.json"))
        self.assertIn("workspace/video_analysis", result)
        
        # Test with custom suffix
        result_custom = analyze_cli.generate_output_path(video_path, "_custom")
        self.assertTrue(result_custom.endswith("_custom.json"))
    
    def test_cli_load_transcript_cues_alternative_format(self):
        """Test loading transcript cues from alternative format."""
        import cli.analyze_video as analyze_cli
        
        # Create alternative transcript format
        alt_transcript_data = {
            "transcript": [
                {"type": "cue", "text": "[slide_transition]"},
                {"type": "text", "text": "Some text"},
                {"type": "cue", "text": "[animation_start]"}
            ]
        }
        
        alt_transcript_path = os.path.join(self.temp_dir, "alt_transcript.json")
        with open(alt_transcript_path, 'w') as f:
            json.dump(alt_transcript_data, f)
        
        result = analyze_cli.load_transcript_cues(alt_transcript_path)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "[slide_transition]")
        self.assertEqual(result[1], "[animation_start]")
    
    @patch('core.video_analysis.analyze_video')
    @patch('core.video_analysis.probe_video_info')
    def test_cli_keyboard_interrupt(self, mock_probe, mock_analyze):
        """Test CLI handling of keyboard interrupt."""
        # Mock video probe
        mock_probe.return_value = {'width': 1920, 'height': 1080}
        
        # Mock analysis to raise KeyboardInterrupt
        mock_analyze.side_effect = KeyboardInterrupt()
        
        import cli.analyze_video as analyze_cli
        
        test_args = ['analyze_video.py', self.test_video]
        
        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                analyze_cli.main()
        
        # Verify exit with interrupt code
        mock_exit.assert_called_with(130)


class TestCLIHelpers(unittest.TestCase):
    """Test CLI helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_validate_video_file_permissions(self):
        """Test video file validation with permission issues."""
        import cli.analyze_video as analyze_cli
        
        # Create a file with restricted permissions
        restricted_file = os.path.join(self.temp_dir, "restricted.mov")
        with open(restricted_file, 'w') as f:
            f.write("content")
        
        # Make file unreadable
        os.chmod(restricted_file, 0o000)
        
        try:
            result = analyze_cli.validate_video_file(restricted_file)
            # On some systems, this might still return True due to root access
            # The important thing is that it doesn't crash
            self.assertIsInstance(result, bool)
        finally:
            # Restore permissions for cleanup
            os.chmod(restricted_file, 0o644)


if __name__ == '__main__':
    unittest.main()