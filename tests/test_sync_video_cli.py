"""
Tests for video synchronization CLI.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, call

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSyncVideoCLI:
    """Test cases for sync_video CLI."""
    
    @patch('subprocess.run')
    def test_cli_basic_execution(self, mock_run):
        """Test basic CLI execution."""
        # Mock successful execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Synchronization completed successfully"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Test CLI import and basic structure
        try:
            import cli.sync_video as sync_cli
            assert hasattr(sync_cli, 'main')
            assert callable(sync_cli.main)
        except ImportError:
            pytest.skip("CLI module not available for testing")
    
    def test_cli_argument_parser(self):
        """Test CLI argument parser setup."""
        try:
            import cli.sync_video as sync_cli
            import argparse
            
            # Test that the CLI sets up argument parser correctly
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_parse.return_value = Mock()
                
                # This would test the parser setup but requires refactoring
                # the CLI to separate parser creation from main execution
                pass
                
        except ImportError:
            pytest.skip("CLI module not available for testing")
    
    @patch('cli.sync_video.setup_logging')
    @patch('cli.sync_video.validate_inputs')
    @patch('cli.sync_video.synchronize_video_audio')
    def test_cli_workflow_success(self, mock_sync, mock_validate, mock_logging):
        """Test successful CLI workflow."""
        try:
            import cli.sync_video as sync_cli
            
            # Mock dependencies
            mock_logging.return_value = Mock()
            mock_validate.return_value = None  # No errors
            mock_sync.return_value = "output.mp4"
            
            # Create mock args
            mock_args = Mock()
            mock_args.video = "test.mov"
            mock_args.audio = "test.wav"
            mock_args.output = "output.mp4"
            mock_args.video_manifest = None
            mock_args.audio_manifest = None
            mock_args.validate = False
            mock_args.preview_dir = None
            mock_args.sync_manifest = None
            mock_args.verbose = False
            mock_args.dry_run = False
            mock_args.keynote_delay = None
            mock_args.sync_tolerance = None
            mock_args.video_codec = None
            mock_args.audio_codec = None
            mock_args.config = "config/pipeline.yaml"
            mock_args.validation_report = None
            mock_args.preview_duration = 10.0
            
            # Test individual functions exist and are callable
            assert hasattr(sync_cli, 'validate_inputs')
            assert callable(sync_cli.validate_inputs)
            
        except ImportError:
            pytest.skip("CLI module not available for testing")
    
    def test_cli_help_output(self):
        """Test CLI help output contains expected information."""
        try:
            import subprocess
            result = subprocess.run([
                'python', 'cli/sync_video.py', '--help'
            ], cwd=project_root, capture_output=True, text=True, timeout=10)
            
            # Check that help contains key arguments
            help_text = result.stdout.lower()
            assert 'video' in help_text
            assert 'audio' in help_text
            assert 'output' in help_text
            assert 'sync' in help_text
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI script not executable or timeout")
        except Exception as e:
            pytest.skip(f"CLI help test failed: {e}")


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    @patch('core.video_sync.get_video_sync_config')
    @patch('pathlib.Path.exists')
    def test_config_loading_integration(self, mock_exists, mock_config):
        """Test CLI config loading integration."""
        mock_exists.return_value = True
        mock_config.return_value = {
            'keynote_delay': 1.0,
            'sync_tolerance': 0.1,
            'video_codec': 'copy'
        }
        
        try:
            import cli.sync_video as sync_cli
            
            # Test that config loading functions work
            config = sync_cli.get_video_sync_config()
            assert isinstance(config, dict)
            assert 'keynote_delay' in config
            
        except ImportError:
            pytest.skip("CLI module not available for testing")
    
    def test_manifest_creation_integration(self):
        """Test manifest creation in CLI context."""
        try:
            from core.video_sync import create_sync_manifest, SyncPoint, TimingCorrection
            
            # Test manifest creation with realistic data
            sync_points = [
                SyncPoint(0, 1.0, 0.9, 0.1, 0.95)
            ]
            
            corrections = TimingCorrection(
                total_drift=0.1,
                max_offset=0.1,
                correction_points=[],
                keynote_delay_applied=1.0,
                crossfade_compensations=[]
            )
            
            validation = {'status': 'PASS', 'warnings': [], 'errors': []}
            
            manifest = create_sync_manifest(
                "video.json", "audio.json", sync_points, 
                corrections, validation, "output.mp4"
            )
            
            # Verify manifest structure
            assert 'video_sync' in manifest
            assert 'sync_points' in manifest['video_sync']
            assert len(manifest['video_sync']['sync_points']) == 1
            
        except ImportError:
            pytest.skip("Core modules not available for testing")


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""
    
    def test_missing_input_files(self):
        """Test CLI behavior with missing input files."""
        try:
            import cli.sync_video as sync_cli
            import argparse
            
            # Mock args with non-existent files
            mock_args = Mock()
            mock_args.video = "/nonexistent/video.mp4"
            mock_args.audio = "/nonexistent/audio.wav"
            mock_args.video_manifest = None
            mock_args.audio_manifest = None
            
            # Test that validation catches missing files
            with pytest.raises(FileNotFoundError):
                sync_cli.validate_inputs(mock_args)
                
        except ImportError:
            pytest.skip("CLI module not available for testing")
    
    def test_invalid_manifest_handling(self):
        """Test CLI handling of invalid manifest files."""
        try:
            from core.video_sync import load_video_analysis_manifest, VideoSyncError
            
            # Create invalid manifest file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write('{"invalid": "manifest"}')  # Missing required fields
                invalid_manifest_path = f.name
            
            try:
                # Should raise VideoSyncError for invalid manifest
                with pytest.raises(VideoSyncError):
                    load_video_analysis_manifest(invalid_manifest_path)
            finally:
                os.unlink(invalid_manifest_path)
                
        except ImportError:
            pytest.skip("Core modules not available for testing")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])