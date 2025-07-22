"""
Test suite for audio splicing functionality.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audio_splicing import (
    load_audio_file,
    apply_crossfade,
    normalize_audio,
    concatenate_audio_chunks,
    calculate_timing_metadata,
    save_audio_file,
    validate_audio_chunks,
    get_audio_splicing_config
)


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing"""
    # Create 1 second of audio at 44100 Hz
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate sine wave
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    
    return audio, sample_rate


@pytest.fixture
def temp_audio_files(sample_audio_data, tmp_path):
    """Create temporary audio files for testing"""
    audio, sample_rate = sample_audio_data
    
    files = []
    for i in range(3):
        file_path = tmp_path / f"chunk_{i}.wav"
        
        # Mock soundfile.write
        with patch('core.audio_splicing.sf.write') as mock_write:
            with patch('core.audio_splicing.sf.read', return_value=(audio, sample_rate)):
                # Just create the file, soundfile operations are mocked
                file_path.touch()
                files.append(str(file_path))
    
    return files


@pytest.fixture
def mock_config_file(tmp_path):
    """Create mock configuration file for audio splicing"""
    config = {
        "audio_splicing": {
            "crossfade_duration": 0.1,
            "target_db": -20.0,
            "normalize": True
        }
    }
    
    config_file = tmp_path / "pipeline.yaml"
    with open(config_file, 'w') as f:
        import yaml
        yaml.dump(config, f)
    
    return str(config_file)


class TestAudioLoading:
    """Test audio file loading functionality"""
    
    def test_load_audio_file_success(self, sample_audio_data):
        """Test successful audio file loading"""
        audio, sample_rate = sample_audio_data
        
        with patch('core.audio_splicing.sf.read', return_value=(audio, sample_rate)):
            with patch('core.audio_splicing.Path.exists', return_value=True):
                loaded_audio, loaded_sr = load_audio_file("test.wav")
                
                assert np.array_equal(loaded_audio, audio)
                assert loaded_sr == sample_rate
    
    def test_load_audio_file_not_found(self):
        """Test loading non-existent audio file"""
        with pytest.raises(FileNotFoundError):
            load_audio_file("nonexistent.wav")
    
    def test_load_audio_file_read_error(self):
        """Test audio file read error"""
        with patch('core.audio_splicing.Path.exists', return_value=True):
            with patch('core.audio_splicing.sf.read', side_effect=Exception("Read error")):
                with pytest.raises(RuntimeError, match="Failed to load audio file"):
                    load_audio_file("test.wav")


class TestCrossfade:
    """Test crossfade functionality"""
    
    def test_apply_crossfade_basic(self):
        """Test basic crossfade application"""
        # Create two audio segments
        audio1 = np.ones(1000) * 0.5
        audio2 = np.ones(1000) * -0.5
        crossfade_samples = 100
        
        result = apply_crossfade(audio1, audio2, crossfade_samples)
        
        # Check result length
        expected_length = len(audio1) + len(audio2) - crossfade_samples
        assert len(result) == expected_length
        
        # Check crossfade region has intermediate values
        crossfade_region = result[len(audio1) - crossfade_samples:len(audio1)]
        assert np.all(crossfade_region != 0.5)  # Not just audio1
        assert np.all(crossfade_region != -0.5)  # Not just audio2
    
    def test_apply_crossfade_empty_audio(self):
        """Test crossfade with empty audio"""
        audio1 = np.array([])
        audio2 = np.ones(100)
        
        result = apply_crossfade(audio1, audio2, 10)
        assert np.array_equal(result, audio2)
    
    def test_apply_crossfade_no_overlap(self):
        """Test crossfade with zero overlap"""
        audio1 = np.ones(100)
        audio2 = np.ones(100) * 2
        
        result = apply_crossfade(audio1, audio2, 0)
        expected = np.concatenate([audio1, audio2])
        assert np.array_equal(result, expected)
    
    def test_apply_crossfade_stereo(self):
        """Test crossfade with stereo audio"""
        audio1 = np.ones((100, 2)) * 0.5
        audio2 = np.ones((100, 2)) * -0.5
        crossfade_samples = 20
        
        result = apply_crossfade(audio1, audio2, crossfade_samples)
        
        # Check shape is preserved
        assert result.shape[1] == 2  # Stereo
        
        # Check length
        expected_length = len(audio1) + len(audio2) - crossfade_samples
        assert len(result) == expected_length


class TestAudioNormalization:
    """Test audio normalization functionality"""
    
    def test_normalize_audio_basic(self):
        """Test basic audio normalization"""
        # Create audio with known RMS
        audio = np.ones(1000) * 0.1  # Low level audio
        target_db = -20.0
        
        normalized = normalize_audio(audio, target_db)
        
        # Check that audio level increased
        original_rms = np.sqrt(np.mean(audio**2))
        normalized_rms = np.sqrt(np.mean(normalized**2))
        assert normalized_rms > original_rms
    
    def test_normalize_audio_empty(self):
        """Test normalization with empty audio"""
        audio = np.array([])
        result = normalize_audio(audio)
        assert len(result) == 0
    
    def test_normalize_audio_silent(self):
        """Test normalization with silent audio"""
        audio = np.zeros(1000)
        result = normalize_audio(audio)
        assert np.array_equal(result, audio)  # Should remain unchanged
    
    def test_normalize_audio_clipping_prevention(self):
        """Test that normalization prevents clipping"""
        # Create loud audio
        audio = np.ones(1000) * 0.9
        target_db = 0.0  # Very loud target
        max_db = -1.0  # Clipping prevention
        
        normalized = normalize_audio(audio, target_db, max_db)
        
        # Check no clipping occurred
        assert np.max(np.abs(normalized)) <= 1.0


class TestAudioConcatenation:
    """Test audio concatenation functionality"""
    
    def test_concatenate_audio_chunks_basic(self, temp_audio_files, sample_audio_data):
        """Test basic audio chunk concatenation"""
        audio, sample_rate = sample_audio_data
        
        with patch('core.audio_splicing.load_audio_file', return_value=(audio, sample_rate)):
            result_audio, result_sr = concatenate_audio_chunks(
                temp_audio_files,
                crossfade_duration=0.1,
                normalize=False
            )
            
            assert result_sr == sample_rate
            # Should be roughly 3 chunks minus crossfades
            expected_length = 3 * len(audio) - 2 * int(0.1 * sample_rate)
            assert abs(len(result_audio) - expected_length) < sample_rate * 0.01  # Small tolerance
    
    def test_concatenate_audio_chunks_empty_list(self):
        """Test concatenation with empty chunk list"""
        with pytest.raises(ValueError, match="No audio chunks provided"):
            concatenate_audio_chunks([])
    
    def test_concatenate_audio_chunks_sample_rate_mismatch(self, temp_audio_files, sample_audio_data):
        """Test concatenation with mismatched sample rates"""
        audio, sample_rate = sample_audio_data
        
        def mock_load_audio(path):
            if "chunk_0" in path:
                return audio, 44100
            else:
                return audio, 48000  # Different sample rate
        
        with patch('core.audio_splicing.load_audio_file', side_effect=mock_load_audio):
            with pytest.raises(ValueError, match="Sample rate mismatch"):
                concatenate_audio_chunks(temp_audio_files)
    
    def test_concatenate_audio_chunks_with_normalization(self, temp_audio_files, sample_audio_data):
        """Test concatenation with normalization enabled"""
        audio, sample_rate = sample_audio_data
        
        with patch('core.audio_splicing.load_audio_file', return_value=(audio * 0.1, sample_rate)):
            result_audio, result_sr = concatenate_audio_chunks(
                temp_audio_files,
                normalize=True,
                target_db=-20.0
            )
            
            # Check that audio was normalized (louder than input)
            input_rms = np.sqrt(np.mean((audio * 0.1)**2))
            result_rms = np.sqrt(np.mean(result_audio**2))
            assert result_rms > input_rms


class TestTimingMetadata:
    """Test timing metadata calculation"""
    
    def test_calculate_timing_metadata_basic(self):
        """Test basic timing metadata calculation"""
        chunks = [
            {"id": "chunk_0", "duration": 2.0, "text": "Hello"},
            {"id": "chunk_1", "duration": 1.5, "text": "World"},
            {"id": "chunk_2", "duration": 3.0, "text": "Test"}
        ]
        
        timing = calculate_timing_metadata(chunks, crossfade_duration=0.1)
        
        assert timing["total_duration"] == 6.3  # 2.0 + 1.5 + 3.0 - 2*0.1
        assert timing["crossfade_duration"] == 0.1
        assert len(timing["chunks"]) == 3
        
        # Check first chunk timing
        assert timing["chunks"][0]["start_time"] == 0.0
        assert timing["chunks"][0]["end_time"] == 2.0
        
        # Check second chunk timing (with crossfade)
        assert timing["chunks"][1]["start_time"] == 1.9  # 2.0 - 0.1
        assert timing["chunks"][1]["end_time"] == 3.4  # 1.9 + 1.5
    
    def test_calculate_timing_metadata_empty(self):
        """Test timing metadata with empty chunks"""
        timing = calculate_timing_metadata([])
        
        assert timing["total_duration"] == 0.0
        assert len(timing["chunks"]) == 0
    
    def test_calculate_timing_metadata_single_chunk(self):
        """Test timing metadata with single chunk"""
        chunks = [{"id": "chunk_0", "duration": 2.0, "text": "Solo"}]
        
        timing = calculate_timing_metadata(chunks, crossfade_duration=0.1)
        
        assert timing["total_duration"] == 2.0
        assert timing["chunks"][0]["start_time"] == 0.0
        assert timing["chunks"][0]["end_time"] == 2.0


class TestAudioValidation:
    """Test audio chunk validation"""
    
    def test_validate_audio_chunks_empty(self):
        """Test validation with empty chunk list"""
        is_valid, error_msg = validate_audio_chunks([])
        assert not is_valid
        assert "No audio chunks provided" in error_msg
    
    def test_validate_audio_chunks_missing_file(self):
        """Test validation with missing file"""
        is_valid, error_msg = validate_audio_chunks(["nonexistent.wav"])
        assert not is_valid
        assert "not found" in error_msg
    
    def test_validate_audio_chunks_success(self, temp_audio_files, sample_audio_data):
        """Test successful chunk validation"""
        audio, sample_rate = sample_audio_data
        
        with patch('core.audio_splicing.load_audio_file', return_value=(audio, sample_rate)):
            is_valid, error_msg = validate_audio_chunks(temp_audio_files)
            assert is_valid
            assert error_msg == ""
    
    def test_validate_audio_chunks_sample_rate_mismatch(self, temp_audio_files, sample_audio_data):
        """Test validation with sample rate mismatch"""
        audio, _ = sample_audio_data
        
        def mock_load_audio(path):
            if "chunk_0" in path:
                return audio, 44100
            else:
                return audio, 48000
        
        with patch('core.audio_splicing.load_audio_file', side_effect=mock_load_audio):
            is_valid, error_msg = validate_audio_chunks(temp_audio_files)
            assert not is_valid
            assert "Sample rate mismatch" in error_msg


class TestSplicingConfiguration:
    """Test audio splicing configuration"""
    
    def test_get_audio_splicing_config_missing_file(self):
        """Test configuration loading with missing file"""
        with pytest.raises(FileNotFoundError):
            get_audio_splicing_config("nonexistent.yaml")
    
    def test_get_audio_splicing_config_defaults(self, mock_config_file, tmp_path):
        """Test configuration loading with defaults"""
        with patch("core.audio_splicing.Path") as mock_path:
            mock_path(__file__).parent.parent = tmp_path
            config = get_audio_splicing_config(mock_config_file)
            
            assert config["crossfade_duration"] == 0.1
            assert config["target_db"] == -20.0
            assert config["normalize"] is True
    
    def test_get_audio_splicing_config_missing_section(self, tmp_path):
        """Test configuration with missing audio_splicing section"""
        config_data = {"other_section": {"key": "value"}}
        
        config_file = tmp_path / "minimal.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(config_data, f)
        
        with patch("core.audio_splicing.Path") as mock_path:
            mock_path(__file__).parent.parent = tmp_path
            config = get_audio_splicing_config(str(config_file))
            
            # Should use defaults
            assert config["crossfade_duration"] == 0.1
            assert config["target_db"] == -20.0
            assert config["normalize"] is True


class TestAudioSaving:
    """Test audio file saving functionality"""
    
    def test_save_audio_file_success(self, sample_audio_data, tmp_path):
        """Test successful audio file saving"""
        audio, sample_rate = sample_audio_data
        output_path = tmp_path / "output.wav"
        
        with patch('core.audio_splicing.sf.write') as mock_write:
            save_audio_file(audio, sample_rate, str(output_path))
            mock_write.assert_called_once_with(str(output_path), audio, sample_rate, format="WAV")
    
    def test_save_audio_file_error(self, sample_audio_data, tmp_path):
        """Test audio file saving error"""
        audio, sample_rate = sample_audio_data
        output_path = tmp_path / "output.wav"
        
        with patch('core.audio_splicing.sf.write', side_effect=Exception("Write error")):
            with pytest.raises(RuntimeError, match="Failed to save audio file"):
                save_audio_file(audio, sample_rate, str(output_path))