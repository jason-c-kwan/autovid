"""
Test suite for RVC voice conversion functionality.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rvc_processing import (
    validate_rvc_environment,
    validate_rvc_model_files,
    get_rvc_config,
    build_rvc_command,
    setup_rvc_environment,
    parse_rvc_error
)


@pytest.fixture
def mock_rvc_environment(tmp_path):
    """Create mock RVC environment structure"""
    rvc_dir = tmp_path / "third_party" / "Mangio-RVC-Fork"
    rvc_dir.mkdir(parents=True)
    
    # Create essential files
    (rvc_dir / "infer_batch_rvc.py").touch()
    (rvc_dir / "vc_infer_pipeline.py").touch()
    (rvc_dir / "my_utils.py").touch()
    (rvc_dir / "lib" / "infer_pack").mkdir(parents=True)
    (rvc_dir / "lib" / "infer_pack" / "models.py").touch()
    
    return rvc_dir


@pytest.fixture
def mock_config_file(tmp_path):
    """Create mock configuration file"""
    config = {
        "rvc": {
            "executable": "third_party/Mangio-RVC-Fork/infer_batch_rvc.py",
            "model_path": "models/rvc/jason.pth",
            "index_path": "models/rvc/jason.index",
            "f0_up_key": 0,
            "f0_method": "harvest",
            "index_rate": 0.66,
            "device": "cuda:0",
            "is_half": True,
            "filter_radius": 3,
            "resample_sr": 0,
            "rms_mix_rate": 1.0,
            "protect": 0.33
        }
    }
    
    config_file = tmp_path / "pipeline.yaml"
    with open(config_file, 'w') as f:
        import yaml
        yaml.dump(config, f)
    
    return str(config_file)


@pytest.fixture
def mock_model_files(tmp_path):
    """Create mock RVC model files"""
    models_dir = tmp_path / "models" / "rvc"
    models_dir.mkdir(parents=True)
    
    model_file = models_dir / "jason.pth"
    index_file = models_dir / "jason.index"
    
    # Create dummy files with some content
    model_file.write_bytes(b"dummy model data")
    index_file.write_bytes(b"dummy index data")
    
    return str(model_file), str(index_file)


class TestRVCEnvironmentValidation:
    """Test RVC environment validation functions"""
    
    def test_validate_rvc_environment_missing_directory(self):
        """Test validation when RVC directory is missing"""
        with patch.object(Path, 'exists', return_value=False):
            is_valid, error_msg = validate_rvc_environment()
            assert not is_valid
            assert "Mangio-RVC-Fork not found" in error_msg
    
    def test_validate_rvc_environment_missing_cli(self, mock_rvc_environment, monkeypatch):
        """Test validation when CLI file is missing"""
        # Remove the CLI file
        (mock_rvc_environment / "infer_batch_rvc.py").unlink()
        
        # Mock the project root to point to our temp directory
        monkeypatch.setattr(Path(__file__).parent.parent, "exists", lambda: True)
        
        with patch("core.rvc_processing.Path") as mock_path:
            mock_path(__file__).parent.parent = mock_rvc_environment.parent.parent
            is_valid, error_msg = validate_rvc_environment()
            assert not is_valid
            assert "infer_batch_rvc.py not found" in error_msg
    
    def test_validate_rvc_environment_success(self, mock_rvc_environment, monkeypatch):
        """Test successful RVC environment validation"""
        with patch("core.rvc_processing.Path") as mock_path:
            mock_path(__file__).parent.parent = mock_rvc_environment.parent.parent
            is_valid, error_msg = validate_rvc_environment()
            assert is_valid
            assert error_msg == ""


class TestRVCModelValidation:
    """Test RVC model file validation"""
    
    def test_validate_model_files_missing_model(self, tmp_path):
        """Test validation when model file is missing"""
        with patch("core.rvc_processing.Path") as mock_path:
            mock_path(__file__).parent.parent = tmp_path
            is_valid, error_msg = validate_rvc_model_files("missing.pth", "missing.index")
            assert not is_valid
            assert "model file not found" in error_msg
    
    def test_validate_model_files_wrong_extension(self, mock_model_files, tmp_path):
        """Test validation with wrong file extensions"""
        model_file, index_file = mock_model_files
        
        # Create file with wrong extension
        wrong_model = Path(model_file).parent / "jason.txt"
        wrong_model.write_bytes(b"dummy")
        
        with patch("core.rvc_processing.Path") as mock_path:
            mock_path(__file__).parent.parent = tmp_path
            is_valid, error_msg = validate_rvc_model_files("models/rvc/jason.txt", index_file)
            assert not is_valid
            assert "must be .pth format" in error_msg
    
    def test_validate_model_files_success(self, mock_model_files, tmp_path):
        """Test successful model file validation"""
        model_file, index_file = mock_model_files
        
        with patch("core.rvc_processing.Path") as mock_path:
            mock_path(__file__).parent.parent = tmp_path
            is_valid, error_msg = validate_rvc_model_files(
                "models/rvc/jason.pth", 
                "models/rvc/jason.index"
            )
            assert is_valid
            assert error_msg == ""


class TestRVCConfiguration:
    """Test RVC configuration handling"""
    
    def test_get_rvc_config_missing_file(self):
        """Test configuration loading with missing file"""
        with pytest.raises(FileNotFoundError):
            get_rvc_config("nonexistent.yaml")
    
    def test_get_rvc_config_defaults(self, mock_config_file, tmp_path):
        """Test configuration loading with defaults"""
        with patch("core.rvc_processing.Path") as mock_path:
            mock_path(__file__).parent.parent = tmp_path
            config = get_rvc_config(mock_config_file)
            
            assert config["f0_method"] == "harvest"
            assert config["index_rate"] == 0.66
            assert config["device"] == "cuda:0"
    
    def test_get_rvc_config_custom_values(self, tmp_path):
        """Test configuration with custom values"""
        config_data = {
            "rvc": {
                "f0_method": "rmvpe",
                "index_rate": 0.8,
                "device": "cpu"
            }
        }
        
        config_file = tmp_path / "custom.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(config_data, f)
        
        with patch("core.rvc_processing.Path") as mock_path:
            mock_path(__file__).parent.parent = tmp_path
            config = get_rvc_config(str(config_file))
            
            assert config["f0_method"] == "rmvpe"
            assert config["index_rate"] == 0.8
            assert config["device"] == "cpu"


class TestRVCCommandBuilding:
    """Test RVC command construction"""
    
    def test_build_rvc_command_basic(self, tmp_path):
        """Test basic RVC command building"""
        with patch("core.rvc_processing.Path") as mock_path:
            mock_path(__file__).parent.parent = tmp_path
            
            cmd = build_rvc_command(
                executable_path="rvc/infer.py",
                model_path="models/model.pth",
                index_path="models/index.index",
                input_audio="input.wav",
                output_dir="output/",
                f0_up_key=2,
                f0_method="rmvpe",
                index_rate=0.75
            )
            
            assert cmd[1].endswith("infer.py")
            assert "2" in cmd  # f0_up_key
            assert "input.wav" in cmd
            assert "rmvpe" in cmd
            assert "0.75" in cmd
    
    def test_build_rvc_command_parameter_order(self, tmp_path):
        """Test RVC command parameter ordering"""
        with patch("core.rvc_processing.Path") as mock_path:
            mock_path(__file__).parent.parent = tmp_path
            
            cmd = build_rvc_command(
                executable_path="rvc/infer.py",
                model_path="models/model.pth",
                index_path="models/index.index",
                input_audio="input.wav",
                output_dir="output/"
            )
            
            # Check parameter order matches Mangio-RVC-Fork expectations
            assert len(cmd) == 15  # Expected number of parameters
            assert cmd[2] == "0"  # f0_up_key default
            assert cmd[3] == "input.wav"  # input audio
            assert cmd[5] == "harvest"  # f0_method default


class TestRVCErrorHandling:
    """Test RVC error parsing and handling"""
    
    def test_parse_rvc_error_cuda_memory(self):
        """Test parsing CUDA memory errors"""
        error_output = "RuntimeError: CUDA out of memory. Tried to allocate..."
        parsed = parse_rvc_error(error_output)
        
        assert "GPU memory error" in parsed
        assert "try reducing batch size" in parsed
    
    def test_parse_rvc_error_file_not_found(self):
        """Test parsing file not found errors"""
        error_output = "FileNotFoundError: No such file or directory: 'model.pth'"
        parsed = parse_rvc_error(error_output)
        
        assert "File not found" in parsed
        assert "check model and index paths" in parsed
    
    def test_parse_rvc_error_unknown(self):
        """Test parsing unknown errors"""
        error_output = "Some unknown error occurred"
        parsed = parse_rvc_error(error_output)
        
        assert "RVC Error:" in parsed
        assert "Some unknown error occurred" in parsed
    
    def test_parse_rvc_error_empty(self):
        """Test parsing empty error output"""
        parsed = parse_rvc_error("")
        assert "Unknown RVC error" in parsed


class TestRVCEnvironmentSetup:
    """Test RVC environment setup"""
    
    def test_setup_rvc_environment_pythonpath(self, tmp_path):
        """Test PYTHONPATH setup for RVC"""
        with patch("core.rvc_processing.Path") as mock_path:
            mock_path(__file__).parent.parent = tmp_path
            
            env = setup_rvc_environment()
            
            assert "PYTHONPATH" in env
            assert "Mangio-RVC-Fork" in env["PYTHONPATH"]
    
    def test_setup_rvc_environment_existing_pythonpath(self, tmp_path):
        """Test PYTHONPATH setup with existing value"""
        with patch("core.rvc_processing.Path") as mock_path:
            mock_path(__file__).parent.parent = tmp_path
            
            with patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}):
                env = setup_rvc_environment()
                
                assert "/existing/path" in env["PYTHONPATH"]
                assert "Mangio-RVC-Fork" in env["PYTHONPATH"]