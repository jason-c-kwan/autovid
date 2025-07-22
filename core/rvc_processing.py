"""
RVC processing functions for voice conversion integration.
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def validate_rvc_environment() -> Tuple[bool, str]:
    """
    Validate RVC third_party installation and dependencies.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    project_root = Path(__file__).parent.parent
    
    # Check if third_party/Mangio-RVC-Fork exists
    rvc_path = project_root / "third_party" / "Mangio-RVC-Fork"
    if not rvc_path.exists():
        return False, "Mangio-RVC-Fork not found in third_party/. Run: git submodule update --init --recursive"
    
    # Check if infer_batch_rvc.py exists
    cli_path = rvc_path / "infer_batch_rvc.py"
    if not cli_path.exists():
        return False, f"infer_batch_rvc.py not found at {cli_path}"
    
    # Check if essential RVC files exist
    essential_files = [
        "vc_infer_pipeline.py",
        "my_utils.py",
        "lib/infer_pack/models.py"
    ]
    
    for file_path in essential_files:
        if not (rvc_path / file_path).exists():
            return False, f"Essential RVC file missing: {file_path}"
    
    return True, ""


def validate_rvc_model_files(model_path: str, index_path: str) -> Tuple[bool, str]:
    """
    Validate RVC model and index files exist.
    
    Args:
        model_path: Path to RVC model (.pth file)
        index_path: Path to RVC index file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    project_root = Path(__file__).parent.parent
    
    model_file = project_root / model_path
    index_file = project_root / index_path
    
    if not model_file.exists():
        return False, f"RVC model file not found: {model_file}"
    
    if not index_file.exists():
        return False, f"RVC index file not found: {index_file}"
    
    # Check file extensions
    if not model_file.suffix == ".pth":
        return False, f"RVC model file must be .pth format: {model_file}"
    
    if not index_file.suffix == ".index":
        return False, f"RVC index file must be .index format: {index_file}"
    
    return True, ""


def get_rvc_config(config_path: str = "config/pipeline.yaml") -> Dict[str, Any]:
    """
    Extract RVC configuration from pipeline config.
    
    Args:
        config_path: Path to pipeline configuration file
        
    Returns:
        Dictionary of RVC configuration parameters
    """
    project_root = Path(__file__).parent.parent
    config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    rvc_config = config.get("rvc", {})
    
    # Set defaults if not specified
    defaults = {
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
    
    # Merge defaults with config
    for key, value in defaults.items():
        if key not in rvc_config:
            rvc_config[key] = value
    
    # Resolve relative paths to absolute paths
    for path_key in ["model_path", "index_path"]:
        if path_key in rvc_config:
            path_value = rvc_config[path_key]
            if not Path(path_value).is_absolute():
                rvc_config[path_key] = str(project_root / path_value)
    
    return rvc_config


def build_rvc_command(
    executable_path: str,
    model_path: str,
    index_path: str,
    input_audio: str,
    output_dir: str,
    **rvc_params
) -> List[str]:
    """
    Build RVC CLI command with proper parameter ordering.
    
    Args:
        executable_path: Path to RVC executable
        model_path: Path to RVC model
        index_path: Path to RVC index
        input_audio: Input audio file path
        output_dir: Output directory for converted audio
        **rvc_params: RVC parameters from config
        
    Returns:
        List of command arguments
    """
    project_root = Path(__file__).parent.parent
    
    # Resolve paths
    executable = project_root / executable_path
    model = project_root / model_path
    index = project_root / index_path
    
    # RVC batch command parameters (in order):
    # f0up_key, input_path, index_path, f0method, opt_path, model_path, 
    # index_rate, device, is_half, filter_radius, resample_sr, rms_mix_rate, protect
    
    cmd = [
        sys.executable,
        str(executable),
        str(rvc_params.get("f0_up_key", 0)),
        input_audio,
        str(index),
        rvc_params.get("f0_method", "harvest"),
        output_dir,
        str(model),
        str(rvc_params.get("index_rate", 0.66)),
        rvc_params.get("device", "cuda:0"),
        str(rvc_params.get("is_half", True)),
        str(rvc_params.get("filter_radius", 3)),
        str(rvc_params.get("resample_sr", 0)),
        str(rvc_params.get("rms_mix_rate", 1.0)),
        str(rvc_params.get("protect", 0.33))
    ]
    
    return cmd


def setup_rvc_environment() -> Dict[str, str]:
    """
    Setup environment variables for RVC execution.
    
    Returns:
        Environment variables dictionary
    """
    project_root = Path(__file__).parent.parent
    rvc_path = project_root / "third_party" / "Mangio-RVC-Fork"
    
    # Copy current environment
    env = os.environ.copy()
    
    # Add RVC path to Python path for imports
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{rvc_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(rvc_path)
    
    # Set working directory for RVC
    env["RVC_WORKING_DIR"] = str(rvc_path)
    
    return env


def create_rvc_workspace(output_dir: str) -> str:
    """
    Create RVC workspace directory.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Path to created workspace directory
    """
    workspace_path = Path(output_dir)
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    return str(workspace_path)


def validate_rvc_output(output_file: str) -> Tuple[bool, str]:
    """
    Validate RVC conversion output.
    
    Args:
        output_file: Path to output audio file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    output_path = Path(output_file)
    
    if not output_path.exists():
        return False, f"RVC output file not found: {output_file}"
    
    # Check file size (should be > 0)
    if output_path.stat().st_size == 0:
        return False, f"RVC output file is empty: {output_file}"
    
    # Check file extension
    if output_path.suffix.lower() not in ['.wav', '.mp3', '.flac']:
        return False, f"RVC output file has invalid extension: {output_file}"
    
    return True, ""


def parse_rvc_error(error_output: str) -> str:
    """
    Parse RVC error output for meaningful error messages.
    
    Args:
        error_output: Raw error output from RVC
        
    Returns:
        Formatted error message
    """
    if not error_output:
        return "Unknown RVC error"
    
    # Common RVC error patterns
    error_patterns = [
        ("CUDA out of memory", "GPU memory error - try reducing batch size or using CPU"),
        ("No such file or directory", "File not found - check model and index paths"),
        ("ModuleNotFoundError", "Missing dependencies - check RVC installation"),
        ("RuntimeError", "RVC runtime error - check model compatibility"),
        ("ValueError", "Invalid parameter values - check configuration")
    ]
    
    for pattern, message in error_patterns:
        if pattern in error_output:
            return f"RVC Error: {message}\nDetails: {error_output}"
    
    return f"RVC Error: {error_output}"