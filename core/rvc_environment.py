"""
RVC environment management for isolated execution.
"""

import os
import sys
import subprocess
import shutil
import tempfile
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import json
import hashlib


def get_rvc_environment_path() -> Path:
    """Get path to RVC conda environment."""
    home = Path.home()
    return home / "mambaforge" / "envs" / "autovid-rvc"


def check_rvc_environment_exists() -> bool:
    """Check if RVC conda environment exists."""
    env_path = get_rvc_environment_path()
    return env_path.exists()


def create_rvc_environment() -> bool:
    """Create isolated conda environment for RVC."""
    try:
        # Create environment with Python 3.10
        cmd = [
            "conda", "create", "-n", "autovid-rvc", "python=3.10", "-y"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to create RVC environment: {result.stderr}")
            return False
        
        # Install basic requirements
        install_cmd = [
            "conda", "run", "-n", "autovid-rvc", 
            "pip", "install",
            "torch==2.0.0",
            "torchaudio==2.0.1", 
            "numpy==1.23.5",
            "librosa==0.9.1",
            "soundfile",
            "scipy==1.9.3",
            "praat-parselmouth==0.4.2",
            "pyworld==0.3.2",
            "faiss-cpu==1.7.2",
            "torchcrepe",
            "ffmpeg-python",
            "pyyaml"
        ]
        
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to install RVC dependencies: {result.stderr}")
            return False
        
        print("RVC environment created successfully")
        return True
        
    except Exception as e:
        print(f"Error creating RVC environment: {e}")
        return False


def download_pretrained_models() -> bool:
    """Download required pretrained models for RVC."""
    project_root = Path(__file__).parent.parent
    rvc_dir = project_root / "third_party" / "Mangio-RVC-Fork"
    
    # Models to download with their URLs and checksums
    models = {
        "hubert_base.pt": {
            "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
            "sha256": "b9e8c8a52c48b9e14a47a7f2c98f4e57bb0e5e8f5b3e5b3c2f2f9c9e1a9b2c3d"  # placeholder
        },
        "rmvpe.pt": {
            "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt", 
            "sha256": "placeholder_checksum"
        }
    }
    
    try:
        for model_name, model_info in models.items():
            model_path = rvc_dir / model_name
            
            # Skip if already exists
            if model_path.exists():
                print(f"Model {model_name} already exists")
                continue
            
            print(f"Downloading {model_name}...")
            
            # Download with progress
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r{model_name}: {percent:.1f}%", end='', flush=True)
            
            print(f"\n{model_name} downloaded successfully")
        
        return True
        
    except Exception as e:
        print(f"Error downloading pretrained models: {e}")
        return False


def setup_rvc_environment() -> bool:
    """Setup complete RVC environment (create env + download models)."""
    print("Setting up RVC environment...")
    
    # Create environment if it doesn't exist
    if not check_rvc_environment_exists():
        print("Creating RVC conda environment...")
        if not create_rvc_environment():
            return False
    else:
        print("RVC environment already exists")
    
    # Download pretrained models
    print("Checking pretrained models...")
    if not download_pretrained_models():
        return False
    
    print("RVC environment setup complete!")
    return True


def run_rvc_in_environment(
    model_path: str,
    index_path: str,
    input_audio: str,
    output_audio: str,
    rvc_params: Dict,
    working_dir: str
) -> Tuple[bool, str]:
    """
    Run RVC conversion in isolated environment.
    
    Args:
        model_path: Path to RVC model (.pth)
        index_path: Path to index file (.index)
        input_audio: Input audio file path
        output_audio: Output audio file path
        rvc_params: RVC parameters
        working_dir: Working directory for RVC
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        project_root = Path(__file__).parent.parent
        
        # Use our own RVC inference script instead of the third-party batch script
        rvc_script = project_root / "cli" / "rvc_inference.py"
        
        # Build command for our RVC inference script - use absolute paths
        cmd = [
            "conda", "run", "-n", "autovid-rvc",
            "python", str(rvc_script),
            "--model_path", str(Path(model_path).absolute()),
            "--index_path", str(Path(index_path).absolute()),
            "--input", str(Path(input_audio).absolute()),
            "--output", str(Path(output_audio).absolute()),
            "--f0_up_key", str(rvc_params.get("f0_up_key", 0)),
            "--f0_method", rvc_params.get("f0_method", "harvest"),
            "--index_rate", str(rvc_params.get("index_rate", 0.66)),
            "--device", rvc_params.get("device", "cuda:0"),
            "--filter_radius", str(rvc_params.get("filter_radius", 3)),
            "--resample_sr", str(rvc_params.get("resample_sr", 0)),
            "--rms_mix_rate", str(rvc_params.get("rms_mix_rate", 1.0)),
            "--protect", str(rvc_params.get("protect", 0.33)),
            "--crepe_hop_length", str(rvc_params.get("crepe_hop_length", 128))
        ]
        
        # Add --is_half flag if enabled
        if rvc_params.get("is_half", True):
            cmd.append("--is_half")
        
        # Set environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root / "third_party" / "Mangio-RVC-Fork")
        
        # Execute in RVC environment - use RVC directory as working directory
        rvc_dir = project_root / "third_party" / "Mangio-RVC-Fork"
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=rvc_dir,  # Run from RVC directory so relative paths work
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Our script outputs directly to the target file, so just check if it exists
            if Path(output_audio).exists():
                return True, ""
            else:
                return False, "RVC conversion completed but no output file found"
        else:
            return False, f"RVC Error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "RVC conversion timeout (5 minutes)"
    except Exception as e:
        return False, f"RVC execution error: {str(e)}"


def validate_rvc_environment() -> Tuple[bool, str]:
    """Validate RVC environment is ready for use."""
    try:
        # Check environment exists
        if not check_rvc_environment_exists():
            return False, "RVC environment not found. Run setup first."
        
        # Check pretrained models
        project_root = Path(__file__).parent.parent
        rvc_dir = project_root / "third_party" / "Mangio-RVC-Fork"
        
        required_models = ["hubert_base.pt"]  # rmvpe.pt is optional for harvest method
        for model in required_models:
            model_path = rvc_dir / model
            if not model_path.exists():
                return False, f"Required model {model} not found"
        
        # Test basic import in RVC environment
        test_cmd = [
            "conda", "run", "-n", "autovid-rvc",
            "python", "-c", "import torch; import librosa; import soundfile; print('OK')"
        ]
        
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, f"RVC environment validation failed: {result.stderr}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def cleanup_rvc_environment() -> bool:
    """Remove RVC environment (for cleanup/reset)."""
    try:
        cmd = ["conda", "env", "remove", "-n", "autovid-rvc", "-y"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("RVC environment removed successfully")
            return True
        else:
            print(f"Failed to remove RVC environment: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error removing RVC environment: {e}")
        return False


def get_rvc_environment_info() -> Dict:
    """Get information about RVC environment status."""
    project_root = Path(__file__).parent.parent
    rvc_dir = project_root / "third_party" / "Mangio-RVC-Fork"
    
    info = {
        "environment_exists": check_rvc_environment_exists(),
        "environment_path": str(get_rvc_environment_path()),
        "pretrained_models": {},
        "rvc_script_exists": (rvc_dir / "infer_batch_rvc.py").exists()
    }
    
    # Check pretrained models
    models = ["hubert_base.pt", "rmvpe.pt"]
    for model in models:
        model_path = rvc_dir / model
        info["pretrained_models"][model] = {
            "exists": model_path.exists(),
            "size": model_path.stat().st_size if model_path.exists() else 0
        }
    
    return info