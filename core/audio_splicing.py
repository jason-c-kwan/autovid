"""
Audio splicing functions for concatenating audio chunks with crossfade.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import yaml


def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return audio data and sample rate.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        audio_data, sample_rate = sf.read(file_path)
        return audio_data, sample_rate
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {str(e)}")


def apply_crossfade(
    audio1: np.ndarray,
    audio2: np.ndarray,
    crossfade_samples: int
) -> np.ndarray:
    """
    Apply crossfade between two audio segments.
    
    Args:
        audio1: First audio segment
        audio2: Second audio segment  
        crossfade_samples: Number of samples for crossfade
        
    Returns:
        Crossfaded audio segment
    """
    if len(audio1) == 0 or len(audio2) == 0:
        return np.concatenate([audio1, audio2])
    
    # Ensure crossfade doesn't exceed audio lengths
    crossfade_samples = min(crossfade_samples, len(audio1), len(audio2))
    
    if crossfade_samples == 0:
        return np.concatenate([audio1, audio2])
    
    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)
    
    # Apply fades
    audio1_faded = audio1.copy()
    audio2_faded = audio2.copy()
    
    # Handle mono and stereo audio
    if len(audio1.shape) == 1:  # Mono
        audio1_faded[-crossfade_samples:] *= fade_out
        audio2_faded[:crossfade_samples] *= fade_in
    else:  # Stereo
        audio1_faded[-crossfade_samples:, :] *= fade_out[:, np.newaxis]
        audio2_faded[:crossfade_samples, :] *= fade_in[:, np.newaxis]
    
    # Combine crossfaded regions
    crossfaded_region = audio1_faded[-crossfade_samples:] + audio2_faded[:crossfade_samples]
    
    # Combine full audio
    result = np.concatenate([
        audio1_faded[:-crossfade_samples],
        crossfaded_region,
        audio2_faded[crossfade_samples:]
    ])
    
    return result


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0,
    max_db: float = -1.0
) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio data
        target_db: Target dB level
        max_db: Maximum dB level (prevents clipping)
        
    Returns:
        Normalized audio data
    """
    if len(audio) == 0:
        return audio
    
    # Calculate RMS
    rms = np.sqrt(np.mean(audio**2))
    
    if rms == 0:
        return audio
    
    # Convert to dB
    current_db = 20 * np.log10(rms)
    
    # Calculate gain needed
    gain_db = target_db - current_db
    
    # Limit gain to prevent clipping
    max_gain_db = max_db - current_db
    gain_db = min(gain_db, max_gain_db)
    
    # Apply gain
    gain_linear = 10**(gain_db / 20)
    normalized_audio = audio * gain_linear
    
    return normalized_audio


def concatenate_audio_chunks(
    chunk_paths: List[str],
    crossfade_duration: float = 0.1,
    normalize: bool = True,
    target_db: float = -20.0
) -> Tuple[np.ndarray, int]:
    """
    Concatenate audio chunks with crossfade.
    
    Args:
        chunk_paths: List of audio file paths
        crossfade_duration: Crossfade duration in seconds
        normalize: Whether to normalize audio levels
        target_db: Target dB level for normalization
        
    Returns:
        Tuple of (concatenated_audio, sample_rate)
    """
    if not chunk_paths:
        raise ValueError("No audio chunks provided")
    
    # Load first chunk to get sample rate
    combined_audio, sample_rate = load_audio_file(chunk_paths[0])
    
    # Normalize first chunk if requested
    if normalize:
        combined_audio = normalize_audio(combined_audio, target_db)
    
    # Calculate crossfade samples
    crossfade_samples = int(crossfade_duration * sample_rate)
    
    # Process remaining chunks
    for chunk_path in chunk_paths[1:]:
        try:
            next_audio, next_sr = load_audio_file(chunk_path)
            
            # Verify sample rates match
            if next_sr != sample_rate:
                raise ValueError(f"Sample rate mismatch: {next_sr} vs {sample_rate}")
            
            # Normalize if requested
            if normalize:
                next_audio = normalize_audio(next_audio, target_db)
            
            # Apply crossfade
            combined_audio = apply_crossfade(combined_audio, next_audio, crossfade_samples)
            
        except Exception as e:
            print(f"Warning: Failed to process chunk {chunk_path}: {str(e)}")
            continue
    
    return combined_audio, sample_rate


def calculate_timing_metadata(
    chunks: List[Dict[str, Any]],
    crossfade_duration: float = 0.1
) -> Dict[str, Any]:
    """
    Calculate timing metadata for spliced audio.
    
    Args:
        chunks: List of audio chunk metadata
        crossfade_duration: Crossfade duration in seconds
        
    Returns:
        Dictionary with timing information
    """
    timing_data = {
        "chunks": [],
        "total_duration": 0.0,
        "crossfade_duration": crossfade_duration
    }
    
    current_time = 0.0
    
    for i, chunk in enumerate(chunks):
        chunk_duration = chunk.get("duration", 0.0)
        
        # Account for crossfade overlap (except first chunk)
        if i > 0:
            current_time -= crossfade_duration
        
        chunk_timing = {
            "id": chunk.get("id"),
            "start_time": current_time,
            "end_time": current_time + chunk_duration,
            "duration": chunk_duration,
            "text": chunk.get("text", "")
        }
        
        timing_data["chunks"].append(chunk_timing)
        current_time += chunk_duration
    
    timing_data["total_duration"] = current_time
    
    return timing_data


def save_audio_file(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str,
    format: str = "WAV"
) -> None:
    """
    Save audio data to file.
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        output_path: Output file path
        format: Audio format (WAV, FLAC, etc.)
    """
    try:
        sf.write(output_path, audio, sample_rate, format=format)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio file {output_path}: {str(e)}")


def validate_audio_chunks(chunk_paths: List[str]) -> Tuple[bool, str]:
    """
    Validate audio chunks before splicing.
    
    Args:
        chunk_paths: List of audio file paths
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not chunk_paths:
        return False, "No audio chunks provided"
    
    # Check if all files exist
    for chunk_path in chunk_paths:
        if not Path(chunk_path).exists():
            return False, f"Audio chunk not found: {chunk_path}"
    
    # Check if all files have same sample rate
    sample_rates = []
    for chunk_path in chunk_paths:
        try:
            _, sr = load_audio_file(chunk_path)
            sample_rates.append(sr)
        except Exception as e:
            return False, f"Failed to load audio chunk {chunk_path}: {str(e)}"
    
    if len(set(sample_rates)) > 1:
        return False, f"Sample rate mismatch: {sample_rates}"
    
    return True, ""


def get_audio_splicing_config(config_path: str = "config/pipeline.yaml") -> Dict[str, Any]:
    """
    Extract audio splicing configuration from pipeline config.
    
    Args:
        config_path: Path to pipeline configuration file
        
    Returns:
        Dictionary of audio splicing configuration
    """
    project_root = Path(__file__).parent.parent
    config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    splicing_config = config.get("audio_splicing", {})
    
    # Set defaults if not specified
    defaults = {
        "crossfade_duration": 0.1,
        "target_db": -20.0,
        "normalize": True
    }
    
    # Merge defaults with config
    for key, value in defaults.items():
        if key not in splicing_config:
            splicing_config[key] = value
    
    return splicing_config