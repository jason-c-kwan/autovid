#!/usr/bin/env python3
"""
Centralized configuration loading utility for AutoVid pipeline.

This module provides functions to load and merge configuration files,
handle environment variable substitution, parameter validation, and caching.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from functools import lru_cache

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


@lru_cache(maxsize=16)
def load_pipeline_config(config_path: str = "config/pipeline.yaml") -> Dict[str, Any]:
    """
    Load the main pipeline configuration with caching.
    
    Args:
        config_path: Path to pipeline configuration file
        
    Returns:
        Dictionary containing pipeline configuration
        
    Raises:
        ConfigurationError: If configuration cannot be loaded or is invalid
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigurationError(f"Pipeline configuration not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Environment variable substitution
        config = _substitute_env_vars(config)
        
        logger.debug(f"Loaded pipeline configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration {config_path}: {e}")


@lru_cache(maxsize=16)
def load_qc_config(config_path: str = "config/qc_config.yaml") -> Dict[str, Any]:
    """
    Load quality control configuration with caching.
    
    Args:
        config_path: Path to QC configuration file
        
    Returns:
        Dictionary containing QC configuration
        
    Raises:
        ConfigurationError: If configuration cannot be loaded
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"QC configuration not found: {config_path}, using defaults")
            return _get_default_qc_config()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config = _substitute_env_vars(config)
        logger.debug(f"Loaded QC configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load QC configuration {config_path}: {e}")


@lru_cache(maxsize=16)
def load_scene_detection_config(config_path: str = "config/scene_detection.yaml") -> Dict[str, Any]:
    """
    Load scene detection configuration with caching.
    
    Args:
        config_path: Path to scene detection configuration file
        
    Returns:
        Dictionary containing scene detection configuration
        
    Raises:
        ConfigurationError: If configuration cannot be loaded
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Scene detection configuration not found: {config_path}, using defaults")
            return _get_default_scene_detection_config()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config = _substitute_env_vars(config)
        logger.debug(f"Loaded scene detection configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load scene detection configuration {config_path}: {e}")


def get_step_config(step_id: str, pipeline_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract configuration for a specific pipeline step.
    
    Args:
        step_id: Pipeline step identifier (e.g., 'qc_pronounce', 'analyze_video')
        pipeline_config: Pre-loaded pipeline configuration (optional)
        
    Returns:
        Dictionary containing step-specific configuration parameters
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()
    
    # Find step configuration in pipeline
    steps = pipeline_config.get("steps", [])
    for step in steps:
        if step.get("id") == step_id:
            return step.get("parameters", {})
    
    logger.warning(f"Step '{step_id}' not found in pipeline configuration")
    return {}


def get_video_analysis_config(
    step_config: Optional[Dict[str, Any]] = None,
    scene_detection_config: Optional[Dict[str, Any]] = None,
    pipeline_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get comprehensive video analysis configuration by merging step config,
    scene detection config, and pipeline defaults.
    
    Args:
        step_config: Step-specific parameters from pipeline
        scene_detection_config: Dedicated scene detection configuration
        pipeline_config: Full pipeline configuration
        
    Returns:
        Merged video analysis configuration
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()
    
    if step_config is None:
        step_config = get_step_config("analyze_video", pipeline_config)
    
    if scene_detection_config is None:
        scene_detection_config = load_scene_detection_config()
    
    # Start with pipeline video_analysis section
    base_config = pipeline_config.get("video_analysis", {}).copy()
    
    # Override with step parameters
    base_config.update(step_config)
    
    # Add scene detection algorithms configuration
    if "algorithms" in scene_detection_config:
        base_config["algorithms"] = scene_detection_config["algorithms"]
    
    # Add keynote optimizations
    if "keynote_optimizations" in scene_detection_config:
        base_config["keynote_optimizations"] = scene_detection_config["keynote_optimizations"]
    
    # Add ensemble configuration
    if "ensemble" in scene_detection_config:
        base_config["ensemble"] = scene_detection_config["ensemble"]
    
    # Add validation configuration
    if "validation" in scene_detection_config:
        base_config["validation"] = scene_detection_config["validation"]
    
    return base_config


def get_qc_config(
    step_config: Optional[Dict[str, Any]] = None,
    qc_config: Optional[Dict[str, Any]] = None,
    pipeline_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get comprehensive QC configuration by merging step config,
    dedicated QC config, and pipeline defaults.
    
    Args:
        step_config: Step-specific parameters from pipeline
        qc_config: Dedicated QC configuration
        pipeline_config: Full pipeline configuration
        
    Returns:
        Merged QC configuration
    """
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()
    
    if step_config is None:
        step_config = get_step_config("qc_pronounce", pipeline_config)
    
    if qc_config is None:
        qc_config = load_qc_config()
    
    # Start with step parameters (highest priority)
    merged_config = step_config.copy()
    
    # Add quality thresholds from dedicated config
    if "quality_thresholds" in qc_config:
        thresholds = qc_config["quality_thresholds"]
        merged_config.setdefault("mos_threshold", thresholds.get("mos", {}).get("minimum", 3.5))
        merged_config.setdefault("wer_threshold", thresholds.get("wer", {}).get("maximum", 0.10))
    
    # Add transcription settings
    if "transcription" in qc_config:
        transcription = qc_config["transcription"]
        merged_config.setdefault("whisper_model", transcription.get("whisper", {}).get("model", "large-v3"))
        merged_config.setdefault("transcription_timeout", 30)
    
    # Add retry strategies
    if "retry_strategies" in qc_config:
        retry = qc_config["retry_strategies"]
        merged_config.setdefault("max_attempts", retry.get("max_attempts", 3))
    
    # Add performance settings
    if "performance" in qc_config:
        perf = qc_config["performance"]
        merged_config.setdefault("parallel_processing", perf.get("parallel_processing", True))
        merged_config.setdefault("max_workers", perf.get("max_workers", 4))
    
    return merged_config


def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively substitute environment variables in configuration values.
    
    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.
    """
    if isinstance(config, dict):
        return {key: _substitute_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Simple substitution for ${VAR} and ${VAR:-default} patterns
        import re
        
        def replace_env_var(match):
            var_expr = match.group(1)
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(var_expr, match.group(0))  # Return original if not found
        
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, config)
    else:
        return config


def _get_default_qc_config() -> Dict[str, Any]:
    """Return default QC configuration when dedicated config file is not found."""
    return {
        "quality_thresholds": {
            "mos": {"minimum": 3.5, "target": 4.0, "excellent": 4.5},
            "wer": {"maximum": 0.10, "target": 0.05, "excellent": 0.02},
            "duration": {"min_chunk": 0.3, "max_chunk": 30.0, "silence_threshold": -40},
            "audio_quality": {"max_clipping": 0.01, "min_rms": -50, "max_rms": -6}
        },
        "transcription": {
            "whisper": {"model": "large-v3", "language": "auto", "device": "cuda", "batch_size": 16},
            "alignment": {"model": "WAV2VEC2_ASR_LARGE_LV60K_960H", "return_char_alignments": False}
        },
        "retry_strategies": {
            "max_attempts": 3,
            "strategies": [
                {"name": "parameter_adjustment", "enabled": True, "priority": 1},
                {"name": "text_preprocessing", "enabled": True, "priority": 2},
                {"name": "engine_fallback", "enabled": True, "priority": 3}
            ]
        },
        "performance": {
            "parallel_processing": True,
            "max_workers": 4,
            "gpu_acceleration": True
        }
    }


def _get_default_scene_detection_config() -> Dict[str, Any]:
    """Return default scene detection configuration when dedicated config file is not found."""
    return {
        "algorithms": {
            "ffmpeg_scene": {
                "enabled": True,
                "threshold": 0.1,
                "min_scene_len": 0.5,
                "weight": 0.3
            },
            "static_detection": {
                "enabled": True,
                "static_threshold": 0.95,
                "min_static_duration": 0.8,
                "max_static_duration": 2.0,
                "weight": 0.4
            },
            "content_analysis": {
                "enabled": True,
                "histogram_bins": 256,
                "comparison_method": "correlation",
                "threshold": 0.15,
                "weight": 0.3
            }
        },
        "keynote_optimizations": {
            "delay_compensation": 1.0,
            "presentation_mode": True,
            "slide_characteristics": {
                "typical_duration": [5, 30],
                "min_slide_duration": 2.0,
                "max_slide_duration": 120.0
            },
            "animation_detection": {
                "enabled": True,
                "animation_threshold": 0.3,
                "ignore_minor_animations": True,
                "animation_duration_limit": 3.0
            }
        },
        "ensemble": {
            "decision_method": "weighted_average",
            "confidence_threshold": 0.6,
            "temporal_consistency": True
        },
        "validation": {
            "transcript_validation": {
                "enabled": True,
                "cue_token": "[transition]",
                "tolerance_range": [0.7, 1.3]
            },
            "auto_adjustment": {
                "enabled": True,
                "max_iterations": 5,
                "adjustment_step": 0.02,
                "convergence_tolerance": 0.1
            }
        },
        "performance": {
            "frame_sampling": 2,
            "parallel_processing": True,
            "max_workers": 4
        }
    }


def validate_configuration(config: Dict[str, Any], config_type: str = "pipeline") -> bool:
    """
    Validate configuration against expected structure and ranges.
    
    Args:
        config: Configuration dictionary to validate
        config_type: Type of configuration ('pipeline', 'qc', 'scene_detection')
        
    Returns:
        True if configuration is valid
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Import validation here to avoid circular imports
    try:
        from core.config_validation import ConfigurationValidator
        
        validator = ConfigurationValidator()
        is_valid, errors = validator.validate_config(config, config_type)
        
        if not is_valid:
            error_msg = f"Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ConfigurationError(error_msg)
        
        return True
        
    except ImportError:
        logger.warning("Configuration validation not available (missing jsonschema)")
        return True


def clear_config_cache():
    """Clear the configuration cache to force reloading on next access."""
    load_pipeline_config.cache_clear()
    load_qc_config.cache_clear()
    load_scene_detection_config.cache_clear()
    logger.debug("Configuration cache cleared")


# Convenience functions for common use cases
def get_tts_config(pipeline_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get TTS configuration from pipeline."""
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()
    return get_step_config("tts_run", pipeline_config)


def get_rvc_config(pipeline_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get RVC configuration from pipeline."""
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()
    return pipeline_config.get("rvc", {})


def get_sync_config(pipeline_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get slide synchronization configuration from pipeline."""
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()
    
    step_config = get_step_config("sync_slides", pipeline_config)
    base_config = pipeline_config.get("slide_sync", {}).copy()
    base_config.update(step_config)
    
    return base_config