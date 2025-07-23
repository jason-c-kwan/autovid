#!/usr/bin/env python3
"""
Configuration validation system for AutoVid pipeline.

This module provides JSON Schema-based validation for all configuration files,
ensuring that parameters are valid before pipeline execution begins.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from jsonschema import Draft7Validator, ValidationError, validators

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Configuration validator with schema-based validation and helpful error reporting."""
    
    def __init__(self, schema_dir: Optional[str] = None):
        """
        Initialize the configuration validator.
        
        Args:
            schema_dir: Directory containing JSON schema files. Defaults to config/schemas/
        """
        if schema_dir is None:
            schema_dir = Path(__file__).parent.parent / "config" / "schemas"
        
        self.schema_dir = Path(schema_dir)
        self.schemas = {}
        self._load_schemas()
    
    def _load_schemas(self) -> None:
        """Load all JSON schema files from the schema directory."""
        if not self.schema_dir.exists():
            logger.warning(f"Schema directory not found: {self.schema_dir}")
            return
        
        for schema_file in self.schema_dir.glob("*.json"):
            schema_name = schema_file.stem
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                self.schemas[schema_name] = schema
                logger.debug(f"Loaded schema: {schema_name}")
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")
    
    def validate_config(
        self, 
        config: Dict[str, Any], 
        schema_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate a configuration against a named schema.
        
        Args:
            config: Configuration dictionary to validate
            schema_name: Name of the schema to validate against
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if schema_name not in self.schemas:
            return False, [f"Schema '{schema_name}' not found"]
        
        schema = self.schemas[schema_name]
        validator = Draft7Validator(schema)
        
        errors = []
        for error in validator.iter_errors(config):
            error_msg = self._format_validation_error(error)
            errors.append(error_msg)
        
        return len(errors) == 0, errors
    
    def _format_validation_error(self, error: ValidationError) -> str:
        """
        Format a validation error into a human-readable message.
        
        Args:
            error: ValidationError from jsonschema
            
        Returns:
            Formatted error message
        """
        path = " -> ".join(str(p) for p in error.absolute_path)
        if path:
            return f"Error at '{path}': {error.message}"
        else:
            return f"Error: {error.message}"
    
    def validate_file(self, config_path: str, schema_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a configuration file against a named schema.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            schema_name: Name of the schema to validate against
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            config = self._load_config_file(config_path)
        except Exception as e:
            return False, [f"Failed to load config file: {e}"]
        
        return self.validate_config(config, schema_name)
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load a configuration file (YAML or JSON).
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {config_path.suffix}")


def validate_pipeline_config(config_path: str = "config/pipeline.yaml") -> bool:
    """
    Validate the main pipeline configuration file.
    
    Args:
        config_path: Path to pipeline configuration file
        
    Returns:
        True if valid, False otherwise
    """
    validator = ConfigurationValidator()
    is_valid, errors = validator.validate_file(config_path, "pipeline")
    
    if not is_valid:
        logger.error("Pipeline configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("Pipeline configuration validation passed")
    return True


def validate_qc_config(config_path: str = "config/qc_config.yaml") -> bool:
    """
    Validate the QC configuration file.
    
    Args:
        config_path: Path to QC configuration file
        
    Returns:
        True if valid, False otherwise
    """
    validator = ConfigurationValidator()
    is_valid, errors = validator.validate_file(config_path, "qc_config")
    
    if not is_valid:
        logger.error("QC configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("QC configuration validation passed")
    return True


def validate_scene_detection_config(config_path: str = "config/scene_detection.yaml") -> bool:
    """
    Validate the scene detection configuration file.
    
    Args:
        config_path: Path to scene detection configuration file
        
    Returns:
        True if valid, False otherwise
    """
    validator = ConfigurationValidator()
    is_valid, errors = validator.validate_file(config_path, "scene_detection")
    
    if not is_valid:
        logger.error("Scene detection configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("Scene detection configuration validation passed")
    return True


def validate_all_configs(config_dir: str = "config") -> bool:
    """
    Validate all configuration files in the specified directory.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        True if all configurations are valid, False otherwise
    """
    config_dir = Path(config_dir)
    all_valid = True
    
    # Validate main pipeline config
    pipeline_config = config_dir / "pipeline.yaml"
    if pipeline_config.exists():
        if not validate_pipeline_config(str(pipeline_config)):
            all_valid = False
    else:
        logger.warning(f"Pipeline config not found: {pipeline_config}")
    
    # Validate QC config
    qc_config = config_dir / "qc_config.yaml"
    if qc_config.exists():
        if not validate_qc_config(str(qc_config)):
            all_valid = False
    else:
        logger.info(f"QC config not found (optional): {qc_config}")
    
    # Validate scene detection config
    scene_config = config_dir / "scene_detection.yaml"
    if scene_config.exists():
        if not validate_scene_detection_config(str(scene_config)):
            all_valid = False
    else:
        logger.info(f"Scene detection config not found (optional): {scene_config}")
    
    return all_valid


def validate_parameter_ranges(config: Dict[str, Any]) -> List[str]:
    """
    Validate parameter ranges and logical constraints.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # TTS parameter validation
    if 'steps' in config:
        for step in config['steps']:
            if step.get('id') == 'tts_run':
                params = step.get('parameters', {})
                
                # Validate token limits
                max_tokens = params.get('max_tokens_per_chunk', 512)
                if max_tokens > 768:
                    errors.append("max_tokens_per_chunk should not exceed 768 to prevent cutoffs")
                
                # Validate overlap tokens
                overlap_tokens = params.get('overlap_tokens', 50)
                if overlap_tokens >= max_tokens:
                    errors.append("overlap_tokens must be less than max_tokens_per_chunk")
    
    # Video analysis parameter validation
    if 'video_analysis' in config:
        va_config = config['video_analysis']
        
        # Validate thresholds
        scene_threshold = va_config.get('scene_threshold', 0.1)
        if not 0.0 <= scene_threshold <= 1.0:
            errors.append("video_analysis.scene_threshold must be between 0.0 and 1.0")
        
        # Validate algorithm weights
        if 'algorithms' in va_config:
            total_weight = sum(
                alg.get('weight', 0) 
                for alg in va_config['algorithms'].values() 
                if alg.get('enabled', False)
            )
            if abs(total_weight - 1.0) > 0.1:
                errors.append(f"Algorithm weights should sum to ~1.0, got {total_weight}")
    
    # QC parameter validation
    for step in config.get('steps', []):
        if step.get('id') == 'qc_pronounce':
            params = step.get('parameters', {})
            
            # Validate MOS threshold
            mos_threshold = params.get('mos_threshold', 3.5)
            if not 1.0 <= mos_threshold <= 5.0:
                errors.append("mos_threshold must be between 1.0 and 5.0")
            
            # Validate WER threshold
            wer_threshold = params.get('wer_threshold', 0.10)
            if not 0.0 <= wer_threshold <= 1.0:
                errors.append("wer_threshold must be between 0.0 and 1.0")
    
    return errors


def check_environment_dependencies(config: Dict[str, Any]) -> List[str]:
    """
    Check if required environment dependencies are available.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of missing dependency error messages
    """
    errors = []
    
    # Check for required directories
    required_dirs = ['data_dir', 'model_dir', 'workspace_root']
    for dir_key in required_dirs:
        if dir_key in config:
            dir_path = Path(config[dir_key])
            if not dir_path.exists():
                errors.append(f"Required directory not found: {dir_path} ({dir_key})")
    
    # Check for GPU availability if CUDA is specified
    if 'rvc' in config and 'cuda' in str(config['rvc'].get('device', '')):
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA device specified but PyTorch CUDA not available")
        except ImportError:
            errors.append("PyTorch not available for CUDA device check")
    
    # Check for required model files
    if 'rvc' in config:
        rvc_config = config['rvc']
        model_path = rvc_config.get('model_path')
        index_path = rvc_config.get('index_path')
        
        if model_path and not Path(model_path).exists():
            errors.append(f"RVC model file not found: {model_path}")
        
        if index_path and not Path(index_path).exists():
            errors.append(f"RVC index file not found: {index_path}")
    
    return errors


if __name__ == "__main__":
    """Command-line interface for configuration validation."""
    if len(sys.argv) < 2:
        print("Usage: python config_validation.py <config_file> [schema_name]")
        print("       python config_validation.py --all")
        sys.exit(1)
    
    if sys.argv[1] == "--all":
        success = validate_all_configs()
        sys.exit(0 if success else 1)
    
    config_file = sys.argv[1]
    schema_name = sys.argv[2] if len(sys.argv) > 2 else "pipeline"
    
    validator = ConfigurationValidator()
    is_valid, errors = validator.validate_file(config_file, schema_name)
    
    if is_valid:
        print(f"✓ Configuration valid: {config_file}")
        sys.exit(0)
    else:
        print(f"✗ Configuration invalid: {config_file}")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)