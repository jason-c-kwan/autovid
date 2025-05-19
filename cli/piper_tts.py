#!/usr/bin/env python3
"""
Command-line interface for Piper TTS.

Synthesizes text to a WAV audio file using a Piper TTS model specified in
the pipeline configuration and outputs a JSON manifest.
"""

import argparse
import json
import sys
import uuid
import wave # Added
from pathlib import Path
import yaml

try:
    from piper.voice import PiperVoice
    from piper.download import get_voices, ensure_voice_exists # Added
except ImportError:
    print(
        "Error: piper-tts library not found or core components missing. " # Modified error
        "Please install it (e.g., 'pip install piper-tts').",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthesize text to WAV using Piper TTS and output a manifest."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize."
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Output file path for the JSON manifest (default: prints to stdout)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to the pipeline configuration YAML file (default: config/pipeline.yaml)."
    )
    parser.add_argument(
        "--step_id",
        type=str,
        default="tts_run",
        help="Identifier for the TTS step in the config file (default: tts_run)."
    )
    return parser.parse_args()


def get_config_params(config_path: str, step_id: str) -> tuple[str | None, str | None]:
    """
    Load Piper model name and global model directory from the pipeline configuration file.

    Args:
        config_path: Path to the YAML configuration file.
        step_id: The ID of the step in the configuration.

    Returns:
        A tuple (piper_model_name, model_dir_path). Both can be None if not found.
    """
    piper_model_name: str | None = None
    model_dir_path: str | None = None

    try:
        config_file = Path(config_path)
        if not config_file.is_file():
            print(f"Error: Config file not found at {config_path}", file=sys.stderr)
            return None, None

        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            print(f"Error: Invalid or empty config file {config_path}.", file=sys.stderr)
            return None, None

        model_dir_path = config_data.get("model_dir")
        if not model_dir_path:
            print(f"Error: Global 'model_dir' not found in {config_path}", file=sys.stderr)
            # Continue to try and get piper_model_name, but model_dir_path will be None
        else:
            model_dir_path = str(model_dir_path)


        steps_data = config_data.get("steps")
        if not isinstance(steps_data, list):
            print(f"Error: Invalid config format in {config_path}. 'steps' is not a list or is missing.", file=sys.stderr)
            return None, model_dir_path # Return model_dir_path if found, piper_model_name is None

        found_step = False
        for step in steps_data:
            if isinstance(step, dict) and step.get("id") == step_id:
                found_step = True
                parameters = step.get("parameters", {})
                piper_model_name_from_step = parameters.get("piper_model")
                if piper_model_name_from_step:
                    piper_model_name = str(piper_model_name_from_step)
                else:
                    print(f"Error: 'piper_model' not found in parameters for step '{step_id}' in {config_path}", file=sys.stderr)
                break # Found the step, no need to continue loop
        
        if not found_step:
            print(f"Error: Step ID '{step_id}' not found in {config_path}", file=sys.stderr)
            # piper_model_name remains None

        return piper_model_name, model_dir_path

    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}", file=sys.stderr)
        return None, None # If YAML parsing fails, can't trust model_dir_path either
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}", file=sys.stderr)
        return None, None


def synthesize_audio(text_to_synthesize: str, output_wav_path: Path, piper_model_name: str, piper_data_dir: Path) -> bool:
    """
    Synthesize text to a WAV file using PiperVoice, following the provided example.

    Args:
        text_to_synthesize: The text to synthesize.
        output_wav_path: The Path object for the output WAV file.
        piper_model_name: The name of the Piper TTS model (e.g., "en_US-lessac-medium").
        piper_data_dir: The directory for storing/finding Piper models.

    Returns:
        True if synthesis was successful, False otherwise.
    """
    try:
        piper_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Piper-tts setup from example
        data_dirs_str = [str(piper_data_dir)]
        download_dir_str = str(piper_data_dir)

        print(f"Piper data directories: {data_dirs_str}", file=sys.stderr)
        print(f"Piper download directory: {download_dir_str}", file=sys.stderr)
        print(f"Attempting to use Piper model: {piper_model_name}", file=sys.stderr)

        voices_info = get_voices(download_dir_str, update_voices=True) # Try to update voices list

        ensure_voice_exists(
            name=piper_model_name,
            data_dirs=data_dirs_str,
            download_dir=download_dir_str,
            voices_info=voices_info
        )

        model_onnx_path = piper_data_dir / f"{piper_model_name}.onnx"
        model_json_path = piper_data_dir / f"{piper_model_name}.onnx.json" # Config file

        if not model_onnx_path.is_file():
            print(f"Error: Piper model ONNX file not found at {model_onnx_path} after ensure_voice_exists.", file=sys.stderr)
            print(f"Please check if the model '{piper_model_name}' was downloaded correctly to '{piper_data_dir}'.", file=sys.stderr)
            return False
        if not model_json_path.is_file():
            print(f"Error: Piper model JSON config file not found at {model_json_path} after ensure_voice_exists.", file=sys.stderr)
            return False

        print(f"Loading Piper model from: {model_onnx_path}", file=sys.stderr)
        voice = PiperVoice.load(str(model_onnx_path), config_path=str(model_json_path)) # Corrected load
        
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            
            if voice.config and hasattr(voice.config, 'sample_rate'):
                wav_file.setframerate(voice.config.sample_rate)
            else:
                # This case should ideally not happen if model loaded correctly
                print("Warning: Could not determine sample rate from voice config. Defaulting to 22050 Hz.", file=sys.stderr)
                wav_file.setframerate(22050) 
            
            voice.synthesize(text_to_synthesize, wav_file)
        
        print(f"Successfully synthesized audio to {output_wav_path}", file=sys.stderr)
        return True

    except FileNotFoundError as e:
        print(f"Error: A required file was not found during Piper TTS setup/synthesis: {e}", file=sys.stderr)
        return False
    except RuntimeError as e: # Piper can raise RuntimeErrors for various internal issues
        print(f"Error: Piper TTS runtime error: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during Piper TTS synthesis: {e}", file=sys.stderr)
        return False


def main():
    """Main function to orchestrate TTS and manifest generation."""
    args = parse_args()

    status = "failure"
    wav_file_path_str = None
    
    piper_model_name, model_dir_str = get_config_params(args.config, args.step_id)

    if piper_model_name is None or model_dir_str is None:
        # Error messages already printed by get_config_params
        if piper_model_name is None:
            print("Main: Piper model name not resolved from config.", file=sys.stderr)
        if model_dir_str is None:
            print("Main: Model directory not resolved from config.", file=sys.stderr)
        # status remains "failure"
    elif not args.text or not args.text.strip():
        print("Error: Input text cannot be empty.", file=sys.stderr)
        # status remains "failure"
    else:
        # Determine WAV output path
        wav_filename = f"{uuid.uuid4().hex}.wav"
        if args.out:
            # Place WAV in the same directory as the manifest
            wav_output_dir = Path(args.out).parent.resolve() # Resolve to make absolute
        else:
            # Default to a subdirectory in the current working directory
            wav_output_dir = (Path.cwd() / "tts_audio_output").resolve()
        
        actual_wav_path = wav_output_dir / wav_filename
        
        try:
            # Ensure the directory for the WAV file exists
            # This is also done in synthesize_audio, but good to have here too for manifest dir
            actual_wav_path.parent.mkdir(parents=True, exist_ok=True)
            
            piper_data_path = Path(model_dir_str).resolve() # Resolve to make absolute

            if synthesize_audio(args.text, actual_wav_path, piper_model_name, piper_data_path):
                status = "success"
                wav_file_path_str = str(actual_wav_path.resolve()) # Ensure it's absolute
            # If synthesize_audio fails, error messages are printed within it

        except Exception as e:
            print(f"An unexpected error occurred in main processing: {e}", file=sys.stderr)
            # status remains "failure"

    # Build manifest
    manifest = {
        "text": args.text if args.text else "", # Ensure text is always present
        "wav_path": wav_file_path_str,
        "status": status,
    }
    manifest_json = json.dumps(manifest, indent=2)

    # Output manifest
    if args.out:
        try:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True) # Ensure manifest dir exists
            with open(out_path, "w") as f:
                f.write(manifest_json)
            print(f"Manifest written to {out_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing manifest to {args.out}: {e}", file=sys.stderr)
            # If manifest writing fails, the overall status might still be success for audio
            # but the script's primary output (manifest) failed.
            # Consider if this should force a non-zero exit. For now, it doesn't change 'status'.
    else:
        print(manifest_json) # Print to stdout

    # Exit code
    if status == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
