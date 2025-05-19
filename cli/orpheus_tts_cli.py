#!/usr/bin/env python3
"""
Command-line interface for Orpheus TTS.

Synthesizes text to a WAV audio file using an Orpheus TTS model specified in
the pipeline configuration and outputs a JSON manifest.
"""

import argparse
import json
import logging
import sys
import time
import uuid
import wave
from multiprocessing import freeze_support
from pathlib import Path

import yaml

try:
    from orpheus_tts import OrpheusModel
except ImportError:
    logging.error(
        "Error: Orpheus TTS library (orpheus_tts) not found. "
        "Please install it (e.g., 'pip install orpheus-tts')."
    )
    sys.exit(1)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthesize text to WAV using Orpheus TTS and output a manifest."
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Text to synthesize."
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Output file path for the JSON manifest (default: prints to stdout).",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="dan",
        help="Voice to use for synthesis (default: dan). "
             "This can be overridden by pipeline.yaml.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Temperature for synthesis (default: 0.4). "
             "This can be overridden by pipeline.yaml.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to the pipeline configuration YAML file (default: config/pipeline.yaml).",
    )
    parser.add_argument(
        "--step_id",
        type=str,
        default="tts_run",
        help="Identifier for the TTS step in the config file (default: tts_run).",
    )
    return parser.parse_args()


def get_config_params(
    config_path: str, step_id: str
) -> tuple[str | None, str | None, float | None]:
    """
    Load Orpheus model name, voice, and temperature from the pipeline configuration.

    Args:
        config_path: Path to the YAML configuration file.
        step_id: The ID of the step in the configuration.

    Returns:
        A tuple (orpheus_model_name, orpheus_voice, orpheus_temperature).
        Values can be None if not found.
    """
    orpheus_model_name: str | None = None
    orpheus_voice: str | None = None
    orpheus_temperature: float | None = None

    try:
        config_file = Path(config_path)
        if not config_file.is_file():
            logging.error(f"Config file not found at {config_path}")
            return None, None, None

        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            logging.error(f"Invalid or empty config file {config_path}.")
            return None, None, None

        steps_data = config_data.get("steps")
        if not isinstance(steps_data, list):
            logging.error(
                f"Invalid config format in {config_path}. 'steps' is not a list or is missing."
            )
            return None, None, None

        found_step = False
        for step in steps_data:
            if isinstance(step, dict) and step.get("id") == step_id:
                found_step = True
                parameters = step.get("parameters", {})
                orpheus_model_name = parameters.get("orpheus_model")
                orpheus_voice = parameters.get("orpheus_voice")
                orpheus_temperature = parameters.get("orpheus_temperature")

                if orpheus_model_name and not isinstance(orpheus_model_name, str):
                    logging.warning(f"'orpheus_model' in {config_path} for step '{step_id}' is not a string. Ignoring.")
                    orpheus_model_name = None
                if orpheus_voice and not isinstance(orpheus_voice, str):
                    logging.warning(f"'orpheus_voice' in {config_path} for step '{step_id}' is not a string. Ignoring.")
                    orpheus_voice = None
                if orpheus_temperature is not None:
                    try:
                        orpheus_temperature = float(orpheus_temperature)
                    except ValueError:
                        logging.warning(
                            f"'orpheus_temperature' in {config_path} for step '{step_id}' is not a valid float. Ignoring."
                        )
                        orpheus_temperature = None
                break

        if not found_step:
            logging.error(f"Step ID '{step_id}' not found in {config_path}")

        return orpheus_model_name, orpheus_voice, orpheus_temperature

    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        return None, None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading config: {e}")
        return None, None, None


def synthesize_audio(
    text_to_synthesize: str,
    output_wav_path: Path,
    orpheus_model_name: str,
    voice: str,
    temperature: float,
) -> tuple[bool, float]:
    """
    Synthesize text to a WAV file using OrpheusModel.

    Args:
        text_to_synthesize: The text to synthesize.
        output_wav_path: The Path object for the output WAV file.
        orpheus_model_name: The name of the Orpheus TTS model.
        voice: The voice to use for synthesis.
        temperature: The temperature for synthesis.

    Returns:
        A tuple (success_status, duration_in_seconds).
    """
    try:
        logging.info(
            f"Initializing OrpheusModel with model: {orpheus_model_name}"
        )
        model = OrpheusModel(model_name=orpheus_model_name, max_model_len=2048)

        logging.info(
            f"Synthesizing speech for text: '{text_to_synthesize[:50]}...' "
            f"with voice: {voice}, temperature: {temperature}"
        )
        start_time = time.monotonic()
        # Orpheus generate_speech returns a generator of audio chunks
        syn_tokens_generator = model.generate_speech(
            prompt=text_to_synthesize, voice=voice, temperature=temperature
        )

        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        total_frames = 0
        duration = 0.0
        sample_rate = 24000  # Orpheus default sample rate
        sample_width = 2  # 16-bit
        num_channels = 1  # Mono

        with wave.open(str(output_wav_path), "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)

            for chunk in syn_tokens_generator:
                if chunk:  # Ensure chunk is not empty
                    # Number of frames in the chunk
                    # Each frame = sample_width * num_channels bytes
                    frames_in_chunk = len(chunk) // (sample_width * num_channels)
                    total_frames += frames_in_chunk
                    wf.writeframes(chunk)

            if total_frames > 0:
                duration = total_frames / sample_rate
            else: # Handle case where no audio data was generated
                logging.warning("No audio data generated by Orpheus TTS.")


        processing_time = time.monotonic() - start_time
        logging.info(
            f"Audio synthesis complete. Output: {output_wav_path}, "
            f"Duration: {duration:.2f}s, Processing time: {processing_time:.2f}s"
        )
        return True, duration

    except FileNotFoundError as e: # Should not happen with Orpheus as it downloads models
        logging.error(f"A required file was not found: {e}")
        return False, 0.0
    except RuntimeError as e:
        logging.error(f"Orpheus TTS runtime error: {e}")
        return False, 0.0
    except Exception as e:
        logging.error(f"An unexpected error occurred during Orpheus TTS synthesis: {e}")
        return False, 0.0


def main():
    """Main function to orchestrate TTS and manifest generation."""
    args = parse_args()

    status = "failure"
    wav_file_path_str: str | None = None
    duration: float = 0.0
    exit_code = 1

    # --- Parameter Resolution ---
    # 1. Get argparse values (CLI or argparse defaults)
    # 2. Get config values
    # 3. Determine final values based on precedence: CLI > Config > Argparse Default

    cfg_orpheus_model, cfg_voice, cfg_temp = get_config_params(
        args.config, args.step_id
    )

    # Final model name (must come from config or error)
    final_model_name = cfg_orpheus_model
    if not final_model_name:
        logging.error(
            f"Orpheus model name ('orpheus_model') not found in config '{args.config}' "
            f"for step_id '{args.step_id}'. Cannot proceed."
        )
        # Manifest will be written with failure status
    elif not args.text or not args.text.strip():
        logging.error("Input text (--text) cannot be empty.")
        # Manifest will be written with failure status
    else:
        # Determine final voice
        # Argparse default for --voice is "dan"
        if args.voice != "dan":  # User explicitly set --voice via CLI
            final_voice = args.voice
        elif cfg_voice is not None:  # User did not set --voice, and config has a value
            final_voice = cfg_voice
        else:  # User did not set --voice, config has no value, use argparse default
            final_voice = args.voice  # which is "dan"

        # Determine final temperature
        # Argparse default for --temperature is 0.4
        if abs(args.temperature - 0.4) > 1e-9:  # User explicitly set --temperature
            final_temp = args.temperature
        elif cfg_temp is not None:  # User did not set --temperature, config has a value
            final_temp = cfg_temp
        else:  # User did not set --temperature, config has no value, use argparse default
            final_temp = args.temperature  # which is 0.4

        logging.info(f"Using Orpheus model: {final_model_name}")
        logging.info(f"Using voice: {final_voice}, temperature: {final_temp}")

        # Determine WAV output path
        wav_filename = f"orpheus_{uuid.uuid4().hex}.wav"
        if args.out:
            # Place WAV in the same directory as the manifest
            manifest_path = Path(args.out).resolve()
            wav_output_dir = manifest_path.parent
        else:
            # Default to a subdirectory in the current working directory
            wav_output_dir = (Path.cwd() / "tts_audio_output").resolve()

        actual_wav_path = wav_output_dir / wav_filename

        try:
            actual_wav_path.parent.mkdir(parents=True, exist_ok=True)
            synthesis_successful, audio_duration = synthesize_audio(
                args.text, actual_wav_path, final_model_name, final_voice, final_temp
            )

            if synthesis_successful:
                status = "success"
                wav_file_path_str = str(actual_wav_path)
                duration = audio_duration
                exit_code = 0
            # If synthesize_audio fails, error messages are logged within it

        except Exception as e:
            logging.error(f"An unexpected error occurred in main processing: {e}")
            # status remains "failure"

    # Build manifest
    manifest = {
        "text": args.text if args.text else "",
        "wav_path": wav_file_path_str,
        "status": status,
        "duration": duration,
        "pipeline": "orpheus",
    }
    manifest_json = json.dumps(manifest, indent=2)

    # Output manifest
    if args.out:
        try:
            out_path = Path(args.out)
            # Ensure manifest dir exists (parent of actual_wav_path might be different if args.out is nested)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(manifest_json)
            logging.info(f"Manifest written to {out_path}")
        except Exception as e:
            logging.error(f"Error writing manifest to {args.out}: {e}")
            # If manifest writing fails, the overall status might still be success for audio.
            # However, the script's primary output (manifest) failed.
            # We'll stick with the exit_code derived from synthesis success.
            # If synthesis was successful but manifest write fails, exit code is 0.
            # This could be debated, but for now, audio success is primary.
    else:
        print(manifest_json)  # Print to stdout

    sys.exit(exit_code)


if __name__ == "__main__":
    freeze_support()  # Important for multiprocessing, e.g. on Windows
    main()
