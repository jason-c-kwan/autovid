#!/usr/bin/env python3
"""
Command-line interface for Orpheus TTS.

Synthesizes text to a WAV audio file using an Orpheus TTS model specified in
the pipeline configuration and outputs a JSON manifest.
"""

import argparse
import json
import logging
import gc
import os
import orpheus_tts.engine_class as _oe
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import sys
import time
import uuid
import wave
from multiprocessing import freeze_support
from pathlib import Path

import torch
import yaml

def shutdown_vllm(model):
    try:
        engine: AsyncLLMEngine = model._engine      # type: ignore
        engine.shutdown()
    except Exception:
        pass

def _patched_setup_engine(self):
    # hard caps for a 24 GiB card – tune if needed
    max_len = int(os.getenv("ORPHEUS_MAX_LEN", 768))
    util     = float(os.getenv("ORPHEUS_GPU_UTIL", 0.7))   # 70 % of VRAM
    dtype    = os.getenv("ORPHEUS_DTYPE", "float16")        # half-precision weights

    args = AsyncEngineArgs(
        model                   = self.model_name,
        dtype                   = dtype,
        max_model_len           = max_len,
        gpu_memory_utilization  = util,
        # optional extras:
        kv_cache_dtype="fp8_e4m3",   # more VRAM savings if CUDA ≥ 11.8
        block_size=8,                # smaller blocks → less waste
        # max_num_seqs=1,              # disable large speculative batches
        enforce_eager=True
    )
    return AsyncLLMEngine.from_engine_args(args)

_oe.OrpheusModel._setup_engine = _patched_setup_engine

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
        "--text", type=str, required=False, help="Text to synthesize (cannot be used with --texts-file)."
    )
    parser.add_argument(
        "--texts-file", type=str, help="Path to a JSON file containing a list of texts to synthesize (cannot be used with --text)."
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (default: auto-select). Example: 'cuda:0', 'cpu'",
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


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Force garbage collection
    gc.collect()


def synthesize_audio(
    model: OrpheusModel, 
    text_to_synthesize: str,
    output_wav_path: Path,
    voice: str,
    temperature: float,
    device: str = None,
) -> tuple[bool, float]:
    """
    Synthesize text to a WAV file using a pre-initialized OrpheusModel.
    """
    try:
        logging.info(
            f"Synthesizing speech for text: '{text_to_synthesize[:50]}...' "
            f"with voice: {voice}, temperature: {temperature}"
        )
        start_time = time.monotonic()
        
        # Set specific device if provided
        device_context = {}
        if device:
            device_context = {"device": device}
        
        # Disable gradient tracking for memory-efficient inference
        with torch.inference_mode(), torch.autocast("cuda"):
            # Generate speech with the model
            syn_tokens_generator = model.generate_speech(
                prompt=text_to_synthesize, voice=voice, temperature=temperature,
            )

            output_wav_path.parent.mkdir(parents=True, exist_ok=True)

            total_frames = 0
            duration = 0.0
            sample_rate = 24000 
            sample_width = 2  
            num_channels = 1  

            with wave.open(str(output_wav_path), "wb") as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)

                # Process chunks directly without storing them in memory
                for chunk in syn_tokens_generator:
                    if chunk: 
                        frames_in_chunk = len(chunk) // (sample_width * num_channels)
                        total_frames += frames_in_chunk
                        wf.writeframes(chunk)
                        # Release memory within the loop
                        del chunk
                        torch.cuda.empty_cache()

                if total_frames > 0:
                    duration = total_frames / sample_rate
                else: 
                    logging.warning("No audio data generated by Orpheus TTS.")

        # Explicitly delete generator and clear memory
        del syn_tokens_generator
        clear_gpu_memory()

        processing_time = time.monotonic() - start_time
        logging.info(
            f"Audio synthesis complete. Output: {output_wav_path}, "
            f"Duration: {duration:.2f}s, Processing time: {processing_time:.2f}s"
        )
        return True, duration

    except FileNotFoundError as e: 
        logging.error(f"A required file was not found: {e}")
        clear_gpu_memory()
        return False, 0.0
    except RuntimeError as e:
        logging.error(f"Orpheus TTS runtime error: {e}")
        clear_gpu_memory()
        return False, 0.0
    except Exception as e:
        logging.error(f"An unexpected error occurred during Orpheus TTS synthesis: {e}")
        clear_gpu_memory()
        return False, 0.0


def main():
    """Main function to orchestrate TTS and manifest generation."""
    args = parse_args()

    # --- Set device based on arguments or availability ---
    device = args.device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    logging.info(f"Using device: {device}")
    
    # Initial memory clearance
    clear_gpu_memory()

    # --- Argument Validation ---
    if args.text and args.texts_file:
        logging.error("Cannot use both --text and --texts-file. Please provide only one.")
        sys.exit(1)
    
    texts_to_process_source: str | None = None
    if args.texts_file:
        texts_to_process_source = "file"
    elif args.text is not None: 
        if not args.text.strip():
            logging.error("Input text (--text) cannot be empty.")
            output_data = {
                "text": "", "wav_path": None, "status": "failure",
                "duration": 0.0, "pipeline": "orpheus", "error": "Input text is empty."
            }
            json_output_str = json.dumps(output_data, indent=2)
            if args.out:
                try:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    with open(args.out, "w", encoding="utf-8") as f:
                        f.write(json_output_str)
                    logging.info(f"Failure manifest for empty text written to {args.out}")
                except Exception as e_write:
                    logging.error(f"Error writing empty text failure manifest: {e_write}")
            else:
                print(json_output_str)
            sys.exit(1)
        texts_to_process_source = "text_arg"
    else: 
        logging.error("Either --text or --texts-file must be provided.")
        sys.exit(1)

    # --- Parameter Resolution (Config, CLI, Defaults) ---
    cfg_orpheus_model, cfg_voice, cfg_temp = get_config_params(
        args.config, args.step_id
    )

    final_model_name = cfg_orpheus_model
    if not final_model_name:
        logging.error(
            f"Orpheus model name ('orpheus_model') not found in config '{args.config}' "
            f"for step_id '{args.step_id}'. Cannot proceed."
        )
        output_data = {"text": args.text if args.text else "", "wav_path": None, "status": "failure",
                       "duration": 0.0, "pipeline": "orpheus",
                       "error": f"Orpheus model name not configured for step '{args.step_id}'."}
        if args.texts_file: # If batch mode was intended, output empty list on this type of config error
            output_data = []
        
        json_output_str = json.dumps(output_data, indent=2)
        if args.out:
            try:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                with open(args.out, "w", encoding="utf-8") as f: f.write(json_output_str)
                logging.info(f"Failure manifest written to {args.out}")
            except Exception as e: logging.error(f"Error writing failure manifest to {args.out}: {e}")
        else: print(json_output_str)
        sys.exit(1)

    if args.voice != "dan": final_voice = args.voice
    elif cfg_voice is not None: final_voice = cfg_voice
    else: final_voice = args.voice 

    if abs(args.temperature - 0.4) > 1e-9: final_temp = args.temperature
    elif cfg_temp is not None: final_temp = cfg_temp
    else: final_temp = args.temperature 

    # --- Load texts to process ---
    texts_to_process: list[str] = []
    if texts_to_process_source == "file":
        try:
            with open(args.texts_file, "r", encoding="utf-8") as f: # type: ignore
                loaded_texts = json.load(f)
            if not isinstance(loaded_texts, list):
                logging.error(f"--texts-file ({args.texts_file}) does not contain a JSON list.")
                sys.exit(1)
            texts_to_process = [str(item) for item in loaded_texts] 
            if not texts_to_process: 
                 logging.info(f"Empty list of texts provided in {args.texts_file}. No audio will be generated.")
        except FileNotFoundError:
            logging.error(f"--texts-file not found: {args.texts_file}")
            sys.exit(1)
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in --texts-file: {args.texts_file}")
            sys.exit(1)
    elif texts_to_process_source == "text_arg":
        texts_to_process = [args.text] 

    results_manifest_list = []
    overall_success = True 

    if not texts_to_process: # Handles empty list from file or if somehow text_arg was empty (should be caught earlier)
        final_output_data = [] if args.texts_file else {"text": args.text if args.text else "", "wav_path": None, "status": "success" if not args.text else "failure", "duration":0.0, "pipeline":"orpheus", "error": "No text to process" if not args.text else None}
        if args.texts_file and not texts_to_process : # Specifically for empty list from file
             logging.info(f"Empty list of texts provided in {args.texts_file}. Outputting empty manifest.")

        json_output_str = json.dumps(final_output_data, indent=2)
        if args.out:
            try:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                with open(args.out, "w", encoding="utf-8") as f: f.write(json_output_str)
                logging.info(f"Manifest for no texts written to {args.out}")
            except Exception as e:
                logging.error(f"Error writing manifest for no texts: {e}")
                overall_success = False 
        else: print(json_output_str)
        sys.exit(0 if overall_success else 1)

    # --- Initialize Model (only if there are texts to process) ---
    model: OrpheusModel | None = None
    try:
        logging.info(f"Initializing OrpheusModel with model: {final_model_name}")
        # Set device for model initialization
        model_kwargs = {"device": device} if device else {}
        # patch must run BEFORE the first Orpheus import builds vLLM
        model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

        logging.info(f"Using voice: {final_voice}, temperature: {final_temp}")
    except Exception as e_model_init:
        logging.error(f"Failed to initialize OrpheusModel ({final_model_name}): {e_model_init}")
        for text_input_for_failure in texts_to_process:
            results_manifest_list.append({
                "text": text_input_for_failure, "wav_path": None, "status": "failure",
                "duration": 0.0, "pipeline": "orpheus",
                "error": f"Failed to initialize OrpheusModel: {e_model_init}",
            })
        overall_success = False
    
    if model and overall_success: # Only proceed if model loaded and no prior critical error
        for i, text_input in enumerate(texts_to_process):
            status = "failure"
            wav_file_path_str: str | None = None
            duration: float = 0.0
            error_message: str | None = None

            if not text_input or not text_input.strip():
                logging.warning(f"Skipping empty text input from batch: '{text_input}'")
                error_message = "Input text is empty."
                overall_success = False 
            else:
                wav_filename = f"orpheus_{uuid.uuid4().hex}.wav"
                wav_output_dir = (Path(args.out).resolve().parent if args.out 
                                  else (Path.cwd() / "tts_audio_output").resolve())
                actual_wav_path = wav_output_dir / wav_filename

                try:
                    actual_wav_path.parent.mkdir(parents=True, exist_ok=True)
                    synthesis_successful, audio_duration = synthesize_audio(
                        model, text_input, actual_wav_path, final_voice, final_temp, device
                    )
                    if synthesis_successful:
                        status = "success"
                        wav_file_path_str = str(actual_wav_path)
                        duration = audio_duration
                    else: 
                        overall_success = False
                        error_message = "Synthesis failed (see logs for details)."
                except Exception as e_synth:
                    logging.error(f"An unexpected error occurred during synthesis for text '{text_input[:30]}...': {e_synth}")
                    overall_success = False
                    error_message = f"Unexpected error: {e_synth}"

            entry = {"text": text_input, "wav_path": wav_file_path_str, "status": status,
                     "duration": duration, "pipeline": "orpheus"}
            if error_message: entry["error"] = error_message
            results_manifest_list.append(entry)
            
            # Explicitly clear memory after each text in batch mode
            if len(texts_to_process) > 1:
                logging.info(f"Processed item {i+1}/{len(texts_to_process)}, clearing memory")
                clear_gpu_memory()
    
    # Final memory cleanup
    clear_gpu_memory()

    # --- Determine final output data ---
    if args.texts_file:
        final_output_data = results_manifest_list
    elif results_manifest_list: 
        final_output_data = results_manifest_list[0]
    else: 
        logging.error("Internal error: No results in list for single text mode after processing.")
        final_output_data = {"text": args.text, "wav_path": None, "status": "failure",
                             "duration": 0.0, "pipeline": "orpheus", "error": "Unknown error, no result generated."}
        overall_success = False

    final_json_output = json.dumps(final_output_data, indent=2)

    if args.out:
        try:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f: f.write(final_json_output)
            logging.info(f"Manifest written to {out_path}")
        except Exception as e:
            logging.error(f"Error writing manifest to {args.out}: {e}")
            overall_success = False 
    else:
        print(final_json_output)

    if model is not None:
        shutdown_vllm(model)
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()          # silence NCCL warning
        except Exception:
            pass
        del model
        clear_gpu_memory()

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    freeze_support() 
    main()
