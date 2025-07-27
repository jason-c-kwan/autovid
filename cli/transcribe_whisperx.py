import argparse
import argparse
import json
import sys
import glob
import os
import pathlib
import torch
import whisperx
import io # Added for stdout redirection
from contextlib import redirect_stdout # Added for stdout redirection

def resolve_input_paths(input_specs: list[str]) -> list[pathlib.Path]:
    """
    Resolves input specifications to a list of absolute file paths.
    Supports direct file paths, glob patterns, .txt manifest files (one path per line),
    and .json manifest files (list of paths).
    """
    resolved_paths = set()
    for spec in input_specs:
        p_spec = pathlib.Path(spec)
        # Check for manifest-like suffixes first
        is_potential_manifest = p_spec.suffix.lower() in [".txt", ".json"]

        if is_potential_manifest and not p_spec.is_file():
            print(f"Error: Manifest file not found: '{spec}'", file=sys.stderr)
            raise FileNotFoundError(f"Manifest file not found: {spec}")

        if p_spec.is_file():
            if p_spec.suffix.lower() == ".txt":
                try:
                    with open(p_spec, 'r') as f:
                        for line in f:
                            line_path = pathlib.Path(line.strip())
                            if line_path.is_file():
                                resolved_paths.add(line_path.resolve())
                            elif line.strip(): # Non-empty line that's not a file
                                print(f"Warning: Path from manifest '{p_spec}' not found: '{line.strip()}'", file=sys.stderr)
                except Exception as e: # Catching generic exception after specific FileNotFoundError above
                    print(f"Error reading manifest file '{spec}': {e}", file=sys.stderr)
                    raise
            elif p_spec.suffix.lower() == ".json":
                try:
                    with open(p_spec, 'r') as f:
                        path_list = json.load(f)
                        if not isinstance(path_list, list):
                            raise ValueError("JSON manifest must contain a list of paths.")
                        for path_str in path_list:
                            line_path = pathlib.Path(path_str)
                            if line_path.is_file():
                                resolved_paths.add(line_path.resolve())
                            else:
                                print(f"Warning: Path from JSON manifest '{p_spec}' not found: '{path_str}'", file=sys.stderr)
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from manifest file: '{spec}'", file=sys.stderr)
                    raise
                except ValueError as e:
                    print(f"Error: Invalid format in JSON manifest '{spec}': {e}", file=sys.stderr)
                    raise
                except Exception as e: # Catching generic exception
                    print(f"Error reading JSON manifest file '{spec}': {e}", file=sys.stderr)
                    raise
            else: # Direct file path (not .txt or .json, or was but is_file was true)
                resolved_paths.add(p_spec.resolve())
        else: # Not a file, and not a potential manifest that was missing. So, glob or non-existent direct path.
            try:
                # This part handles globs, or single paths that didn't exist and weren't manifests
                glob_paths = glob.glob(spec, recursive=True)
                found_glob = False
                for g_path_str in glob_paths:
                    g_path = pathlib.Path(g_path_str)
                    if g_path.is_file():
                        resolved_paths.add(g_path.resolve())
                        found_glob = True
                if not found_glob and not any("*" in s or "?" in s or "[" in s for s in spec): # Heuristic for non-glob that wasn't found
                     print(f"Warning: Input path or pattern not found/matched: '{spec}'", file=sys.stderr)
            except Exception as e:
                print(f"Error processing glob pattern '{spec}': {e}", file=sys.stderr)
                # Continue, maybe other specs are valid

    if not resolved_paths:
        print("Warning: No valid input audio files found.", file=sys.stderr)
    
    return sorted(list(resolved_paths))


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using WhisperX.")
    parser.add_argument(
        "--in", dest="input_specs", nargs='+', required=True,
        help="Input audio file paths, glob patterns, or paths to .txt/.json manifest files."
    )
    parser.add_argument(
        "--model", default="large-v3",
        help="WhisperX model name (e.g., 'large-v3', 'base'). Default: 'large-v3'."
    )
    parser.add_argument(
        "--out", default=None,
        help="Path to output JSON file. If omitted, prints to stdout."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for WhisperX inference. Default: 16."
    )
    parser.add_argument(
        "--compute_type", default="float16",
        help="Compute type for WhisperX (e.g., 'float16', 'int8', 'float32'). Default: 'float16'."
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device for WhisperX (e.g., 'cuda:0', 'cpu'). Default: 'cuda:0'."
    )
    # Placeholder for language if needed in future
    # parser.add_argument("--language", default=None, help="Language code for transcription (e.g., 'en').")


    args = parser.parse_args()

    try:
        audio_files = resolve_input_paths(args.input_specs)

        if not audio_files:
            # Output empty list if no files, as per plan
            output_results = []
            if args.out:
                with open(args.out, 'w') as f:
                    json.dump(output_results, f, indent=2)
            else:
                print(json.dumps(output_results, indent=2))
            sys.exit(0) # Successful exit, even if no files processed

        if "cuda" in args.device:
            if not torch.cuda.is_available():
                print(f"Error: CUDA device '{args.device}' requested, but CUDA is not available.", file=sys.stderr)
                sys.exit(1)
            try: # Check if the specific CUDA device is valid
                # Attempting to get properties for a non-existent device should fail
                # Extract device index from "cuda:X" format
                if ":" in args.device:
                    device_index = int(args.device.split(":")[1])
                else:
                    device_index = 0
                torch.cuda.get_device_properties(device_index)
            except (RuntimeError, AssertionError, IndexError) as e: # Catch typical errors for invalid device
                print(f"Error: Invalid CUDA device '{args.device}'. PyTorch error: {e}", file=sys.stderr)
                sys.exit(1)
            except Exception as e: # Catch any other unexpected errors during device check
                print(f"Error: Unexpected issue validating CUDA device '{args.device}': {e}", file=sys.stderr)
                sys.exit(1)
        
        print(f"Loading WhisperX model '{args.model}' on device '{args.device}' with compute type '{args.compute_type}'. This may take a while...", file=sys.stderr)
        model_kwargs = {"device": args.device, "compute_type": args.compute_type}
        # if args.language: # Add language if specified
        # model_kwargs["language"] = args.language
        
        # Capture stdout from whisperx library calls
        whisperx_stdout_capture = io.StringIO()
        with redirect_stdout(whisperx_stdout_capture):
            try:
                model = whisperx.load_model(args.model, **model_kwargs)
                # Note: "Model loaded." and transcription progress will now go to whisperx_stdout_capture
                # if they were printed to stdout by whisperx. Our own prints to sys.stderr are unaffected.
                
                audio_file_paths_str = [str(p) for p in audio_files]
                transcription_results_collector = []

                # Our own print to stderr, this is fine.
                print("Model loaded.", file=sys.stderr) 
                print(f"Transcribing {len(audio_file_paths_str)} audio file(s) individually with batch size {args.batch_size}...", file=sys.stderr)

                for i, path_str in enumerate(audio_file_paths_str):
                    print(f"Processing file {i+1}/{len(audio_file_paths_str)}: {path_str}", file=sys.stderr)
                    try:
                        individual_output = model.transcribe(path_str, batch_size=args.batch_size)
                        
                        if not isinstance(individual_output, dict):
                            print(f"Warning: model.transcribe for {path_str} did not return a dict (got {type(individual_output)}). Skipping this file.", file=sys.stderr)
                            transcription_results_collector.append({"error": "Invalid output from transcribe", "segments": []}) 
                            continue
                        transcription_results_collector.append(individual_output)
                    except Exception as e_transcribe:
                        print(f"Error during transcription of {path_str}: {e_transcribe}", file=sys.stderr)
                        transcription_results_collector.append({"error": str(e_transcribe), "segments": []})
                
                transcription_outputs = transcription_results_collector

            except Exception as e_load_transcribe:
                # Handle exceptions during model load or the transcription loop itself
                # These errors should also be reported, and script should exit.
                # The captured whisperx stdout might still be useful.
                print(f"Error during whisperx model loading or transcription process: {e_load_transcribe}", file=sys.stderr)
                # Ensure transcription_outputs is an empty list or indicates error for all files
                # so that the script can proceed to output JSON and exit with error.
                num_files = len(audio_files) if audio_files else 0
                transcription_outputs = [{"error": str(e_load_transcribe), "segments": []} for _ in range(num_files)]


        # Print captured whisperx stdout to our stderr for diagnostics
        captured_library_stdout = whisperx_stdout_capture.getvalue()
        if captured_library_stdout:
            print("\n--- WhisperX Library Output (stdout) ---", file=sys.stderr)
            print(captured_library_stdout, file=sys.stderr)
            print("--- End WhisperX Library Output ---\n", file=sys.stderr)

        results = []
        # Ensure audio_file_paths_str is defined even if transcription block had issues before it was set
        if 'audio_file_paths_str' not in locals() and audio_files:
            audio_file_paths_str = [str(p) for p in audio_files]
        elif not audio_files : # Should have been handled earlier, but as a safeguard
            audio_file_paths_str = []


        if len(transcription_outputs) != len(audio_file_paths_str):
            print(f"Critical Error: Mismatch between number of input files ({len(audio_file_paths_str)}) and collected transcription results ({len(transcription_outputs)}). This indicates a bug in the processing loop.", file=sys.stderr)
            # If this happens, results list might be misaligned.
            # To prevent crashing, we can try to build a partial results list or an error list.
            # For now, let's make results reflect the error for all expected files.
            results = [{"wav": p_str, "asr": "Error: Result misalignment"} for p_str in audio_file_paths_str]
            # And ensure script exits with error later
            # This situation implies a logic error above, so this is a fallback.
        else:
            for i, output_dict in enumerate(transcription_outputs):
                current_audio_file_path = audio_file_paths_str[i]
                if "error" in output_dict:
                    print(f"Skipping processing for '{current_audio_file_path}' due to transcription error: {output_dict['error']}", file=sys.stderr)
                    full_text = f"Error: {output_dict['error']}"
                elif "segments" not in output_dict or not isinstance(output_dict["segments"], list):
                    print(f"Warning: No segments found or invalid segment format for '{current_audio_file_path}'. Skipping.", file=sys.stderr)
                    full_text = "" 
                else:
                    full_text = " ".join([segment["text"].strip() for segment in output_dict["segments"] if "text" in segment])
                results.append({"wav": current_audio_file_path, "asr": full_text})
                if "error" not in output_dict:
                     print(f"Completed processing for: {current_audio_file_path}", file=sys.stderr)

        if args.out:
            output_destination_path = pathlib.Path(args.out)
            output_destination_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_destination_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Output written to {output_destination_path}", file=sys.stderr)
        else:
            # This print should now be clean JSON as whisperx stdout was captured
            print(json.dumps(results, indent=2))

        # Determine exit status based on whether any errors occurred during transcription for any file
        # This is a bit more nuanced now. If any file has an "Error: ..." in ASR, it's a partial success/failure.
        # For simplicity, if the main try/except didn't catch a FileNotFoundError or other major setup error,
        # and we produced some results (even if some are errors), exit 0.
        # The JSON output itself contains error details per file.
        # A more strict approach would be to exit 1 if any 'error' key is present in results.
        # For now, let's stick to exiting 0 if the script completed its main processing loop.
        # The main try-except will catch broader failures.
        
        # Check if any result contains an error to decide final exit code
        # This makes the exit code more accurately reflect if all operations were successful.
        final_status_successful = True
        if not results: # No audio files processed, or error before results populated
             final_status_successful = False # Or based on earlier checks for audio_files
        for res_item in results:
            if isinstance(res_item.get("asr"), str) and res_item["asr"].startswith("Error:"):
                final_status_successful = False
                break
        
        if final_status_successful:
            sys.exit(0)
        else:
            print("Exiting with status 1 due to errors in processing one or more files.", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError: # Handled by resolve_input_paths for manifests, or other critical file issues
        sys.exit(1) # Exit if manifest file itself not found
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
