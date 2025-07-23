import subprocess
import sys
from pathlib import Path
import uuid
from typing import Union, List, Dict, Any # Moved to top level

def check_datasets(data_dir, output_path=None, step_id="check_datasets"):
    """
    Calls cli/check_datasets.py with the provided data directory and optional parameters using subprocess.

    Args:
        data_dir (str): The path to the data directory.
        output_path (str, optional): Output file for the manifest. Defaults to None.
        step_id (str, optional): Identifier for the current step. Defaults to "check_datasets".

    Returns:
        str: The standard output from the subprocess call.

    Raises:
        RuntimeError: If the subprocess call fails.
    """
    command = [sys.executable, 'cli/check_datasets.py', '--data', data_dir]
    if output_path:
        command.extend(['--out', output_path])
    command.extend(['--step_id', step_id])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error checking datasets: {e.stderr}") from e

def extract_transcript(pptx_path, output_path=None, cue_token="[transition]", step_id="extract_transcript"):
    """
    Calls cli/extract_transcript.py with the specified PPTX file and output path using subprocess.

    Args:
        pptx_path (str): The path to the PPTX file.
        output_path (str, optional): Path to output manifest JSON file. Defaults to None.
        cue_token (str, optional): Token used to split segments. Defaults to "[transition]".
        step_id (str, optional): Identifier for the extraction step. Defaults to "extract_transcript".

    Returns:
        str: The standard output from the subprocess call.

    Raises:
        RuntimeError: If the subprocess call fails.
    """
    command = [sys.executable, 'cli/extract_transcript.py', '--pptx', pptx_path]
    if output_path:
        command.extend(['--out', output_path])
    command.extend(['--cue', cue_token])
    command.extend(['--step_id', step_id])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error extracting transcript: {e.stderr}") from e

def piper_tts(transcript_path, output_path=None, step_id="tts_run", config_path: str = "config/pipeline.yaml"):
    """
    Processes transcript through sentence chunking and TTS conversion pipeline.
    
    1. Runs transcript_preprocess.py with sentence chunking
    2. Passes each chunk to piper_tts.py
    3. Aggregates TTS outputs into a manifest
    
    Args:
        transcript_path (str): Path to input transcript JSON
        output_path (str, optional): Output file for TTS manifest
        step_id (str, optional): Pipeline step identifier
        config_path (str, optional): Path to the pipeline configuration YAML file. Defaults to "config/pipeline.yaml".
    
    Returns:
        dict: Aggregated manifest containing all TTS outputs
        
    Raises:
        RuntimeError: If any CLI operation fails
    """
    import json
    
    # Step 1: Chunk the transcript
    # transcript_preprocess.py expects the input file as a positional argument
    # and does not use --step_id.
    preprocess_cmd = [
        sys.executable, 'cli/transcript_preprocess.py',
        transcript_path, # Positional argument for input file
        '--chunk_mode', 'sentence'
        # Removed '--step_id' as it's not an accepted argument
    ]
    
    try:
        pre_result = subprocess.run(
            preprocess_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        # chunks_data is the full JSON structure from transcript_preprocess.py
        chunks_data = json.loads(pre_result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Transcript preprocessing failed: {e.stderr}") from e
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON from preprocessor: {e}"
        if 'pre_result' in locals() and hasattr(pre_result, 'stdout'):
            error_msg += f"\nOutput was: {pre_result.stdout[:500]}" # Show first 500 chars of output
        raise RuntimeError(error_msg) from e

    # Step 2: Process each chunk with TTS
    tts_manifest = []
    
    # Determine the directory for storing individual chunk manifests and audio files
    # This will be a subdirectory relative to the final aggregated manifest's location
    individual_chunks_dir = None
    if output_path: # output_path is for the final aggregated manifest
        agg_manifest_path = Path(output_path)
        individual_chunks_dir = agg_manifest_path.parent / "piper_audio_chunks"
        try:
            individual_chunks_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # If we can't create this dir, we can't reliably save chunk outputs.
            # The CLI script might default to CWD, which is not desired.
            raise RuntimeError(f"Failed to create directory for individual TTS chunks {individual_chunks_dir}: {e}") from e
            
    all_text_segments_to_speak = []
    if isinstance(chunks_data, dict) and 'slides' in chunks_data:
        for slide in chunks_data.get('slides', []):
            if isinstance(slide, dict) and 'tts_texts' in slide:
                for text_segment in slide.get('tts_texts', []):
                    # Ensure we are adding non-empty strings to be spoken
                    if isinstance(text_segment, str) and text_segment.strip():
                        all_text_segments_to_speak.append(text_segment)
    else:
        # transcript_preprocess.py is expected to return a dictionary with a 'slides' key.
        # If not, it's an unexpected structure.
        raise RuntimeError(
            f"Unexpected data structure from transcript preprocessor. Expected a dict with 'slides'. "
            f"Got: {type(chunks_data)}. Content preview: {str(chunks_data)[:500]}"
        )

    for i, text_to_speak in enumerate(all_text_segments_to_speak):
        chunk_specific_manifest_path = None
        if individual_chunks_dir:
            chunk_specific_manifest_path = individual_chunks_dir / f"chunk_{i}_manifest.json"

        tts_cmd = [
            sys.executable, 'cli/piper_tts.py',
            '--text', text_to_speak, # text_to_speak is now a string
            '--step_id', step_id,
            '--config', config_path
        ]
        if chunk_specific_manifest_path:
            tts_cmd.extend(['--out', str(chunk_specific_manifest_path)])
        
        try:
            tts_result = subprocess.run(
                tts_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            # If chunk_specific_manifest_path was provided to cli/piper_tts.py,
            # the script writes its JSON output to that file, not to stdout.
            # So, we need to read the manifest from that file.
            if chunk_specific_manifest_path:
                if chunk_specific_manifest_path.is_file():
                    with open(chunk_specific_manifest_path, 'r') as f_manifest:
                        tts_data = json.load(f_manifest)
                else:
                    # This would be an unexpected error if the subprocess didn't fail
                    raise RuntimeError(f"TTS CLI completed but manifest file {chunk_specific_manifest_path} not found.")
            else:
                # This case should ideally not be hit if individual_chunks_dir is always created when output_path is present.
                # If --out was not passed to CLI, it prints to stdout.
                tts_data = json.loads(tts_result.stdout)
            
            tts_manifest.append(tts_data)
        except subprocess.CalledProcessError as e:
            # Include stdout from the failed process if available, as it might contain useful error details from the script itself
            error_detail = e.stderr
            if e.stdout:
                error_detail = f"{e.stderr}\nCLI STDOUT: {e.stdout}"
            raise RuntimeError(f"TTS failed for chunk {i}: {error_detail}") from e
        except json.JSONDecodeError as e:
            # This error can happen if we try to parse stdout when we shouldn't,
            # or if the manifest file itself is corrupted.
            error_source = "stdout" if not chunk_specific_manifest_path else str(chunk_specific_manifest_path)
            raise RuntimeError(f"Invalid JSON from TTS output for chunk {i} (source: {error_source}): {e}") from e
        except Exception as e: # Catch other potential errors like file IO issues
            raise RuntimeError(f"Unexpected error processing TTS for chunk {i}: {e}") from e


    # Step 3: Save aggregated manifest
    final_manifest = {
        'source_transcript': transcript_path,
        'tts_results': tts_manifest,
        'step_id': step_id
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_manifest, f, indent=2)
            
    return final_manifest

def orpheus_tts(input_json_data: dict, output_path: str = None, step_id: str = "tts_run", config_path: str = "config/pipeline.yaml"):
    """
    Processes transcript data through Orpheus TTS pipeline.

    1. Invokes cli/transcript_preprocess.py --chunk_mode slide on the input JSON.
    2. Passes the extracted texts to cli/orpheus_tts_cli.py through the --texts-file argument for batch processing.
    3. Returns the resulting single manifest dict (aggregating individual TTS results).

    Args:
        input_json_data (dict): The raw transcript JSON data.
        output_path (str, optional): Output file for the final aggregated TTS manifest. Defaults to None.
        step_id (str, optional): Identifier for the TTS step. Defaults to "tts_run".
        config_path (str, optional): Path to the pipeline configuration YAML file. Defaults to "config/pipeline.yaml".

    Returns:
        dict: Aggregated manifest containing all TTS outputs.

    Raises:
        RuntimeError: If any CLI operation fails or JSON processing errors occur.
    """
    import json
    import tempfile
    import os
    import sys # Ensure sys is imported if not already at the top of the file
    import subprocess # Ensure subprocess is imported
    # from typing import Union, List, Dict, Any # Removed from here, moved to top

    # Step 1: Preprocess the transcript
    # Write input_json_data to a temporary file for transcript_preprocess.py
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json") as tmp_input_file:
        json.dump(input_json_data, tmp_input_file)
        tmp_input_file_path = tmp_input_file.name
    
    preprocessed_json_output = None
    try:
        # Assuming cli/transcript_preprocess.py takes 'input_file' not 'input' as per its own parser
        # Corrected: transcript_preprocess.py expects the input file as a positional argument.
        preprocess_cmd = [
            sys.executable, 'cli/transcript_preprocess.py',
            tmp_input_file_path, # Positional argument for input file
            '--chunk_mode', 'slide'
        ]
        
        pre_result = subprocess.run(
            preprocess_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        preprocessed_json_output = json.loads(pre_result.stdout)
    except subprocess.CalledProcessError as e:
        os.unlink(tmp_input_file_path)
        raise RuntimeError(f"Transcript preprocessing failed: {e.stderr}") from e
    except json.JSONDecodeError as e:
        os.unlink(tmp_input_file_path)
        raise RuntimeError(f"Invalid JSON from preprocessor: {pre_result.stdout if 'pre_result' in locals() else 'Unknown output'}") from e
    finally:
        if os.path.exists(tmp_input_file_path): 
             os.unlink(tmp_input_file_path)

    # Step 2: Extract texts for Orpheus TTS
    texts_for_orpheus = []
    if preprocessed_json_output and "slides" in preprocessed_json_output:
        for slide in preprocessed_json_output.get("slides", []):
            if slide.get("tts_texts"):
                texts_for_orpheus.extend(text_item for text_item in slide["tts_texts"] if text_item and text_item.strip())
            elif slide.get("merged_text") and slide["merged_text"].strip(): 
                texts_for_orpheus.append(slide["merged_text"])
    
    # Orpheus CLI handles empty list in texts_file gracefully by outputting an empty list manifest.

    # Step 3: Pass texts to Orpheus TTS CLI
    # Determine directory for Orpheus's own manifest and audio files
    orpheus_cli_output_dir = None
    orpheus_cli_manifest_path_temp = None # Path for the manifest Orpheus CLI itself will write

    if output_path: # output_path is for the final aggregated manifest from this wrapper
        agg_manifest_path_obj = Path(output_path)
        orpheus_cli_output_dir = agg_manifest_path_obj.parent / "orpheus_audio_chunks"
        try:
            orpheus_cli_output_dir.mkdir(parents=True, exist_ok=True)
            # Create a temporary path for Orpheus CLI to write its own manifest
            # This manifest will contain a list of results for each text processed.
            orpheus_cli_manifest_path_temp = orpheus_cli_output_dir / f"orpheus_cli_output_{uuid.uuid4().hex}.json"
        except Exception as e:
            raise RuntimeError(f"Failed to create directory/path for Orpheus CLI outputs {orpheus_cli_output_dir}: {e}") from e

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json") as tmp_texts_file:
        json.dump(texts_for_orpheus, tmp_texts_file)
        tmp_texts_file_path = tmp_texts_file.name

    orpheus_output_manifest_list = None
    try:
        orpheus_cmd = [
            sys.executable, 'cli/orpheus_tts_cli.py',
            '--texts-file', tmp_texts_file_path,
            '--config', config_path,
            '--step_id', step_id
        ]
        if orpheus_cli_manifest_path_temp:
            orpheus_cmd.extend(['--out', str(orpheus_cli_manifest_path_temp)])

        tts_result = subprocess.run(
            orpheus_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # If orpheus_cli_manifest_path_temp was used, read from that file.
        # Otherwise, cli/orpheus_tts_cli.py prints its JSON to stdout.
        if orpheus_cli_manifest_path_temp:
            if orpheus_cli_manifest_path_temp.is_file():
                with open(orpheus_cli_manifest_path_temp, 'r') as f_cli_manifest:
                    orpheus_output_manifest_list = json.load(f_cli_manifest)
                # Clean up the temporary CLI manifest file
                try:
                    os.unlink(orpheus_cli_manifest_path_temp)
                except OSError as e_unlink:
                    # Log this, but don't fail the whole process if unlink fails
                    print(f"Warning: Could not delete temporary Orpheus CLI manifest {orpheus_cli_manifest_path_temp}: {e_unlink}", file=sys.stderr)
            else:
                raise RuntimeError(f"Orpheus TTS CLI completed but its manifest file {orpheus_cli_manifest_path_temp} not found.")
        else:
            # This path is taken if output_path was not provided to the main orpheus_tts wrapper
            orpheus_output_manifest_list = json.loads(tts_result.stdout)
        
        if not isinstance(orpheus_output_manifest_list, list):
            # This case should ideally be handled by Orpheus CLI exiting non-zero if output is not as expected.
            # If it can still exit 0 with non-list JSON, this check is useful.
            # Orpheus CLI should output a list when --texts-file is used.
            # If it's a single dict, it might be from a direct --text call (not used here) or an error manifest.
            error_source_info = str(orpheus_cli_manifest_path_temp) if orpheus_cli_manifest_path_temp else "stdout"
            raise json.JSONDecodeError(
                f"Orpheus CLI did not return a list as expected (source: {error_source_info}). Got: {type(orpheus_output_manifest_list)}",
                json.dumps(orpheus_output_manifest_list) if isinstance(orpheus_output_manifest_list, dict) else str(orpheus_output_manifest_list)[:200], 0)

    except subprocess.CalledProcessError as e:
        error_detail = e.stderr
        if e.stdout: # Include stdout as it might have more info from the script
            error_detail = f"{e.stderr}\nCLI STDOUT: {e.stdout}"
        if os.path.exists(tmp_texts_file_path): os.unlink(tmp_texts_file_path)
        if orpheus_cli_manifest_path_temp and os.path.exists(orpheus_cli_manifest_path_temp):
             try: os.unlink(orpheus_cli_manifest_path_temp) # Attempt to clean up temp manifest on error too
             except OSError: pass
        raise RuntimeError(f"Orpheus TTS CLI failed: {error_detail}") from e
    except json.JSONDecodeError as e:
        error_source = str(orpheus_cli_manifest_path_temp) if orpheus_cli_manifest_path_temp else "stdout"
        output_preview = ""
        if orpheus_cli_manifest_path_temp and orpheus_cli_manifest_path_temp.is_file():
            with open(orpheus_cli_manifest_path_temp, 'r') as f_err: output_preview = f_err.read(500)
        elif 'tts_result' in locals() and hasattr(tts_result, 'stdout'):
            output_preview = tts_result.stdout[:500]
        if os.path.exists(tmp_texts_file_path): os.unlink(tmp_texts_file_path)
        raise RuntimeError(f"Invalid JSON from Orpheus TTS CLI (source: {error_source}): {e}. Content preview: {output_preview}") from e
    except Exception as e: # Catch other potential errors
        if os.path.exists(tmp_texts_file_path): os.unlink(tmp_texts_file_path)
        if orpheus_cli_manifest_path_temp and os.path.exists(orpheus_cli_manifest_path_temp):
            try: os.unlink(orpheus_cli_manifest_path_temp)
            except OSError: pass
        raise RuntimeError(f"Unexpected error during Orpheus TTS processing: {e}") from e
    finally:
        # Ensure the temporary file for texts is always deleted
        if os.path.exists(tmp_texts_file_path):
            try:
                os.unlink(tmp_texts_file_path)
            except OSError as e_unlink_final:
                 print(f"Warning: Could not delete temporary texts file {tmp_texts_file_path} in finally block: {e_unlink_final}", file=sys.stderr)


    # Step 4: Build the final aggregated manifest (this part seems okay)
    final_aggregated_manifest = {
        "source_transcript_data_preview": {
            k: v for k, v in input_json_data.items() if k != "slides" and k != "segments" # Avoid large data
        },
        "preprocessing_details": preprocessed_json_output.get("preprocessing_applied", {}),
        "number_of_tts_segments_processed": len(texts_for_orpheus),
        "tts_results": orpheus_output_manifest_list if orpheus_output_manifest_list is not None else [],
        "step_id": step_id, # Corrected default is handled in signature
        "config_path": config_path
    }

    if output_path:
        try:
            with open(output_path, 'w') as f:
                json.dump(final_aggregated_manifest, f, indent=2)
                f.write("\n") 
        except IOError as e:
            raise RuntimeError(f"Failed to write aggregated manifest to {output_path}: {e}") from e
            
    return final_aggregated_manifest


def whisper_transcribe( # noqa: E501 (line too long)
    in_data: Union[str, List[str], List[Dict[str, str]]],
    model: str = "large-v3",
    batch_size: int = 16,
    compute_type: str = "float16",
    device: str = "cuda:0"
) -> Dict[str, str]:
    """
    Transcribes audio files using cli/transcribe_whisperx.py.

    Accepts:
    - A string path to a JSON manifest file (list of audio paths).
    - A list of dictionaries, each with a "wav_path" key.
    - A list of string WAV file paths.

    The paths in the input manifest or lists can be relative or absolute.
    Relative paths in a manifest file are resolved relative to the manifest's directory.
    Relative paths in input lists are resolved relative to the current working directory.

    The function calls cli/transcribe_whisperx.py via subprocess, providing it
    with a temporary manifest containing absolute paths to the audio files.
    It parses the JSON output from the script (which contains a list of dictionaries,
    each with "wav" (absolute path) and "asr" (transcription text)) and returns
    a dictionary mapping absolute WAV file paths to their ASR text.

    Args:
        in_data: Input data specifying audio files.
        model: WhisperX model name.
        batch_size: Batch size for WhisperX inference.
        compute_type: Compute type for WhisperX.
        device: Device for WhisperX.

    Returns:
        A dictionary mapping absolute audio file paths to their transcriptions.

    Raises:
        FileNotFoundError: If the input manifest file (if `in_data` is a path) is not found.
        RuntimeError: If the cli/transcribe_whisperx.py script fails or if JSON parsing fails.
    """
    import json
    import tempfile
    import os
    import sys
    import subprocess
    import pathlib

    audio_file_paths_absolute_set: set[str] = set()

    if isinstance(in_data, str):
        manifest_path = pathlib.Path(in_data).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Input manifest file not found: {manifest_path}")
        
        manifest_dir = manifest_path.parent
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                if manifest_path.suffix.lower() == ".json":
                    path_list = json.load(f)
                    if not isinstance(path_list, list):
                        raise ValueError("JSON manifest must contain a list of paths.")
                    for p_str in path_list:
                        p = pathlib.Path(p_str)
                        abs_p = p if p.is_absolute() else manifest_dir / p
                        audio_file_paths_absolute_set.add(str(abs_p.resolve()))
                elif manifest_path.suffix.lower() == ".txt":
                    for line in f:
                        p_str = line.strip()
                        if p_str:
                            p = pathlib.Path(p_str)
                            abs_p = p if p.is_absolute() else manifest_dir / p
                            audio_file_paths_absolute_set.add(str(abs_p.resolve()))
                else:
                    raise ValueError(f"Unsupported manifest file type: {manifest_path.suffix}. Must be .json or .txt.")
        except Exception as e:
            raise RuntimeError(f"Error reading manifest file {manifest_path}: {e}") from e

    elif isinstance(in_data, list):
        if not in_data: # Empty list
            return {}
        
        current_working_dir = pathlib.Path.cwd()
        if isinstance(in_data[0], dict):
            for item in in_data:
                if not isinstance(item, dict) or "wav_path" not in item:
                    raise ValueError("Invalid item in list of dicts: must be a dict with 'wav_path'.")
                p = pathlib.Path(item["wav_path"])
                abs_p = p if p.is_absolute() else current_working_dir / p
                audio_file_paths_absolute_set.add(str(abs_p.resolve()))
        elif isinstance(in_data[0], str):
            for path_str in in_data:
                if not isinstance(path_str, str):
                     raise ValueError("Invalid item in list of strings: all items must be strings (paths).")
                p = pathlib.Path(path_str)
                abs_p = p if p.is_absolute() else current_working_dir / p
                audio_file_paths_absolute_set.add(str(abs_p.resolve()))
        else:
            raise ValueError("Input list must contain either all dicts (with 'wav_path') or all strings (paths).")
    else:
        raise TypeError("in_data must be a str (manifest path) or a list (of paths or dicts).")

    audio_file_paths_absolute = sorted(list(audio_file_paths_absolute_set))

    if not audio_file_paths_absolute:
        return {}

    tmp_manifest_path = None
    tmp_output_path = None

    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as tmp_manifest_file:
            json.dump(audio_file_paths_absolute, tmp_manifest_file)
            tmp_manifest_path = tmp_manifest_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_output_file: # Not opened in text mode, just need name
            tmp_output_path = tmp_output_file.name

        script_path = str(pathlib.Path(__file__).parent.parent / "cli" / "transcribe_whisperx.py")

        cmd = [
            sys.executable, script_path,
            "--in", tmp_manifest_path,
            "--model", model,
            "--out", tmp_output_path,
            "--batch_size", str(batch_size),
            "--compute_type", compute_type,
            "--device", device
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8')

        if process.returncode != 0:
            # Check if output file was created and has content, even on error
            error_output_content = ""
            if os.path.exists(tmp_output_path):
                try:
                    with open(tmp_output_path, 'r', encoding='utf-8') as f_err_out:
                        error_output_content = f_err_out.read(500) # Read first 500 chars
                    if error_output_content:
                         error_output_content = f"\nContent of output file '{tmp_output_path}' (partial):\n{error_output_content}"
                except Exception:
                    pass # Ignore if can't read error output file

            raise RuntimeError(
                f"cli/transcribe_whisperx.py failed with exit code {process.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {process.stderr.strip() if process.stderr else 'N/A'}\n"
                f"Stdout: {process.stdout.strip() if process.stdout else 'N/A'}"
                f"{error_output_content}"
            )

        if not os.path.exists(tmp_output_path) or os.path.getsize(tmp_output_path) == 0:
             # This case implies the script might have exited 0 but failed to produce the output file.
            raise RuntimeError(
                f"cli/transcribe_whisperx.py completed successfully but output file '{tmp_output_path}' is missing or empty.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {process.stderr.strip() if process.stderr else 'N/A'}\n"
                f"Stdout: {process.stdout.strip() if process.stdout else 'N/A'}"
            )

        with open(tmp_output_path, 'r', encoding='utf-8') as f:
            results_list = json.load(f)
            
        # The script cli/transcribe_whisperx.py already outputs absolute paths in the 'wav' field.
        return {item["wav"]: item["asr"] for item in results_list}

    finally:
        if tmp_manifest_path and os.path.exists(tmp_manifest_path):
            try:
                os.remove(tmp_manifest_path)
            except OSError as e:
                print(f"Warning: Could not delete temporary manifest file {tmp_manifest_path}: {e}", file=sys.stderr)
        if tmp_output_path and os.path.exists(tmp_output_path):
            try:
                os.remove(tmp_output_path)
            except OSError as e:
                print(f"Warning: Could not delete temporary output file {tmp_output_path}: {e}", file=sys.stderr)


def run_rvc_convert(
    input_manifest: str,
    output_dir: str,
    config_path: str = "config/pipeline.yaml",
    step_id: str = "apply_rvc"
) -> Dict[str, Any]:
    """
    Wrapper for RVC voice conversion CLI.
    
    Args:
        input_manifest: Path to TTS audio manifest
        output_dir: Output directory for RVC converted audio
        config_path: Path to pipeline configuration file
        step_id: Identifier for the RVC step
        
    Returns:
        Dict: RVC conversion results
        
    Raises:
        RuntimeError: If RVC conversion fails
    """
    cmd = [
        sys.executable, "cli/rvc_convert.py",
        "--input", input_manifest,
        "--output", output_dir,
        "--config", config_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Load the generated manifest
        manifest_path = Path(output_dir) / "rvc_conversion_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return json.load(f)
        else:
            raise RuntimeError(f"RVC manifest not found at {manifest_path}")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"RVC conversion failed: {e.stderr}") from e


def run_splice_audio(
    input_manifest: str,
    output_dir: str,
    output_name: str = "final_narration.wav",
    config_path: str = "config/pipeline.yaml",
    step_id: str = "splice_audio"
) -> Dict[str, Any]:
    """
    Wrapper for audio splicing CLI.
    
    Args:
        input_manifest: Path to RVC audio manifest
        output_dir: Output directory for spliced audio
        output_name: Output filename for spliced audio
        config_path: Path to pipeline configuration file
        step_id: Identifier for the splicing step
        
    Returns:
        Dict: Audio splicing results
        
    Raises:
        RuntimeError: If audio splicing fails
    """
    cmd = [
        sys.executable, "cli/splice_audio.py",
        "--input", input_manifest,
        "--output", output_dir,
        "--output-name", output_name,
        "--config", config_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Load the generated manifest
        manifest_path = Path(output_dir) / "splice_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return json.load(f)
        else:
            raise RuntimeError(f"Splice manifest not found at {manifest_path}")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio splicing failed: {e.stderr}") from e


def analyze_video(video_path, transcript_path=None, output_path=None, 
                 scene_threshold=0.4, movement_threshold=0.1, keynote_delay=1.0,
                 presentation_mode=False, expected_transitions=0,
                 step_id="analyze_video"):
    """
    Calls cli/analyze_video.py to analyze video for scene transitions and movement ranges.
    
    Args:
        video_path (str): Path to the video file to analyze
        transcript_path (str, optional): Path to transcript JSON for validation. Defaults to None.
        output_path (str, optional): Path for analysis manifest output. Defaults to None.
        scene_threshold (float, optional): Scene detection sensitivity. Defaults to 0.4.
        movement_threshold (float, optional): Movement detection sensitivity. Defaults to 0.1.
        keynote_delay (float, optional): Keynote delay compensation in seconds. Defaults to 1.0.
        presentation_mode (bool, optional): Enable Keynote-specific optimizations. Defaults to False.
        expected_transitions (int, optional): Expected number of transitions for validation. Defaults to 0.
        step_id (str, optional): Step identifier for pipeline integration. Defaults to "analyze_video".
    
    Returns:
        dict: Analysis manifest containing scene transitions and movement ranges
        
    Raises:
        RuntimeError: If video analysis fails
    """
    import json
    
    command = [sys.executable, 'cli/analyze_video.py', video_path]
    
    if transcript_path:
        command.extend(['--transcript', transcript_path])
    
    if output_path:
        command.extend(['--output', output_path])
    
    command.extend([
        '--scene-threshold', str(scene_threshold),
        '--movement-threshold', str(movement_threshold),
        '--keynote-delay', str(keynote_delay),
        '--step-id', step_id
    ])
    
    if presentation_mode:
        command.append('--presentation-mode')
    
    if expected_transitions > 0:
        command.extend(['--expected-transitions', str(expected_transitions)])
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the JSON output from the CLI tool
        try:
            analysis_result = json.loads(result.stdout)
            
            # If output_path was specified and file exists, load the full manifest
            if output_path and Path(output_path).exists():
                with open(output_path, 'r') as f:
                    full_manifest = json.load(f)
                return full_manifest
            else:
                # Return the summary result
                return analysis_result
                
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse video analysis output: {e}") from e
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Video analysis failed: {e.stderr}") from e


def sync_video(video_path: str,
               audio_path: str,
               output_path: str,
               video_manifest: str = None,
               audio_manifest: str = None,
               step_id: str = "sync_video",
               config_path: str = "config/pipeline.yaml") -> Dict[str, Any]:
    """
    Synchronizes video with processed audio using the AutoVid sync engine.
    
    This wrapper function calls cli/sync_video.py to perform video-audio synchronization
    using timing manifests from video analysis and audio splicing steps. It supports
    both basic and advanced synchronization modes with comprehensive validation.
    
    Args:
        video_path (str): Path to input video file
        audio_path (str): Path to input audio file  
        output_path (str): Path for output synchronized video
        video_manifest (str, optional): Path to video analysis manifest
        audio_manifest (str, optional): Path to audio splice manifest
        step_id (str, optional): Identifier for the sync step. Defaults to "sync_video"
        config_path (str, optional): Path to pipeline configuration. Defaults to "config/pipeline.yaml"
        
    Returns:
        dict: Synchronization manifest containing sync points and validation results
        
    Raises:
        RuntimeError: If video synchronization fails
    """
    import json
    import tempfile
    import os
    
    # Build command
    command = [sys.executable, 'cli/sync_video.py', video_path, audio_path, output_path]
    
    # Add optional manifests
    if video_manifest:
        command.extend(['--video-manifest', video_manifest])
    
    if audio_manifest:
        command.extend(['--audio-manifest', audio_manifest])
    
    # Add config
    command.extend(['--config', config_path])
    
    # Enable validation and create temporary manifest file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp_manifest:
        tmp_manifest_path = tmp_manifest.name
    
    command.extend([
        '--validate',
        '--sync-manifest', tmp_manifest_path
    ])
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Load the synchronization manifest
        sync_manifest = {}
        if os.path.exists(tmp_manifest_path):
            try:
                with open(tmp_manifest_path, 'r') as f:
                    sync_manifest = json.load(f)
            except json.JSONDecodeError as e:
                # If manifest parsing fails, create a basic success response
                sync_manifest = {
                    'video_sync': {
                        'output_video': output_path,
                        'status': 'success',
                        'step_id': step_id,
                        'manifest_parse_error': str(e)
                    }
                }
        else:
            # If no manifest was created, create a basic success response
            sync_manifest = {
                'video_sync': {
                    'output_video': output_path,
                    'status': 'success',
                    'step_id': step_id
                }
            }
        
        # Clean up temporary file
        try:
            os.unlink(tmp_manifest_path)
        except OSError:
            pass
        
        return sync_manifest
        
    except subprocess.CalledProcessError as e:
        # Clean up temporary file on error
        try:
            os.unlink(tmp_manifest_path)
        except OSError:
            pass
        raise RuntimeError(f"Video synchronization failed: {e.stderr}") from e


def run_audio_qc(
    input_manifest: str,
    output_dir: str,
    mos_threshold: float = 3.5,
    wer_threshold: float = 0.10,
    max_attempts: int = 3,
    whisper_model: str = "large-v3",
    enable_transcription: bool = True,
    transcription_timeout: int = 30,
    retry_with_phonemes: bool = True,
    retry_different_engine: bool = True,
    preserve_original_on_failure: bool = False,
    detect_clipping: bool = True,
    detect_silence: bool = True,
    silence_threshold: float = -40,
    min_chunk_duration: float = 0.5,
    max_chunk_duration: float = 30.0,
    step_id: str = "qc_pronounce"
) -> Dict[str, Any]:
    """
    Quality control wrapper for TTS audio validation with comprehensive features.
    
    Args:
        input_manifest: Path to TTS audio manifest JSON
        output_dir: Output directory for QC results  
        mos_threshold: Minimum MOS score threshold
        wer_threshold: Maximum WER threshold
        max_attempts: Maximum re-synthesis attempts
        whisper_model: WhisperX model for transcription
        enable_transcription: Enable WhisperX transcription for WER
        transcription_timeout: Timeout for transcription in seconds
        retry_with_phonemes: Use phoneme hints for failed chunks
        retry_different_engine: Try alternate TTS engine if available
        preserve_original_on_failure: Keep original audio if all retries fail
        detect_clipping: Detect audio clipping
        detect_silence: Detect unexpected silence periods
        silence_threshold: dB threshold for silence detection
        min_chunk_duration: Minimum expected chunk duration (seconds)
        max_chunk_duration: Maximum expected chunk duration (seconds)
        step_id: Step identifier
        
    Returns:
        Dict: QC results manifest
        
    Raises:
        RuntimeError: If QC processing fails
    """
    import json
    
    cmd = [
        sys.executable, "cli/qc_audio.py",
        "--input", input_manifest,
        "--output", output_dir,
        "--mos-threshold", str(mos_threshold),
        "--wer-threshold", str(wer_threshold),
        "--max-attempts", str(max_attempts),
        "--whisper-model", whisper_model,
        "--transcription-timeout", str(transcription_timeout),
        "--silence-threshold", str(silence_threshold),
        "--min-chunk-duration", str(min_chunk_duration),
        "--max-chunk-duration", str(max_chunk_duration),
        "--step-id", step_id
    ]
    
    # Add boolean flags
    if enable_transcription:
        cmd.append("--enable-transcription")
    if retry_with_phonemes:
        cmd.append("--retry-with-phonemes")
    if retry_different_engine:
        cmd.append("--retry-different-engine")
    if preserve_original_on_failure:
        cmd.append("--preserve-original-on-failure")
    if detect_clipping:
        cmd.append("--detect-clipping")
    if detect_silence:
        cmd.append("--detect-silence")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Load and return QC manifest
        manifest_stem = Path(input_manifest).stem
        qc_manifest_path = Path(output_dir) / f"qc_manifest_{manifest_stem}.json"
        
        if qc_manifest_path.exists():
            with open(qc_manifest_path, 'r') as f:
                qc_manifest = json.load(f)
                qc_manifest["manifest_path"] = str(qc_manifest_path)
                return qc_manifest
        else:
            raise RuntimeError(f"QC manifest not found: {qc_manifest_path}")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio QC failed: {e.stderr}") from e
