import subprocess
import sys
from pathlib import Path

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

        tts_result = subprocess.run(
            orpheus_cmd,
            capture_output=True,
            text=True,
            check=True 
        )
        orpheus_output_manifest_list = json.loads(tts_result.stdout)
        
        if not isinstance(orpheus_output_manifest_list, list):
            # This case should ideally be handled by Orpheus CLI exiting non-zero if output is not as expected.
            # If it can still exit 0 with non-list JSON, this check is useful.
            raise json.JSONDecodeError("Orpheus CLI did not return a list for --texts-file mode.", tts_result.stdout, 0)

    except subprocess.CalledProcessError as e:
        # Stderr from Orpheus CLI should provide good error info
        error_detail = e.stderr if e.stderr else e.stdout # Sometimes errors go to stdout
        os.unlink(tmp_texts_file_path) # Ensure cleanup
        raise RuntimeError(f"Orpheus TTS CLI failed: {error_detail}") from e
    except json.JSONDecodeError as e:
        # This means Orpheus CLI output was not valid JSON
        os.unlink(tmp_texts_file_path) # Ensure cleanup
        raise RuntimeError(f"Invalid JSON from Orpheus TTS CLI: {tts_result.stdout if 'tts_result' in locals() else 'Unknown output'}") from e
    finally:
        if os.path.exists(tmp_texts_file_path):
            os.unlink(tmp_texts_file_path)

    # Step 4: Aggregate results into a single manifest
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
