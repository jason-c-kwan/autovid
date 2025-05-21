import subprocess
import sys

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

def piper_tts(transcript_path, output_path=None, step_id="tts_run"):
    """
    Processes transcript through sentence chunking and TTS conversion pipeline.
    
    1. Runs transcript_preprocess.py with sentence chunking
    2. Passes each chunk to piper_tts.py
    3. Aggregates TTS outputs into a manifest
    
    Args:
        transcript_path (str): Path to input transcript JSON
        output_path (str, optional): Output file for TTS manifest
        step_id (str, optional): Pipeline step identifier
    
    Returns:
        dict: Aggregated manifest containing all TTS outputs
        
    Raises:
        RuntimeError: If any CLI operation fails
    """
    import json
    
    # Step 1: Chunk the transcript
    preprocess_cmd = [
        sys.executable, 'cli/transcript_preprocess.py',
        '--input', transcript_path,
        '--chunk_mode', 'sentence',
        '--step_id', step_id
    ]
    
    try:
        pre_result = subprocess.run(
            preprocess_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        chunks = json.loads(pre_result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Transcript preprocessing failed: {e.stderr}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from preprocessor: {e}") from e

    # Step 2: Process each chunk with TTS
    tts_manifest = []
    for i, chunk in enumerate(chunks):
        tts_cmd = [
            sys.executable, 'cli/piper_tts.py',
            '--text', chunk['text'],
            '--step_id', f"{step_id}.chunk_{i}"
        ]
        
        try:
            tts_result = subprocess.run(
                tts_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            tts_data = json.loads(tts_result.stdout)
            tts_manifest.append(tts_data)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"TTS failed for chunk {i}: {e.stderr}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid TTS output for chunk {i}: {e}") from e

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
        preprocess_cmd = [
            sys.executable, 'cli/transcript_preprocess.py',
            '--input_file', tmp_input_file_path, 
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
