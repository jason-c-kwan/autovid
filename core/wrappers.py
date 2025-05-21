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
