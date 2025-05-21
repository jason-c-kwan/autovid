import subprocess
import json
import os
import pathlib
import tempfile
import pytest
import shutil
import torch

SCRIPT_PATH = (pathlib.Path(__file__).parent.parent / "cli" / "transcribe_whisperx.py").resolve()
PIPER_TTS_SCRIPT_PATH = (pathlib.Path(__file__).parent.parent / "cli" / "piper_tts.py").resolve()
# FIXTURES_DIR and TEST_LECTURE_MOV are no longer needed if we only use TTS
# FIXTURES_DIR = (pathlib.Path(__file__).parent / "fixtures").resolve()
# TEST_LECTURE_MOV = FIXTURES_DIR / "test_lecture.mov"

# def is_ffmpeg_available(): # No longer needed
#     """Check if ffmpeg is available in PATH."""
#     return shutil.which("ffmpeg") is not None

@pytest.fixture(scope="module")
def sample_wav_files(tmp_path_factory):
    """
    Generates a few short sample WAV files using cli/piper_tts.py.
    """
    # Base temporary directory for all TTS outputs for this module run
    module_tts_tmp_dir = tmp_path_factory.mktemp("tts_module_samples")
    generated_wav_paths = []
    
    samples_to_synthesize = [
        {"id": "sample1", "text": "Hello world, this is the first audio sample for testing."},
        {"id": "sample2", "text": "This is a second, different audio sample for the whisperx tests."},
    ]

    for sample_info in samples_to_synthesize:
        # Create a dedicated subdirectory for this specific sample's TTS output
        # This helps in reliably finding the generated UUID-named WAV file.
        sample_output_dir = module_tts_tmp_dir / sample_info["id"]
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the path for the dummy manifest.json for piper_tts.py
        manifest_path = sample_output_dir / "manifest.json"

        cmd = [
            "python", str(PIPER_TTS_SCRIPT_PATH),
            "--text", sample_info["text"],
            "--out", str(manifest_path),
            # Assuming default --config config/pipeline.yaml and --step_id tts_run are used
            # and correctly configured in the environment.
        ]
        
        found_wav_path = None
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find the generated .wav file in sample_output_dir
            # piper_tts.py creates a UUID.wav file in the same dir as the --out manifest.
            wav_files_in_dir = list(sample_output_dir.glob("*.wav"))
            if len(wav_files_in_dir) == 1:
                found_wav_path = wav_files_in_dir[0]
                if found_wav_path.is_file() and found_wav_path.stat().st_size > 0:
                    generated_wav_paths.append(found_wav_path.resolve())
                else:
                    # File found by glob but is_file is false or size is 0
                    err_msg = f"piper_tts.py ran for '{sample_info['id']}', found {found_wav_path}, but it's invalid (empty or not a file)."
                    if hasattr(process, 'stderr'): err_msg += f" stderr: {process.stderr}"
                    pytest.fail(err_msg)
            elif len(wav_files_in_dir) > 1:
                pytest.fail(f"piper_tts.py ran for '{sample_info['id']}' and created multiple WAV files in {sample_output_dir}: {wav_files_in_dir}")
            else: # No WAV files found
                err_msg = f"piper_tts.py ran for '{sample_info['id']}' but no WAV file was found in {sample_output_dir}."
                if hasattr(process, 'stderr'): err_msg += f" stderr: {process.stderr}"
                pytest.fail(err_msg)

        except subprocess.CalledProcessError as e:
            pytest.fail(f"piper_tts.py command failed for '{sample_info['id']}': {e.stderr}")
        except FileNotFoundError: # For python interpreter or PIPER_TTS_SCRIPT_PATH itself
            pytest.fail(f"Failed to find Python interpreter or piper_tts.py script at {PIPER_TTS_SCRIPT_PATH} for sample '{sample_info['id']}'.")
            
    if not generated_wav_paths or len(generated_wav_paths) != len(samples_to_synthesize):
        pytest.fail("Not all WAV files were successfully generated using piper_tts.py.")
        
    return generated_wav_paths


def run_script(args_list, expect_success=True):
    """Helper to run the transcription script."""
    cmd = ["python", str(SCRIPT_PATH)] + args_list
    process = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if expect_success and process.returncode != 0:
        print(f"STDOUT:\n{process.stdout}")
        print(f"STDERR:\n{process.stderr}")
        pytest.fail(f"Script execution failed with code {process.returncode}")
    elif not expect_success and process.returncode == 0:
        print(f"STDOUT:\n{process.stdout}")
        print(f"STDERR:\n{process.stderr}")
        pytest.fail("Script execution succeeded but was expected to fail.")
    return process

# Common arguments for tests to speed them up
COMMON_TEST_ARGS = ["--model", "tiny", "--compute_type", "int8", "--device", "cpu", "--batch_size", "2"]

def validate_output(result_list, expected_num_files, sample_wav_paths_str):
    assert isinstance(result_list, list), "Output should be a list."
    assert len(result_list) == expected_num_files, f"Expected {expected_num_files} results, got {len(result_list)}."
    
    processed_wav_names = {pathlib.Path(item["wav"]).name for item in result_list}
    expected_wav_names = {pathlib.Path(p).name for p in sample_wav_paths_str}

    for item in result_list:
        assert isinstance(item, dict), "Each item in result list should be a dict."
        assert {"wav", "asr"} <= item.keys(), "Each item must have 'wav' and 'asr' keys."
        assert pathlib.Path(item["wav"]).name in expected_wav_names, f"Unexpected WAV path {item['wav']} in results."
        assert isinstance(item["asr"], str), "ASR result should be a string."
        # Basic check that ASR is not empty, more specific checks are hard without knowing content
        # For very short files, ASR can sometimes be empty if no speech is detected by 'tiny'
        # assert len(item["asr"].strip()) > 0, f"ASR for {item['wav']} is empty or whitespace."

    assert processed_wav_names == expected_wav_names, "Mismatch in processed WAV files."


def test_transcription_stdout(sample_wav_files):
    sample_paths_str = [str(p) for p in sample_wav_files]
    args = ["--in"] + sample_paths_str + COMMON_TEST_ARGS
    process = run_script(args)
    
    result_data = json.loads(process.stdout)
    validate_output(result_data, len(sample_wav_files), sample_paths_str)

def test_transcription_outfile(sample_wav_files, tmp_path):
    sample_paths_str = [str(p) for p in sample_wav_files]
    out_file = tmp_path / "output.json"
    args = ["--in"] + sample_paths_str + ["--out", str(out_file)] + COMMON_TEST_ARGS
    
    run_script(args)
    
    assert out_file.exists(), "Output file was not created."
    with open(out_file, 'r') as f:
        result_data = json.load(f)
    validate_output(result_data, len(sample_wav_files), sample_paths_str)

def test_transcription_manifest_txt(sample_wav_files, tmp_path):
    sample_paths_str = [str(p) for p in sample_wav_files]
    manifest_file = tmp_path / "manifest.txt"
    with open(manifest_file, 'w') as f:
        for p_str in sample_paths_str:
            f.write(p_str + "\n")
            
    args = ["--in", str(manifest_file)] + COMMON_TEST_ARGS
    process = run_script(args)
    result_data = json.loads(process.stdout)
    validate_output(result_data, len(sample_wav_files), sample_paths_str)

def test_transcription_manifest_json(sample_wav_files, tmp_path):
    sample_paths_str = [str(p) for p in sample_wav_files]
    manifest_file = tmp_path / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(sample_paths_str, f)
        
    args = ["--in", str(manifest_file)] + COMMON_TEST_ARGS
    process = run_script(args)
    result_data = json.loads(process.stdout)
    validate_output(result_data, len(sample_wav_files), sample_paths_str)

def test_transcription_glob_input(sample_wav_files, tmp_path):
    # Ensure sample_wav_files are in a predictable location for globbing (they are, due to tmp_path_factory)
    # The sample_wav_files fixture already places them in a unique temp dir per test module.
    # We just need to make sure the glob pattern correctly points there.
    # sample_wav_files contains absolute paths.
    
    if not sample_wav_files:
        pytest.skip("No sample WAV files generated, skipping glob test.")

    # Get the common parent directory for all generated sample subdirectories
    # e.g., if files are /tmp/tts_module_samples/sample1/uuid.wav and /tmp/tts_module_samples/sample2/uuid.wav,
    # base_glob_dir will be /tmp/tts_module_samples/
    base_glob_dir = sample_wav_files[0].parent.parent 
    glob_pattern = str(base_glob_dir / "*/*.wav") # Glob all wavs in all sample subdirectories
    
    sample_paths_str = [str(p) for p in sample_wav_files]

    args = ["--in", glob_pattern] + COMMON_TEST_ARGS
    process = run_script(args)
    result_data = json.loads(process.stdout)
    validate_output(result_data, len(sample_wav_files), sample_paths_str)


def test_invalid_input_file_path():
    args = ["--in", "non_existent_file.wav"] + COMMON_TEST_ARGS
    process = run_script(args, expect_success=True) # Script exits 0 for this case
    # The script now exits 0 with empty list if no files found after resolving inputs
    # So, if "non_existent_file.wav" is the *only* input and it's not a manifest,
    # resolve_input_paths will print a warning and return an empty list.
    # The main script will then print an empty JSON list and exit 0.
    assert process.returncode == 0, f"Script should exit 0 for non-found files, got {process.returncode}. Stderr: {process.stderr}"
    result_data = json.loads(process.stdout)
    assert result_data == [], "Expected empty list for non-existent input file."
    assert "Warning: Input path or pattern not found/matched: 'non_existent_file.wav'" in process.stderr


def test_invalid_manifest_file_path():
    args = ["--in", "non_existent_manifest.txt"] + COMMON_TEST_ARGS
    process = run_script(args, expect_success=False) 
    # This should fail because resolve_input_paths raises FileNotFoundError for manifest,
    # and the main script catches this and exits 1.
    assert process.returncode != 0, "Script should exit non-zero for non-existent manifest."
    assert "Error: Manifest file not found: 'non_existent_manifest.txt'" in process.stderr


def test_no_input_files_found_glob(tmp_path):
    # Use a glob pattern that won't match anything in an empty temp directory
    args = ["--in", str(tmp_path / "non_existent_pattern_*.wav")] + COMMON_TEST_ARGS
    process = run_script(args) # Expect success (exit 0)
    assert process.returncode == 0
    result_data = json.loads(process.stdout)
    assert result_data == [], "Expected empty list for glob pattern with no matches."
    assert "Warning: No valid input audio files found." in process.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for this specific test, or no CUDA devices found.")
def test_invalid_cuda_device_when_cuda_available():
    """
    Tests script failure when CUDA is available but an invalid CUDA device ID is requested.
    This test only runs if torch.cuda.is_available() is True.
    """
    # Create a dummy wav file so resolve_input_paths doesn't exit early
    # and the script proceeds to device checks.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        dummy_wav_path = tmp_wav.name
    
    invalid_device_id = "cuda:999" # Assuming this device does not exist
    args_with_dummy = [
        "--in", dummy_wav_path,
        "--model", "tiny",
        "--compute_type", "int8", # Use fast settings
        "--device", invalid_device_id
    ]
    
    try:
        process = run_script(args_with_dummy, expect_success=False)
        assert process.returncode != 0
        # The script's check `torch.cuda.get_device_properties(args.device)` should raise an error,
        # leading to the custom error message.
        expected_error_prefix = f"Error: Invalid CUDA device '{invalid_device_id}'."
        assert expected_error_prefix in process.stderr, \
            f"Expected stderr to contain '{expected_error_prefix}', but got: {process.stderr}"
    finally:
        os.remove(dummy_wav_path)

def test_empty_input_list():
    # Test scenario where --in is provided but after resolving, the list is empty
    # e.g., a manifest file exists but is empty or contains only invalid paths
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_manifest:
        tmp_manifest.write("\n") # Empty or invalid content
        manifest_path = tmp_manifest.name
    
    try:
        args = ["--in", manifest_path] + COMMON_TEST_ARGS
        process = run_script(args) # Expect success (exit 0)
        assert process.returncode == 0
        result_data = json.loads(process.stdout)
        assert result_data == [], "Expected empty list for empty/invalid manifest."
        assert "Warning: No valid input audio files found." in process.stderr
    finally:
        os.remove(manifest_path)
