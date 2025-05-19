import json
import subprocess
import sys
from pathlib import Path
import os
import yaml

import pytest

# Define the path to the CLI script
ORPHEUS_TTS_CLI = Path(__file__).parent.parent / "cli" / "orpheus_tts_cli.py"

# Default values from config/pipeline.yaml for orpheus (or typical test values)
DEFAULT_MODEL_NAME = "canopylabs/orpheus-tts-0.1-finetune-prod" # Ensure this model is accessible or use a mock
DEFAULT_CONFIG_VOICE = "dan"
DEFAULT_CONFIG_TEMPERATURE = 0.2

@pytest.fixture
def default_config_file(tmp_path: Path) -> Path:
    """Creates a temporary config file with default Orpheus settings."""
    config_content = {
        "model_dir": str(tmp_path / "models"), # Dummy model_dir
        "steps": [
            {
                "id": "tts_run",
                "parameters": {
                    "orpheus_model": DEFAULT_MODEL_NAME,
                    "orpheus_voice": DEFAULT_CONFIG_VOICE,
                    "orpheus_temperature": DEFAULT_CONFIG_TEMPERATURE,
                },
            }
        ],
    }
    config_file = tmp_path / "test_pipeline_batch.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)
    return config_file

@pytest.fixture
def texts_file_json(tmp_path: Path):
    """
    Factory fixture to create a temporary JSON file with a list of texts.
    Usage: texts_file = texts_file_json(["text1", "text2"])
    """
    def _texts_file_json(content: list[str], filename="texts.json") -> Path:
        file_path = tmp_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(content, f)
        return file_path
    return _texts_file_json

def run_cli(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Helper to run the Orpheus TTS CLI script."""
    command = [sys.executable, str(ORPHEUS_TTS_CLI)] + args
    timeout_seconds = 300 # Increased timeout for potential multiple syntheses

    env = os.environ.copy()
    orpheus_tts_source_dir = str(Path(__file__).parent.parent / "third_party" / "Orpheus-TTS" / "orpheus_tts_pypi")
    env['PYTHONPATH'] = orpheus_tts_source_dir
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # Ensure consistent GPU ordering
    env['CUDA_VISIBLE_DEVICES'] = '1' # Attempt to use GPU 1 (hoping it's less loaded)
    
    # For debugging test environment
    # print(f"CLI command: {' '.join(command)}")
    # print(f"PYTHONPATH for subprocess: {env['PYTHONPATH']}")
    # print(f"Working directory for subprocess: {kwargs.get('cwd', Path.cwd())}")


    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_seconds, env=env, **kwargs)

def assert_single_manifest_entry_success(
    entry: dict,
    expected_text: str,
    expected_wav_dir: Path | None = None,
):
    """Asserts common success conditions for a single entry in a batch manifest."""
    assert entry["status"] == "success"
    assert entry["text"] == expected_text
    assert entry["pipeline"] == "orpheus"
    assert isinstance(entry["duration"], float)
    assert entry["duration"] > 0  # Should have some duration
    assert "error" not in entry

    wav_path_str = entry["wav_path"]
    assert wav_path_str is not None
    wav_path = Path(wav_path_str)
    assert wav_path.exists(), f"WAV file {wav_path} does not exist."
    assert wav_path.is_file()
    assert wav_path.suffix == ".wav"
    assert wav_path.stat().st_size > 1000, f"WAV file {wav_path} seems too small."

    if expected_wav_dir:
        assert wav_path.parent.resolve() == expected_wav_dir.resolve(), \
               f"WAV parent dir {wav_path.parent.resolve()} != expected {expected_wav_dir.resolve()}"

def assert_single_manifest_entry_failure(
    entry: dict,
    expected_text: str,
    expected_error_msg_part: str | None = None,
):
    """Asserts common failure conditions for a single entry."""
    assert entry["status"] == "failure"
    assert entry["text"] == expected_text
    assert entry["pipeline"] == "orpheus"
    assert entry["duration"] == 0.0
    assert entry["wav_path"] is None
    assert "error" in entry
    if expected_error_msg_part:
        assert expected_error_msg_part.lower() in entry["error"].lower()


# === Test Cases ===

def test_batch_success_file_output(tmp_path: Path, default_config_file: Path, texts_file_json):
    """Test successful batch TTS to a file output manifest."""
    texts = ["Batch test one.", "Batch test two, with a comma."]
    input_json_file = texts_file_json(texts, "batch_input.json")
    manifest_output_file = tmp_path / "batch_output.json"

    args = [
        "--texts-file", str(input_json_file),
        "--out", str(manifest_output_file),
        "--config", str(default_config_file),
    ]
    result = run_cli(args)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"CLI exited with {result.returncode}, STDERR: {result.stderr}"

    assert manifest_output_file.exists()
    with open(manifest_output_file, "r", encoding="utf-8") as f:
        manifest_list = json.load(f)

    assert isinstance(manifest_list, list)
    assert len(manifest_list) == len(texts)

    for i, text_content in enumerate(texts):
        assert_single_manifest_entry_success(manifest_list[i], text_content, expected_wav_dir=tmp_path)

def test_batch_success_stdout_output(tmp_path: Path, default_config_file: Path, texts_file_json):
    """Test successful batch TTS with manifest to stdout."""
    texts = ["Stdout batch one.", "Stdout batch two."]
    input_json_file = texts_file_json(texts, "stdout_batch.json")
    
    # Store CWD before running CLI, as CLI might create tts_audio_output relative to its CWD
    # For tests, run_cli doesn't change CWD by default unless specified.
    # So Path.cwd() here is the project root.
    expected_wav_parent_dir = Path.cwd() / "tts_audio_output"


    args = [
        "--texts-file", str(input_json_file),
        "--config", str(default_config_file),
        # No --out, so stdout
    ]
    result = run_cli(args)

    print("STDOUT (raw):", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"CLI exited with {result.returncode}, STDERR: {result.stderr}"
    
    stdout_content = result.stdout
    manifest_list = None
    current_search_idx = 0
    found_json_successfully = False

    # Try to find the start of the JSON list by looking for specific patterns
    # '[\n' for non-empty, pretty-printed lists
    # '[]' for empty lists
    while current_search_idx < len(stdout_content):
        idx_pattern_non_empty = stdout_content.find('[\n', current_search_idx)
        idx_pattern_empty = stdout_content.find('[]', current_search_idx)

        json_start_index = -1

        # Determine which pattern (if any) comes first
        if idx_pattern_non_empty != -1 and (idx_pattern_empty == -1 or idx_pattern_non_empty < idx_pattern_empty):
            json_start_index = idx_pattern_non_empty
        elif idx_pattern_empty != -1 and (idx_pattern_non_empty == -1 or idx_pattern_empty < idx_pattern_non_empty):
            json_start_index = idx_pattern_empty
        # If both found at same spot (unlikely for '[\n' and '[]'), prioritize one or handle as error
        # For these distinct patterns, this case isn't a primary concern.

        if json_start_index == -1:
            # No more occurrences of our target patterns
            break 
        
        potential_json_str = stdout_content[json_start_index:]
        try:
            loaded_json = json.loads(potential_json_str)
            # Ensure it's a list, as expected by this batch test
            if isinstance(loaded_json, list):
                manifest_list = loaded_json
                found_json_successfully = True
                break # Successfully parsed the manifest
            else:
                # Parsed valid JSON, but it wasn't a list. This isn't our manifest.
                # Continue searching from after this found JSON-like structure.
                # A simple way to advance past is to use the length of the string representation
                # of the loaded_json, but this might be complex if it's very large.
                # For now, advancing by 1 from the start of this attempt is safer if it's not our list.
                current_search_idx = json_start_index + 1 # Advance past this attempt
        except json.JSONDecodeError:
            # This substring (from json_start_index to end) didn't parse.
            # Advance search to after the point where we started this attempt.
            current_search_idx = json_start_index + 1 
            # Adding 1 is a minimal advance; if '[\n' or '[]' was part of a larger non-JSON string,
            # this ensures we don't retry the exact same spot infinitely.

    if not found_json_successfully or manifest_list is None: # manifest_list check for clarity
        pytest.fail(
            f"Failed to parse JSON list from stdout. Tried patterns '[]' and '[\\n'.\n"
            f"Stdout content (raw) was:\n{result.stdout}" # Show raw stdout for debugging
        )

    assert isinstance(manifest_list, list)
    assert len(manifest_list) == len(texts)

    for i, text_content in enumerate(texts):
        assert_single_manifest_entry_success(manifest_list[i], text_content, expected_wav_dir=expected_wav_parent_dir)


def test_batch_with_empty_texts_list_in_file(tmp_path: Path, default_config_file: Path, texts_file_json):
    """Test batch processing with an empty list in the JSON file."""
    input_json_file = texts_file_json([], "empty_batch.json")
    manifest_output_file = tmp_path / "empty_batch_output.json"

    args = [
        "--texts-file", str(input_json_file),
        "--out", str(manifest_output_file),
        "--config", str(default_config_file),
    ]
    result = run_cli(args)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"CLI should exit 0 for empty list, STDERR: {result.stderr}"

    assert manifest_output_file.exists()
    with open(manifest_output_file, "r", encoding="utf-8") as f:
        manifest_list = json.load(f)

    assert isinstance(manifest_list, list)
    assert len(manifest_list) == 0

def test_batch_error_texts_file_not_found(tmp_path: Path, default_config_file: Path):
    """Test error when --texts-file does not exist."""
    non_existent_file = tmp_path / "non_existent.json"
    args = [
        "--texts-file", str(non_existent_file),
        "--config", str(default_config_file),
    ]
    result = run_cli(args)
    assert result.returncode != 0
    assert f"--texts-file not found: {non_existent_file}".lower() in result.stderr.lower()

def test_batch_error_invalid_json_in_texts_file(tmp_path: Path, default_config_file: Path, texts_file_json):
    """Test error when --texts-file contains invalid JSON."""
    input_file = tmp_path / "invalid.json"
    with open(input_file, "w") as f:
        f.write("this is not json [")
    
    args = [
        "--texts-file", str(input_file),
        "--config", str(default_config_file),
    ]
    result = run_cli(args)
    assert result.returncode != 0
    assert f"Invalid JSON in --texts-file: {input_file}".lower() in result.stderr.lower()

def test_batch_error_texts_file_not_a_list(tmp_path: Path, default_config_file: Path, texts_file_json):
    """Test error when --texts-file JSON is not a list."""
    input_file = tmp_path / "not_a_list.json"
    with open(input_file, "w") as f:
        json.dump({"oops": "not a list"}, f)

    args = [
        "--texts-file", str(input_file),
        "--config", str(default_config_file),
    ]
    result = run_cli(args)
    assert result.returncode != 0
    assert f"--texts-file ({input_file}) does not contain a JSON list".lower() in result.stderr.lower()

def test_mutual_exclusion_text_and_texts_file(tmp_path: Path, default_config_file: Path, texts_file_json):
    """Test error if both --text and --texts-file are provided."""
    input_json_file = texts_file_json(["hello"], "dummy.json")
    args = [
        "--text", "some text",
        "--texts-file", str(input_json_file),
        "--config", str(default_config_file),
    ]
    result = run_cli(args)
    assert result.returncode != 0
    assert "Cannot use both --text and --texts-file".lower() in result.stderr.lower()

def test_error_neither_text_nor_texts_file(default_config_file: Path):
    """Test error if neither --text nor --texts-file is provided."""
    args = [
        "--config", str(default_config_file), # Only config
    ]
    result = run_cli(args)
    assert result.returncode != 0
    assert "Either --text or --texts-file must be provided".lower() in result.stderr.lower()

def test_batch_item_empty_string(tmp_path: Path, default_config_file: Path, texts_file_json):
    """Test a batch where one item is an empty string."""
    texts = ["Valid text.", "", "Another valid text."]
    input_json_file = texts_file_json(texts, "batch_with_empty.json")
    manifest_output_file = tmp_path / "batch_empty_item_output.json"

    args = [
        "--texts-file", str(input_json_file),
        "--out", str(manifest_output_file),
        "--config", str(default_config_file),
    ]
    result = run_cli(args)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    # The script should exit with non-zero because one item failed (empty string)
    assert result.returncode != 0, f"CLI should exit non-zero for partial failure. STDERR: {result.stderr}"

    assert manifest_output_file.exists()
    with open(manifest_output_file, "r", encoding="utf-8") as f:
        manifest_list = json.load(f)

    assert isinstance(manifest_list, list)
    assert len(manifest_list) == len(texts)

    # First item should succeed
    assert_single_manifest_entry_success(manifest_list[0], texts[0], expected_wav_dir=tmp_path)

    # Second item (empty string) should fail
    assert_single_manifest_entry_failure(manifest_list[1], texts[1], expected_error_msg_part="Input text is empty")
    
    # Third item should succeed
    assert_single_manifest_entry_success(manifest_list[2], texts[2], expected_wav_dir=tmp_path)

def test_single_text_mode_still_works_file_output(tmp_path: Path, default_config_file: Path):
    """Test that original single --text mode still works with file output."""
    text_to_synth = "Single text mode test, file output."
    manifest_file = tmp_path / "single_output.json"

    args = [
        "--text", text_to_synth,
        "--out", str(manifest_file),
        "--config", str(default_config_file),
    ]
    result = run_cli(args)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"CLI exited with {result.returncode}, STDERR: {result.stderr}"

    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f) # Single object, not a list

    assert isinstance(manifest, dict)
    assert_single_manifest_entry_success(manifest, text_to_synth, expected_wav_dir=tmp_path)

def test_single_text_mode_empty_text_fails(tmp_path: Path, default_config_file: Path):
    """Test that single --text mode with empty string fails correctly."""
    manifest_file = tmp_path / "single_empty_fail.json"
    args = [
        "--text", "",
        "--out", str(manifest_file),
        "--config", str(default_config_file),
    ]
    result = run_cli(args)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode != 0, f"CLI should exit non-zero for empty text. STDERR: {result.stderr}"

    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert_single_manifest_entry_failure(manifest, "", expected_error_msg_part="Input text is empty")
