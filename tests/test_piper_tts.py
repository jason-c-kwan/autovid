import json
import subprocess
import sys
import yaml
from pathlib import Path
import pytest

# Path to the script to be tested
SCRIPT_PATH = Path(__file__).parent.parent / "cli" / "piper_tts.py"
DEFAULT_TEST_MODEL = "en_US-lessac-medium" # A common, smallish model for testing

# Helper to create a dummy pipeline.yaml for testing
def create_test_config_file(
    tmp_path: Path, 
    step_id: str, 
    piper_model_name: str | None, 
    global_model_dir: str | None, # Can be None to test missing global_model_dir
    params_dict: dict | None = None
) -> Path:
    """
    Creates a temporary pipeline.yaml file for testing.
    - global_model_dir: Value for the top-level 'model_dir'. If None, key is omitted.
    - piper_model_name: Value for 'piper_model' in step parameters. If None, key omitted.
    - params_dict: Overrides step parameters if provided.
    """
    config_file_path = tmp_path / "test_pipeline.yaml"
    
    config_content = {} # Initialize as dict for top-level keys
    if global_model_dir is not None:
        config_content["model_dir"] = global_model_dir
    
    config_content["steps"] = []
    step_data = {"id": step_id}

    if params_dict is not None:
        step_data["parameters"] = params_dict
    elif piper_model_name is not None:
        step_data["parameters"] = {"piper_model": piper_model_name}
    # If both params_dict and piper_model_name are None, step has no 'parameters' key

    config_content["steps"].append(step_data)

    with open(config_file_path, "w") as f:
        yaml.dump(config_content, f)
    return config_file_path

def run_script(cmd_args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Helper to run the script using subprocess."""
    base_cmd = [sys.executable, str(SCRIPT_PATH)]
    full_cmd = base_cmd + cmd_args
    # print(f"Running command: {' '.join(full_cmd)}", file=sys.stderr) # For debugging tests
    return subprocess.run(full_cmd, capture_output=True, text=True, check=False, cwd=cwd)

def assert_manifest_success(manifest_data: dict, expected_text: str, wav_parent_dir: Path):
    """Asserts for a successful manifest."""
    assert manifest_data["text"] == expected_text
    assert manifest_data["status"] == "success"
    assert manifest_data["wav_path"] is not None
    wav_path = Path(manifest_data["wav_path"])
    assert wav_path.name.endswith(".wav")
    assert wav_path.is_file(), f"WAV file {wav_path} not found or not a file."
    assert wav_path.stat().st_size > 0, f"WAV file {wav_path} is empty."
    # Check if the wav_path is absolute and its parent is the expected directory
    assert wav_path.is_absolute()
    assert wav_path.parent == wav_parent_dir.resolve()


def assert_manifest_failure(manifest_data: dict, expected_text: str | None):
    """Asserts for a failed manifest."""
    if expected_text is not None: # Text might not be in manifest if parsing args fails before text is stored
        assert manifest_data["text"] == expected_text
    assert manifest_data["status"] == "failure"
    assert manifest_data["wav_path"] is None


@pytest.mark.skipif(not Path("/usr/bin/espeak-ng").exists() and not Path("/usr/local/bin/espeak-ng").exists(), reason="espeak-ng not found, piper-tts may fail to download/use models")
def test_piper_tts_success_to_file(tmp_path: Path):
    """Test successful TTS synthesis writing manifest and WAV to file."""
    test_text = "Hello from piper tts to file."
    manifest_file = tmp_path / "output.json"
    test_model_data_dir = tmp_path / "test_piper_models"
    test_model_data_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = create_test_config_file(
        tmp_path, 
        step_id="tts_run", 
        piper_model_name=DEFAULT_TEST_MODEL, 
        global_model_dir=str(test_model_data_dir)
    )

    cmd_args = [
        "--text", test_text,
        "--out", str(manifest_file),
        "--config", str(config_file)
    ]
    result = run_script(cmd_args)

    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
    assert manifest_file.exists()
    
    manifest_data = json.loads(manifest_file.read_text())
    assert_manifest_success(manifest_data, test_text, manifest_file.parent)
    # Cleanup the created WAV file
    if manifest_data.get("wav_path"):
        Path(manifest_data["wav_path"]).unlink(missing_ok=True)

@pytest.mark.skipif(not Path("/usr/bin/espeak-ng").exists() and not Path("/usr/local/bin/espeak-ng").exists(), reason="espeak-ng not found, piper-tts may fail to download/use models")
def test_piper_tts_success_to_stdout(tmp_path: Path, capsys):
    """Test successful TTS synthesis printing manifest to stdout."""
    test_text = "Hello from piper tts to stdout."
    test_model_data_dir = tmp_path / "test_piper_models_stdout"
    test_model_data_dir.mkdir(parents=True, exist_ok=True)
    
    # WAV files will go into tmp_path / "tts_audio_output" when script is run from tmp_path
    # This is because the script defaults to CWD/tts_audio_output if --out is not given
    # and we run the script with cwd=tmp_path
    wav_output_parent_dir = tmp_path / "tts_audio_output" 

    config_file = create_test_config_file(
        tmp_path, 
        step_id="tts_run", 
        piper_model_name=DEFAULT_TEST_MODEL,
        global_model_dir=str(test_model_data_dir)
    )

    cmd_args = [
        "--text", test_text,
        "--config", str(config_file)
    ]
    result = run_script(cmd_args, cwd=tmp_path) # Run from tmp_path

    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
    
    # captured = capsys.readouterr() # Not needed if using result.stdout
    # manifest_data = json.loads(captured.out)
    manifest_data = json.loads(result.stdout)
    assert_manifest_success(manifest_data, test_text, wav_output_parent_dir)
    # Cleanup the created WAV file
    if manifest_data.get("wav_path"):
        Path(manifest_data["wav_path"]).unlink(missing_ok=True)
        # Attempt to remove the directory if it's empty
        try:
            wav_output_parent_dir.rmdir()
        except OSError:
            pass # Directory not empty or other error, fine for test


def test_piper_tts_failure_empty_text_to_file(tmp_path: Path):
    """Test failure when --text is empty, writing manifest to file."""
    manifest_file = tmp_path / "failure_output.json"
    test_model_data_dir = tmp_path / "test_piper_models_empty_text"
    
    config_file = create_test_config_file(
        tmp_path, 
        step_id="tts_run", 
        piper_model_name=DEFAULT_TEST_MODEL,
        global_model_dir=str(test_model_data_dir)
    )
    
    cmd_args = [
        "--text", "",
        "--out", str(manifest_file),
        "--config", str(config_file)
    ]
    result = run_script(cmd_args)
    
    assert result.returncode != 0, "Script should exit with non-zero for empty text."
    assert manifest_file.exists()
    manifest_data = json.loads(manifest_file.read_text())
    assert_manifest_failure(manifest_data, "")


def test_piper_tts_failure_empty_text_to_stdout(tmp_path: Path, capsys):
    """Test failure when --text is empty, printing manifest to stdout."""
    test_model_data_dir = tmp_path / "test_piper_models_empty_stdout"
    config_file = create_test_config_file(
        tmp_path, 
        step_id="tts_run", 
        piper_model_name=DEFAULT_TEST_MODEL,
        global_model_dir=str(test_model_data_dir)
    )

    cmd_args = [
        "--text", "",
        "--config", str(config_file)
    ]
    result = run_script(cmd_args, cwd=tmp_path)

    assert result.returncode != 0, "Script should exit with non-zero for empty text."
    # captured = capsys.readouterr()
    # manifest_data = json.loads(captured.out)
    manifest_data = json.loads(result.stdout)
    assert_manifest_failure(manifest_data, "")


def test_piper_tts_failure_missing_config_file(tmp_path: Path):
    """Test failure when the config file is missing."""
    manifest_file = tmp_path / "failure_config_missing.json"
    cmd_args = [
        "--text", "test",
        "--out", str(manifest_file),
        "--config", str(tmp_path / "non_existent_config.yaml")
    ]
    result = run_script(cmd_args)

    assert result.returncode != 0
    assert manifest_file.exists() # Script should still produce a failure manifest
    manifest_data = json.loads(manifest_file.read_text())
    assert_manifest_failure(manifest_data, "test") # Text is known
    assert "Error: Config file not found" in result.stderr


def test_piper_tts_failure_missing_piper_model_in_config(tmp_path: Path):
    """Test failure when piper_model is missing in config parameters."""
    manifest_file = tmp_path / "failure_model_missing.json"
    test_model_data_dir = tmp_path / "test_piper_models_missing_param"
    # Create config where 'piper_model' key is absent from parameters
    config_file = create_test_config_file(
        tmp_path, 
        step_id="tts_run", 
        piper_model_name=None, # This will omit piper_model from params
        global_model_dir=str(test_model_data_dir),
        params_dict={} # Explicitly empty params dict
    )
    
    cmd_args = [
        "--text", "test",
        "--out", str(manifest_file),
        "--config", str(config_file)
    ]
    result = run_script(cmd_args)

    assert result.returncode != 0
    assert manifest_file.exists()
    manifest_data = json.loads(manifest_file.read_text())
    assert_manifest_failure(manifest_data, "test")
    assert "'piper_model' not found in parameters" in result.stderr


def test_piper_tts_failure_missing_parameters_key_in_config(tmp_path: Path):
    """Test failure when 'parameters' key is missing for the step in config."""
    manifest_file = tmp_path / "failure_params_key_missing.json"
    test_model_data_dir = tmp_path / "test_piper_models_missing_params_key"
    # Create config where the step has no 'parameters' key
    config_file = create_test_config_file(
        tmp_path, 
        step_id="tts_run", 
        piper_model_name=None, 
        global_model_dir=str(test_model_data_dir),
        params_dict=None # This will lead to no 'parameters' key for the step
    )

    cmd_args = [
        "--text", "test",
        "--out", str(manifest_file),
        "--config", str(config_file)
    ]
    result = run_script(cmd_args)

    assert result.returncode != 0
    assert manifest_file.exists()
    manifest_data = json.loads(manifest_file.read_text())
    assert_manifest_failure(manifest_data, "test")
    assert "'piper_model' not found in parameters" in result.stderr # Same error as piper_model key missing


def test_piper_tts_failure_invalid_config_yaml_format(tmp_path: Path):
    """Test failure with a malformed YAML config file."""
    manifest_file = tmp_path / "failure_invalid_yaml.json"
    test_model_data_dir = tmp_path / "test_piper_models_invalid_yaml" # Still need a model dir for config structure
    config_file_path = tmp_path / "invalid.yaml"
    # Write malformed YAML, but include model_dir to satisfy initial parsing before error
    config_file_path.write_text(f"model_dir: {str(test_model_data_dir)}\nsteps: \n  - id: tts_run\n    parameters: [piper_model: 'foo'")

    cmd_args = [
        "--text", "test",
        "--out", str(manifest_file),
        "--config", str(config_file_path)
    ]
    result = run_script(cmd_args)
    
    assert result.returncode != 0
    assert manifest_file.exists()
    manifest_data = json.loads(manifest_file.read_text())
    assert_manifest_failure(manifest_data, "test")
    assert "Error parsing YAML file" in result.stderr

@pytest.mark.skipif(not Path("/usr/bin/espeak-ng").exists() and not Path("/usr/local/bin/espeak-ng").exists(), reason="espeak-ng not found, piper-tts may fail to download/use models")
def test_piper_tts_failure_bad_model_name(tmp_path: Path):
    """Test failure when a non-existent/bad Piper model name is provided."""
    test_text = "Testing with a bad model name."
    manifest_file = tmp_path / "bad_model_output.json"
    test_model_data_dir = tmp_path / "test_piper_models_bad_name"
    # Use a model name that is unlikely to exist or be downloadable
    bad_model_name = "non_existent_model_for_testing_12345"
    config_file = create_test_config_file(
        tmp_path, 
        step_id="tts_run", 
        piper_model_name=bad_model_name,
        global_model_dir=str(test_model_data_dir)
    )

    cmd_args = [
        "--text", test_text,
        "--out", str(manifest_file),
        "--config", str(config_file)
    ]
    result = run_script(cmd_args)

    assert result.returncode != 0, f"Script should fail for bad model. Stderr: {result.stderr}"
    assert manifest_file.exists()
    
    manifest_data = json.loads(manifest_file.read_text())
    assert_manifest_failure(manifest_data, test_text)
    # Check for a more specific error.
    # Based on the new logic, it should be "Piper model ONNX file not found" or "Piper model JSON config file not found"
    # if ensure_voice_exists doesn't throw a more direct error that PiperVoice.load would have.
    assert "Error: Piper model ONNX file not found" in result.stderr or \
           "Error: Piper model JSON config file not found" in result.stderr or \
           "Error: Piper TTS runtime error" in result.stderr or \
           "ailed to download" in result.stderr or \
           "An unexpected error occurred during Piper TTS synthesis" in result.stderr


def test_piper_tts_failure_missing_global_model_dir(tmp_path: Path):
    """Test failure when the global 'model_dir' is missing in config."""
    manifest_file = tmp_path / "failure_global_model_dir_missing.json"
    # Create config where 'model_dir' top-level key is absent
    config_file = create_test_config_file(
        tmp_path,
        step_id="tts_run",
        piper_model_name=DEFAULT_TEST_MODEL,
        global_model_dir=None, # This will omit the global_model_dir
        params_dict={"piper_model": DEFAULT_TEST_MODEL} # Ensure step params are otherwise fine
    )
    
    cmd_args = [
        "--text", "test",
        "--out", str(manifest_file),
        "--config", str(config_file)
    ]
    result = run_script(cmd_args)

    assert result.returncode != 0
    assert manifest_file.exists()
    manifest_data = json.loads(manifest_file.read_text())
    assert_manifest_failure(manifest_data, "test")
    assert "Global 'model_dir' not found" in result.stderr
