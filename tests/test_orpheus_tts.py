import json
import subprocess
import sys
from pathlib import Path
import os # Added for environment manipulation

import pytest
import yaml

# Define the path to the CLI script
ORPHEUS_TTS_CLI = Path(__file__).parent.parent / "cli" / "orpheus_tts_cli.py"

# Default values from config/pipeline.yaml for orpheus
DEFAULT_CONFIG_VOICE = "dan"
DEFAULT_CONFIG_TEMPERATURE = 0.2
DEFAULT_MODEL_NAME = "canopylabs/orpheus-tts-0.1-finetune-prod"

# Argparse defaults
ARGPARSE_DEFAULT_VOICE = "dan"
ARGPARSE_DEFAULT_TEMP = 0.4


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
    config_file = tmp_path / "test_pipeline.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)
    return config_file

@pytest.fixture
def minimal_config_file(tmp_path: Path) -> Path:
    """Creates a temporary config file with only orpheus_model."""
    config_content = {
        "model_dir": str(tmp_path / "models"),
        "steps": [
            {
                "id": "tts_run",
                "parameters": {
                    "orpheus_model": DEFAULT_MODEL_NAME,
                    # No orpheus_voice or orpheus_temperature
                },
            }
        ],
    }
    config_file = tmp_path / "minimal_pipeline.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)
    return config_file


def run_cli(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Helper to run the Orpheus TTS CLI script."""
    command = [sys.executable, str(ORPHEUS_TTS_CLI)] + args
    timeout_seconds = 300 # 5 minutes, increased like in batch tests

    # Create a copy of the current environment
    env = os.environ.copy()
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # Ensure consistent GPU ordering
    env['CUDA_VISIBLE_DEVICES'] = '1' # Attempt to use GPU 1
    
    # Path to the directory containing the 'orpheus_tts' package for direct import
    # This is the directory where setup.py is located and where find_packages() would look.
    orpheus_tts_source_dir = str(Path(__file__).parent.parent / "third_party" / "Orpheus-TTS" / "orpheus_tts_pypi")
    
    # Prepend this path to PYTHONPATH
    # This helps Python find the 'orpheus_tts' module located inside 'orpheus_tts_source_dir'
    # current_python_path = env.get('PYTHONPATH', '') # Keep existing PYTHONPATH
    # if current_python_path:
    #     env['PYTHONPATH'] = f"{orpheus_tts_source_dir}{os.pathsep}{current_python_path}"
    # else:
    #     env['PYTHONPATH'] = orpheus_tts_source_dir
    
    # Try setting PYTHONPATH directly to only this path to simplify debugging
    env['PYTHONPATH'] = orpheus_tts_source_dir

    # For debugging test environment
    # print(f"CLI command: {' '.join(command)}")
    # print(f"PYTHONPATH for subprocess: {env['PYTHONPATH']}")

    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_seconds, env=env, **kwargs)


def assert_manifest_success(
    manifest: dict,
    expected_text: str,
    expected_wav_dir: Path | None = None,
    expected_voice: str | None = None,
    expected_temp: float | None = None,
):
    """Asserts common success conditions for a manifest."""
    assert manifest["status"] == "success"
    assert manifest["text"] == expected_text
    assert manifest["pipeline"] == "orpheus"
    assert isinstance(manifest["duration"], float)
    assert manifest["duration"] > 0  # Should have some duration

    wav_path_str = manifest["wav_path"]
    assert wav_path_str is not None
    wav_path = Path(wav_path_str)
    assert wav_path.exists()
    assert wav_path.is_file()
    assert wav_path.suffix == ".wav"
    assert wav_path.stat().st_size > 1000  # Basic check for non-empty WAV

    if expected_wav_dir:
        assert wav_path.parent.resolve() == expected_wav_dir.resolve()

    # These checks are harder as they are not in the manifest.
    # We rely on logs or specific test setups if we need to verify voice/temp used.


def test_orpheus_tts_success_file_output(tmp_path: Path, default_config_file: Path):
    """Test successful TTS to a file with default config values."""
    text_to_synth = "Hello from Orpheus, this is a file output test."
    manifest_file = tmp_path / "output.json"

    args = [
        "--text", text_to_synth,
        "--out", str(manifest_file),
        "--config", str(default_config_file),
    ]
    result = run_cli(args)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"CLI exited with {result.returncode}"

    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert_manifest_success(manifest, text_to_synth, expected_wav_dir=tmp_path)
    # Voice and temp should be from default_config_file (dan, 0.2)


def test_orpheus_tts_success_stdout_output(tmp_path: Path, default_config_file: Path):
    """Test successful TTS to stdout with default config values."""
    text_to_synth = "Hello Orpheus, this is a stdout test."
    args = [
        "--text", text_to_synth,
        "--config", str(default_config_file),
    ]
    result = run_cli(args)

    print("STDOUT (raw):", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"CLI exited with {result.returncode}"

    # Extract JSON part from stdout, as vLLM logs to stdout as well
    stdout_content = result.stdout
    # The actual JSON manifest starts with '{\n  "text": ...' due to pretty printing
    json_start_index = stdout_content.find('{\n  "text":')
    assert json_start_index != -1, "Could not find start of JSON manifest in stdout"
    manifest_json_str = stdout_content[json_start_index:]
    
    manifest = json.loads(manifest_json_str)
    # Default output dir is cwd / "tts_audio_output"
    expected_wav_parent_dir = Path.cwd() / "tts_audio_output"
    assert_manifest_success(manifest, text_to_synth, expected_wav_dir=expected_wav_parent_dir)


def test_orpheus_tts_override_voice_temp_cli(tmp_path: Path, default_config_file: Path):
    """Test overriding voice and temperature via CLI arguments."""
    text_to_synth = "Testing CLI override for voice and temperature."
    manifest_file = tmp_path / "override_output.json"
    cli_voice = "susan" # Assuming 'susan' is a valid voice for the model
    cli_temp = 0.88

    args = [
        "--text", text_to_synth,
        "--out", str(manifest_file),
        "--config", str(default_config_file),
        "--voice", cli_voice,
        "--temperature", str(cli_temp),
    ]
    result = run_cli(args)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"CLI exited with {result.returncode}"

    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert_manifest_success(manifest, text_to_synth, expected_wav_dir=tmp_path)
    # Here, we'd ideally check logs to confirm cli_voice and cli_temp were used.
    # For now, success implies they were accepted.


def test_orpheus_tts_minimal_config_uses_argparse_defaults(tmp_path: Path, minimal_config_file: Path):
    """Test that argparse defaults for voice/temp are used if not in config."""
    text_to_synth = "Testing with minimal config, relying on argparse defaults for voice and temp."
    manifest_file = tmp_path / "minimal_cfg_output.json"

    args = [
        "--text", text_to_synth,
        "--out", str(manifest_file),
        "--config", str(minimal_config_file),
        # No --voice or --temperature, so argparse defaults (dan, 0.4) should be used
        # as minimal_config_file doesn't provide them.
    ]
    result = run_cli(args)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"CLI exited with {result.returncode}"

    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert_manifest_success(manifest, text_to_synth, expected_wav_dir=tmp_path)
    # We expect voice=ARGPARSE_DEFAULT_VOICE ("dan") and temp=ARGPARSE_DEFAULT_TEMP (0.4)
    # This is harder to verify without log parsing or more direct output.


def test_orpheus_tts_failure_no_text(tmp_path: Path, default_config_file: Path):
    """Test CLI failure when --text is missing."""
    manifest_file = tmp_path / "failure_no_text.json"
    args = [
        "--out", str(manifest_file),
        "--config", str(default_config_file),
    ] # Missing --text
    result = run_cli(args)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    # Argparse itself should cause a non-zero exit for missing required arg
    assert result.returncode != 0

    # The script might not even reach manifest generation if argparse exits early.
    # The script should now exit due to "Either --text or --texts-file must be provided."
    # and it won't create a manifest file in this specific scenario.
    assert not manifest_file.exists(), "Manifest file should not be created when --text is missing."
    assert "Either --text or --texts-file must be provided".lower() in result.stderr.lower()


def test_orpheus_tts_failure_empty_text(tmp_path: Path, default_config_file: Path):
    """Test CLI failure when --text is empty."""
    manifest_file = tmp_path / "failure_empty_text.json"
    args = [
        "--text", "",
        "--out", str(manifest_file),
        "--config", str(default_config_file),
    ]
    result = run_cli(args)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode != 0, "CLI should exit with non-zero for empty text"

    # With the corrected CLI logic, a manifest file should be created for an empty --text input.
    assert manifest_file.exists(), "Manifest file should be created for empty text failure"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["status"] == "failure"
    assert manifest["text"] == ""
    assert manifest["wav_path"] is None
    assert manifest["duration"] == 0.0
    assert "error" in manifest # Check for the error field in manifest
    assert "Input text is empty".lower() in manifest["error"].lower()
    # Also check stderr for the logged error
    assert "Input text (--text) cannot be empty".lower() in result.stderr.lower()


def test_orpheus_tts_failure_missing_model_in_config(tmp_path: Path):
    """Test CLI failure if orpheus_model is missing in config."""
    config_content = {
        "steps": [{"id": "tts_run", "parameters": {}}] # No orpheus_model
    }
    bad_config_file = tmp_path / "bad_config.yaml"
    with open(bad_config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f)

    manifest_file = tmp_path / "failure_missing_model.json"
    args = [
        "--text", "test",
        "--out", str(manifest_file),
        "--config", str(bad_config_file),
    ]
    result = run_cli(args)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode != 0

    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["status"] == "failure"
    assert "Orpheus model name ('orpheus_model') not found" in result.stderr
