import pytest
import json
import tempfile
import pathlib
import os
import sys
from unittest.mock import patch, MagicMock

from core.wrappers import whisper_transcribe

@pytest.fixture
def sample_wav_files(tmp_path: pathlib.Path) -> list[pathlib.Path]:
    """Creates dummy WAV files in a temporary directory and returns their paths."""
    audio_dir = tmp_path / "audio_data"
    audio_dir.mkdir()
    wav1 = audio_dir / "audio1.wav"
    wav2 = audio_dir / "audio2.wav"
    wav1.touch()
    wav2.touch()
    return [wav1.resolve(), wav2.resolve()]

@pytest.fixture
def mock_subprocess_run():
    """Mocks subprocess.run to simulate cli/transcribe_whisperx.py execution."""
    with patch('core.wrappers.subprocess.run') as mock_run:
        yield mock_run

def configure_mock_subprocess(mock_run_method, success=True, error_message=""):
    """Helper to configure the subprocess.run mock's behavior."""
    def side_effect_func(cmd, *args, **kwargs):
        # Expected cmd structure:
        # [sys.executable, script_path, "--in", tmp_manifest_path, "--model", model, 
        #  "--out", tmp_output_path, "--batch_size", str_batch_size, 
        #  "--compute_type", compute_type, "--device", device]
        
        script_path_idx = cmd.index("--in") -1
        assert "transcribe_whisperx.py" in cmd[script_path_idx]

        tmp_manifest_path = cmd[cmd.index("--in") + 1]
        tmp_output_path = cmd[cmd.index("--out") + 1]

        if not success:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = error_message
            mock_result.stdout = "Simulated stdout on error"
            # Simulate script creating an empty or error-indicating output file
            with open(tmp_output_path, 'w', encoding='utf-8') as f:
                json.dump([{"error": "script_failed"}], f)
            return mock_result

        # Simulate successful execution: read manifest, write mock results to output
        try:
            with open(tmp_manifest_path, 'r', encoding='utf-8') as f_manifest:
                audio_paths_from_manifest = json.load(f_manifest)
        except FileNotFoundError:
             # This can happen if the manifest path itself is bad, core.wrappers should catch this before subprocess
            pytest.fail(f"Mock subprocess: Manifest file {tmp_manifest_path} not found by mock.")
        except Exception as e:
            pytest.fail(f"Mock subprocess: Error reading manifest {tmp_manifest_path}: {e}")


        mock_results_list = []
        for i, audio_path_str in enumerate(audio_paths_from_manifest):
            # Ensure paths in mock output are absolute, as the script does
            abs_audio_path = str(pathlib.Path(audio_path_str).resolve())
            mock_results_list.append({
                "wav": abs_audio_path,
                "asr": f"mock transcript for {pathlib.Path(abs_audio_path).name}"
            })
        
        try:
            with open(tmp_output_path, 'w', encoding='utf-8') as f_out:
                json.dump(mock_results_list, f_out, indent=2)
        except Exception as e:
            pytest.fail(f"Mock subprocess: Error writing to output file {tmp_output_path}: {e}")


        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = "Simulated successful stdout"
        return mock_result

    mock_run_method.side_effect = side_effect_func


# --- Test Cases ---

def test_input_json_manifest(mock_subprocess_run, sample_wav_files, tmp_path):
    """Test with input as a path to a JSON manifest file."""
    configure_mock_subprocess(mock_subprocess_run)
    
    manifest_content = [str(p) for p in sample_wav_files]
    manifest_file = tmp_path / "test_manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest_content, f)

    result = whisper_transcribe(str(manifest_file))
    
    assert len(result) == len(sample_wav_files)
    for wav_path_obj in sample_wav_files:
        abs_wav_path_str = str(wav_path_obj)
        assert abs_wav_path_str in result
        assert result[abs_wav_path_str] == f"mock transcript for {wav_path_obj.name}"
    
    mock_subprocess_run.assert_called_once()
    call_args = mock_subprocess_run.call_args[0][0] # Get the 'cmd' list
    assert "--model" in call_args
    assert call_args[call_args.index("--model") + 1] == "large-v3" # Default model


def test_input_txt_manifest(mock_subprocess_run, sample_wav_files, tmp_path):
    """Test with input as a path to a TXT manifest file."""
    configure_mock_subprocess(mock_subprocess_run)
    
    manifest_content = "\n".join([str(p) for p in sample_wav_files])
    manifest_file = tmp_path / "test_manifest.txt"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        f.write(manifest_content)

    result = whisper_transcribe(str(manifest_file))
    
    assert len(result) == len(sample_wav_files)
    for wav_path_obj in sample_wav_files:
        abs_wav_path_str = str(wav_path_obj)
        assert abs_wav_path_str in result
        assert result[abs_wav_path_str] == f"mock transcript for {wav_path_obj.name}"
    mock_subprocess_run.assert_called_once()


def test_input_list_of_dicts(mock_subprocess_run, sample_wav_files):
    """Test with input as a list of dictionaries."""
    configure_mock_subprocess(mock_subprocess_run)
    
    input_data = [{"wav_path": str(p)} for p in sample_wav_files]
    result = whisper_transcribe(input_data)

    assert len(result) == len(sample_wav_files)
    for wav_path_obj in sample_wav_files:
        abs_wav_path_str = str(wav_path_obj)
        assert abs_wav_path_str in result
        assert result[abs_wav_path_str] == f"mock transcript for {wav_path_obj.name}"
    mock_subprocess_run.assert_called_once()

def test_input_list_of_strings(mock_subprocess_run, sample_wav_files):
    """Test with input as a list of string paths."""
    configure_mock_subprocess(mock_subprocess_run)

    input_data = [str(p) for p in sample_wav_files] # Absolute paths
    result = whisper_transcribe(input_data)

    assert len(result) == len(sample_wav_files)
    for wav_path_obj in sample_wav_files:
        abs_wav_path_str = str(wav_path_obj)
        assert abs_wav_path_str in result
        assert result[abs_wav_path_str] == f"mock transcript for {wav_path_obj.name}"
    mock_subprocess_run.assert_called_once()

def test_input_relative_paths_in_list_of_strings(mock_subprocess_run, tmp_path, monkeypatch):
    """Test with relative paths in a list of strings, checking they are resolved."""
    configure_mock_subprocess(mock_subprocess_run)
    
    # Create dummy files in tmp_path, which will be CWD for this test
    relative_wav1 = "relative_audio1.wav"
    relative_wav2 = "relative_audio2.wav"
    (tmp_path / relative_wav1).touch()
    (tmp_path / relative_wav2).touch()
    
    expected_abs_paths = [
        (tmp_path / relative_wav1).resolve(),
        (tmp_path / relative_wav2).resolve()
    ]

    monkeypatch.chdir(tmp_path) # Change CWD to tmp_path
    
    input_data = [relative_wav1, relative_wav2]
    result = whisper_transcribe(input_data)

    assert len(result) == len(expected_abs_paths)
    for abs_path_obj in expected_abs_paths:
        assert str(abs_path_obj) in result
        assert result[str(abs_path_obj)] == f"mock transcript for {abs_path_obj.name}"
    mock_subprocess_run.assert_called_once()

def test_input_relative_paths_in_json_manifest(mock_subprocess_run, tmp_path, monkeypatch):
    """Test with relative paths in a JSON manifest, resolved relative to manifest dir."""
    configure_mock_subprocess(mock_subprocess_run)

    manifest_dir = tmp_path / "manifest_subdir"
    manifest_dir.mkdir()
    
    audio_dir = manifest_dir / "audio_files" # Audio files relative to manifest_dir
    audio_dir.mkdir()

    relative_wav1 = "audio_files/rel_audio1.wav" # Path relative to manifest_dir
    relative_wav2 = "audio_files/rel_audio2.wav"
    
    (manifest_dir / relative_wav1).touch()
    (manifest_dir / relative_wav2).touch()

    expected_abs_paths = [
        (manifest_dir / relative_wav1).resolve(),
        (manifest_dir / relative_wav2).resolve()
    ]

    manifest_content = [relative_wav1, relative_wav2] # Paths as they appear in manifest
    manifest_file = manifest_dir / "relative_manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest_content, f)

    # Call whisper_transcribe with the path to the manifest
    # The CWD doesn't matter here as paths in manifest are relative to manifest's dir
    result = whisper_transcribe(str(manifest_file))

    assert len(result) == len(expected_abs_paths)
    for abs_path_obj in expected_abs_paths:
        assert str(abs_path_obj) in result
        assert result[str(abs_path_obj)] == f"mock transcript for {abs_path_obj.name}"
    mock_subprocess_run.assert_called_once()


def test_subprocess_error_raises_runtimeerror(mock_subprocess_run, sample_wav_files):
    """Test that a non-zero exit code from subprocess raises RuntimeError."""
    error_msg = "Simulated subprocess failure"
    configure_mock_subprocess(mock_subprocess_run, success=False, error_message=error_msg)
    
    input_data = [str(p) for p in sample_wav_files]
    
    with pytest.raises(RuntimeError) as excinfo:
        whisper_transcribe(input_data)
    
    assert "cli/transcribe_whisperx.py failed" in str(excinfo.value)
    assert f"Stderr: {error_msg}" in str(excinfo.value)
    mock_subprocess_run.assert_called_once()

def test_empty_input_list_returns_empty_dict(mock_subprocess_run):
    """Test that an empty list as input results in an empty dict and no subprocess call."""
    result = whisper_transcribe([])
    assert result == {}
    mock_subprocess_run.assert_not_called()

def test_manifest_file_not_found_raises_fileerror():
    """Test that a non-existent manifest path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        whisper_transcribe("non_existent_manifest.json")

def test_empty_json_manifest_file_returns_empty_dict(mock_subprocess_run, tmp_path):
    """Test with an empty JSON manifest file ([]) results in an empty dict."""
    manifest_file = tmp_path / "empty_manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump([], f)
    
    result = whisper_transcribe(str(manifest_file))
    assert result == {}
    mock_subprocess_run.assert_not_called()

def test_empty_txt_manifest_file_returns_empty_dict(mock_subprocess_run, tmp_path):
    """Test with an empty TXT manifest file results in an empty dict."""
    manifest_file = tmp_path / "empty_manifest.txt"
    manifest_file.touch() # Create an empty file
    
    result = whisper_transcribe(str(manifest_file))
    assert result == {}
    mock_subprocess_run.assert_not_called()

def test_non_default_parameters_passed_to_subprocess(mock_subprocess_run, sample_wav_files):
    """Test that non-default model, batch_size, etc., are passed to subprocess."""
    configure_mock_subprocess(mock_subprocess_run)
    
    custom_model = "base"
    custom_batch_size = 8
    custom_compute_type = "int8"
    custom_device = "cpu"

    input_data = [str(p) for p in sample_wav_files]
    whisper_transcribe(
        input_data,
        model=custom_model,
        batch_size=custom_batch_size,
        compute_type=custom_compute_type,
        device=custom_device
    )

    mock_subprocess_run.assert_called_once()
    call_args = mock_subprocess_run.call_args[0][0] # Get the 'cmd' list
    
    assert "--model" in call_args
    assert call_args[call_args.index("--model") + 1] == custom_model
    assert "--batch_size" in call_args
    assert call_args[call_args.index("--batch_size") + 1] == str(custom_batch_size)
    assert "--compute_type" in call_args
    assert call_args[call_args.index("--compute_type") + 1] == custom_compute_type
    assert "--device" in call_args
    assert call_args[call_args.index("--device") + 1] == custom_device

def test_input_mixed_list_types_raises_valueerror(mock_subprocess_run):
    """Test that a list with mixed types (dict and str) raises ValueError."""
    input_data = [{"wav_path": "path1.wav"}, "path2.wav"]
    # This will raise "Invalid item in list of dicts" when it encounters the string "path2.wav"
    # after expecting a dict. The original check for "Input list must contain either all dicts"
    # was based on the first item, which is a more complex check to implement perfectly for all
    # mixed-type scenarios without iterating twice or changing the core logic significantly.
    # The current behavior correctly identifies an invalid item.
    with pytest.raises(ValueError, match="Invalid item in list of dicts: must be a dict with 'wav_path'."):
        whisper_transcribe(input_data)
    mock_subprocess_run.assert_not_called()

def test_input_list_of_invalid_dicts_raises_valueerror(mock_subprocess_run):
    """Test that a list with invalid dicts (missing 'wav_path') raises ValueError."""
    input_data = [{"audio_file": "path1.wav"}]
    with pytest.raises(ValueError, match="Invalid item in list of dicts"):
        whisper_transcribe(input_data)
    mock_subprocess_run.assert_not_called()
    
def test_input_list_with_non_string_path_raises_valueerror(mock_subprocess_run):
    """Test that a list of strings containing a non-string item raises ValueError."""
    input_data = ["path1.wav", 123]
    with pytest.raises(ValueError, match="Invalid item in list of strings"):
        whisper_transcribe(input_data)
    mock_subprocess_run.assert_not_called()

def test_unsupported_manifest_type_raises_valueerror(mock_subprocess_run, tmp_path):
    """Test that an unsupported manifest file type (e.g., .xml) raises RuntimeError (wrapping ValueError)."""
    manifest_file = tmp_path / "manifest.xml"
    manifest_file.touch()
    with pytest.raises(RuntimeError, match="Error reading manifest file .* Unsupported manifest file type: .xml"):
        whisper_transcribe(str(manifest_file))
    mock_subprocess_run.assert_not_called()

def test_malformed_json_manifest_raises_runtimeerror(mock_subprocess_run, tmp_path):
    """Test that a malformed JSON manifest raises RuntimeError."""
    manifest_file = tmp_path / "malformed.json"
    with open(manifest_file, 'w') as f:
        f.write("{not_a_list: true}")
    
    with pytest.raises(RuntimeError, match="Error reading manifest file"):
        whisper_transcribe(str(manifest_file))
    mock_subprocess_run.assert_not_called()

def test_json_manifest_not_a_list_raises_runtimeerror(mock_subprocess_run, tmp_path):
    """Test JSON manifest that is valid JSON but not a list raises RuntimeError (via ValueError)."""
    manifest_file = tmp_path / "not_a_list.json"
    with open(manifest_file, 'w') as f:
        json.dump({"path": "file.wav"}, f) # A dict, not a list
    
    with pytest.raises(RuntimeError, match="JSON manifest must contain a list of paths"):
        whisper_transcribe(str(manifest_file))
    mock_subprocess_run.assert_not_called()

def test_script_produces_empty_or_no_output_file_runtimeerror(mock_subprocess_run, sample_wav_files):
    """Test RuntimeError if script exits 0 but output file is missing/empty."""
    def side_effect_missing_output(cmd, *args, **kwargs):
        # Simulate script not creating the output file or creating an empty one
        tmp_output_path = cmd[cmd.index("--out") + 1]
        # Option 1: Don't create the file
        # Option 2: Create an empty file
        pathlib.Path(tmp_output_path).touch() # Creates an empty file

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = "Simulated success but no output file content"
        return mock_result

    mock_subprocess_run.side_effect = side_effect_missing_output
    input_data = [str(p) for p in sample_wav_files]
    with pytest.raises(RuntimeError, match=r"completed successfully but output file .* is missing or empty"):
        whisper_transcribe(input_data)
    mock_subprocess_run.assert_called_once()

def test_script_produces_malformed_json_output_runtimeerror(mock_subprocess_run, sample_wav_files):
    """Test RuntimeError if script output file contains malformed JSON."""
    def side_effect_malformed_json(cmd, *args, **kwargs):
        tmp_output_path = cmd[cmd.index("--out") + 1]
        with open(tmp_output_path, 'w', encoding='utf-8') as f_out:
            f_out.write("[{'wav': 'file.wav', 'asr': 'text'}]") # Malformed: single quotes

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = "Simulated success but malformed JSON output"
        return mock_result

    mock_subprocess_run.side_effect = side_effect_malformed_json
    input_data = [str(p) for p in sample_wav_files]
    with pytest.raises(json.JSONDecodeError): # The json.load in whisper_transcribe will fail
        whisper_transcribe(input_data)
    mock_subprocess_run.assert_called_once()
