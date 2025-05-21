import pytest
import json
from unittest.mock import Mock, patch
import subprocess # Added import
import os # Added import
from core.wrappers import piper_tts

@pytest.fixture
def mock_subprocess(): # Removed mocker argument
    """Fixture to mock subprocess.run for CLI tools"""
    with patch('subprocess.run') as mock_run:
        # Mock successful transcript_preprocess response (default for test_piper_tts_success)
        preprocess_mock = Mock()
        preprocess_mock.stdout = json.dumps([{"text": "Test chunk 1"}, {"text": "Test chunk 2"}])
        preprocess_mock.check_returncode.return_value = None

        # Mock successful piper_tts responses (default for test_piper_tts_success)
        tts_mock1 = Mock()
        tts_mock1.stdout = json.dumps({"audio_path": "audio1.wav", "duration": 1.5})
        tts_mock1.check_returncode.return_value = None
        tts_mock2 = Mock()
        tts_mock2.stdout = json.dumps({"audio_path": "audio2.wav", "duration": 2.0})
        tts_mock2.check_returncode.return_value = None

        # Set up default side_effect. Tests can override this by re-assigning mock_run.side_effect.
        mock_run.side_effect = [preprocess_mock, tts_mock1, tts_mock2]
        yield mock_run

def test_piper_tts_success(mock_subprocess, tmp_path): # Changed from test_tts_pipeline_success
    """Test successful TTS pipeline execution with mocked CLI tools"""
    input_file = tmp_path / "input.json"
    # Create a dummy JSON structure for the input file, as piper_tts expects it
    input_file.write_text(json.dumps({"segments": [{"text": "dummy"}]}))
    output_file = tmp_path / "output.json"

    result = piper_tts(
        transcript_path=str(input_file),
        output_path=str(output_file),
        step_id="test_tts" # Using a specific step_id for clarity
    )

    # Verify CLI calls
    assert mock_subprocess.call_count == 3 # Preprocess + 2 TTS calls
    
    # Check preprocess call
    pre_cmd_args = mock_subprocess.call_args_list[0][0][0]
    assert 'cli/transcript_preprocess.py' in pre_cmd_args
    assert '--input' in pre_cmd_args
    assert str(input_file) in pre_cmd_args
    assert '--chunk_mode' in pre_cmd_args
    assert 'sentence' in pre_cmd_args
    assert '--step_id' in pre_cmd_args
    assert 'test_tts' in pre_cmd_args

    # Check first TTS call
    tts_cmd_args1 = mock_subprocess.call_args_list[1][0][0]
    assert 'cli/piper_tts.py' in tts_cmd_args1
    assert '--text' in tts_cmd_args1
    assert "Test chunk 1" in tts_cmd_args1 # From mocked preprocess_mock
    assert '--step_id' in tts_cmd_args1
    assert 'test_tts.chunk_0' in tts_cmd_args1
    
    # Check second TTS call
    tts_cmd_args2 = mock_subprocess.call_args_list[2][0][0]
    assert 'cli/piper_tts.py' in tts_cmd_args2
    assert '--text' in tts_cmd_args2
    assert "Test chunk 2" in tts_cmd_args2 # From mocked preprocess_mock
    assert '--step_id' in tts_cmd_args2
    assert 'test_tts.chunk_1' in tts_cmd_args2

    # Verify output structure from piper_tts function
    assert "source_transcript" in result
    assert result["source_transcript"] == str(input_file)
    assert "tts_results" in result
    assert len(result["tts_results"]) == 2
    assert result["tts_results"][0]["audio_path"] == "audio1.wav"
    assert result["tts_results"][1]["audio_path"] == "audio2.wav"
    assert result["step_id"] == "test_tts"

    # Verify output file content
    assert os.path.exists(output_file)
    with open(output_file) as f:
        file_content = json.load(f)
    assert file_content["step_id"] == "test_tts"
    assert len(file_content["tts_results"]) == 2

def test_piper_tts_cli_error(mock_subprocess, tmp_path): # Changed from test_tts_pipeline_cli_error
    """Test error handling when a CLI tool fails (piper_tts fails)"""
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"segments": [{"text": "dummy"}]}))

    # Mock successful transcript_preprocess response
    preprocess_mock = Mock()
    preprocess_mock.stdout = json.dumps([{"text": "Error chunk"}])
    preprocess_mock.check_returncode.return_value = None

    # Simulate error in piper_tts
    tts_error = subprocess.CalledProcessError(1, cmd=['cli/piper_tts.py'], stderr="TTS process error")
    
    # Configure side_effect for subprocess.run
    # First call (preprocess) succeeds, second call (tts) fails
    mock_subprocess.side_effect = [
        preprocess_mock,
        tts_error # This will be raised by subprocess.run
    ]
    
    with pytest.raises(RuntimeError) as excinfo:
        piper_tts(str(input_file))
    
    assert "TTS failed for chunk 0" in str(excinfo.value)
    assert "TTS process error" in str(excinfo.value)
    assert mock_subprocess.call_count == 2 # Preprocess (success) + TTS (failure)

def test_piper_tts_preprocess_cli_error(mock_subprocess, tmp_path):
    """Test error handling when transcript_preprocess.py CLI tool fails"""
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"segments": [{"text": "dummy"}]}))

    # Simulate error in transcript_preprocess.py
    preprocess_error = subprocess.CalledProcessError(1, cmd=['cli/transcript_preprocess.py'], stderr="Preprocessor error")
    mock_subprocess.side_effect = [preprocess_error]
    
    with pytest.raises(RuntimeError) as excinfo:
        piper_tts(str(input_file))
    
    assert "Transcript preprocessing failed" in str(excinfo.value)
    assert "Preprocessor error" in str(excinfo.value)
    assert mock_subprocess.call_count == 1

def test_piper_tts_invalid_json_from_preprocessor(mock_subprocess, tmp_path): # Changed from test_tts_pipeline_invalid_json
    """Test error handling for invalid JSON output from transcript_preprocess.py"""
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"segments": [{"text": "dummy"}]}))

    # Simulate invalid JSON from preprocessor
    preprocess_mock_invalid_json = Mock()
    preprocess_mock_invalid_json.stdout = "this is not valid json"
    preprocess_mock_invalid_json.check_returncode.return_value = None
    
    mock_subprocess.side_effect = [preprocess_mock_invalid_json]
    
    with pytest.raises(RuntimeError) as excinfo:
        piper_tts(str(input_file))
    
    assert "Invalid JSON from preprocessor" in str(excinfo.value)
    assert mock_subprocess.call_count == 1

def test_piper_tts_invalid_json_from_tts(mock_subprocess, tmp_path):
    """Test error handling for invalid JSON output from piper_tts.py"""
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"segments": [{"text": "dummy"}]}))

    # Mock successful transcript_preprocess response
    preprocess_mock = Mock()
    preprocess_mock.stdout = json.dumps([{"text": "Valid chunk"}])
    preprocess_mock.check_returncode.return_value = None

    # Simulate invalid JSON from piper_tts
    tts_mock_invalid_json = Mock()
    tts_mock_invalid_json.stdout = "not valid tts json"
    tts_mock_invalid_json.check_returncode.return_value = None
    
    mock_subprocess.side_effect = [preprocess_mock, tts_mock_invalid_json]
    
    with pytest.raises(RuntimeError) as excinfo:
        piper_tts(str(input_file))
    
    assert "Invalid TTS output for chunk 0" in str(excinfo.value)
    assert mock_subprocess.call_count == 2
