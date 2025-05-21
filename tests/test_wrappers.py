import pytest
import json
from unittest.mock import Mock, patch
import subprocess # Added import
import os # Added import
from core.wrappers import piper_tts, orpheus_tts # Added orpheus_tts

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

# --- Tests for orpheus_tts ---

@pytest.fixture
def sample_transcript_json_data():
    """Provides a sample transcript JSON data for Orpheus tests."""
    return {
        "title": "Test Presentation",
        "slides": [
            {
                "index": 0,
                "title": "Slide 1 Title",
                "segments": [
                    {"kind": "text", "text": "Hello world.", "continue": []},
                    {"kind": "text", "text": "This is the first slide.", "continue": []}
                ]
            },
            {
                "index": 1,
                "title": "Slide 2 Title",
                "segments": [
                    {"kind": "text", "text": "Second slide content.", "continue": []}
                ]
            }
        ]
    }

def test_wrapper_orpheus_tts_success(mock_subprocess, tmp_path, sample_transcript_json_data, capsys):
    """Test successful Orpheus TTS wrapper execution."""
    output_file = tmp_path / "orpheus_manifest.json"

    # Mock for transcript_preprocess.py
    mock_preprocessed_data = {
        "slides": [
            {"index": 0, "merged_text": "Hello world. This is the first slide.", "tts_texts": ["Hello world. This is the first slide."]},
            {"index": 1, "merged_text": "Second slide content.", "tts_texts": ["Second slide content."]}
        ],
        "preprocessing_applied": {"chunk_mode": "slide"}
    }
    preprocess_mock_orpheus = Mock(spec=subprocess.CompletedProcess)
    preprocess_mock_orpheus.stdout = json.dumps(mock_preprocessed_data)
    preprocess_mock_orpheus.stderr = ""
    preprocess_mock_orpheus.returncode = 0

    # Mock for orpheus_tts_cli.py
    mock_orpheus_cli_output = [
        {"text": "Hello world. This is the first slide.", "wav_path": "tts_audio_output/orpheus_abc.wav", "status": "success", "duration": 2.5, "pipeline": "orpheus"},
        {"text": "Second slide content.", "wav_path": "tts_audio_output/orpheus_def.wav", "status": "success", "duration": 1.8, "pipeline": "orpheus"}
    ]
    orpheus_cli_mock = Mock(spec=subprocess.CompletedProcess)
    orpheus_cli_mock.stdout = json.dumps(mock_orpheus_cli_output)
    orpheus_cli_mock.stderr = ""
    orpheus_cli_mock.returncode = 0
    
    mock_subprocess.side_effect = [preprocess_mock_orpheus, orpheus_cli_mock]

    result = orpheus_tts(
        input_json_data=sample_transcript_json_data,
        output_path=str(output_file),
        step_id="test_orpheus_success", # Default is "tts_run"
        config_path="config/pipeline.yaml" 
    )

    assert mock_subprocess.call_count == 2

    # Check transcript_preprocess.py call
    preprocess_call_args = mock_subprocess.call_args_list[0][0][0]
    assert 'cli/transcript_preprocess.py' in preprocess_call_args
    assert '--input_file' in preprocess_call_args 
    assert '--chunk_mode' in preprocess_call_args
    assert 'slide' in preprocess_call_args

    # Check orpheus_tts_cli.py call
    orpheus_cli_call_args = mock_subprocess.call_args_list[1][0][0]
    assert 'cli/orpheus_tts_cli.py' in orpheus_cli_call_args
    assert '--texts-file' in orpheus_cli_call_args 
    assert '--config' in orpheus_cli_call_args
    assert 'config/pipeline.yaml' in orpheus_cli_call_args
    assert '--step_id' in orpheus_cli_call_args
    assert 'test_orpheus_success' in orpheus_cli_call_args
    
    assert result["source_transcript_data_preview"]["title"] == "Test Presentation"
    assert result["preprocessing_details"]["chunk_mode"] == "slide"
    assert result["number_of_tts_segments_processed"] == 2
    assert len(result["tts_results"]) == 2
    assert result["tts_results"][0]["text"] == "Hello world. This is the first slide."
    assert result["step_id"] == "test_orpheus_success"

    assert output_file.exists()
    with open(output_file, 'r') as f:
        content_on_disk = json.load(f)
    assert content_on_disk["step_id"] == "test_orpheus_success"
    assert len(content_on_disk["tts_results"]) == 2


def test_wrapper_orpheus_tts_cli_error(mock_subprocess, tmp_path, sample_transcript_json_data, capsys):
    """Test Orpheus TTS wrapper when orpheus_tts_cli.py fails."""
    mock_preprocessed_data = {
        "slides": [{"merged_text": "Some text for TTS.", "tts_texts": ["Some text for TTS."]}],
        "preprocessing_applied": {"chunk_mode": "slide"}
    }
    preprocess_mock_orpheus = Mock(spec=subprocess.CompletedProcess)
    preprocess_mock_orpheus.stdout = json.dumps(mock_preprocessed_data)
    preprocess_mock_orpheus.stderr = ""
    preprocess_mock_orpheus.returncode = 0

    orpheus_cli_error = subprocess.CalledProcessError(
        returncode=1,
        cmd=['cli/orpheus_tts_cli.py'],
        stderr="Orpheus CLI simulated error message."
    )
    
    mock_subprocess.side_effect = [preprocess_mock_orpheus, orpheus_cli_error]

    with pytest.raises(RuntimeError) as excinfo:
        orpheus_tts(
            input_json_data=sample_transcript_json_data,
            step_id="test_orpheus_cli_fail" # Default is "tts_run"
        )
    
    assert "Orpheus TTS CLI failed" in str(excinfo.value)
    assert "Orpheus CLI simulated error message." in str(excinfo.value)
    assert mock_subprocess.call_count == 2


def test_wrapper_orpheus_tts_preprocess_error(mock_subprocess, tmp_path, sample_transcript_json_data, capsys):
    """Test Orpheus TTS wrapper when transcript_preprocess.py fails."""
    preprocess_error = subprocess.CalledProcessError(
        returncode=1,
        cmd=['cli/transcript_preprocess.py'],
        stderr="Preprocessor simulated error."
    )
    
    mock_subprocess.side_effect = [preprocess_error]

    with pytest.raises(RuntimeError) as excinfo:
        orpheus_tts(
            input_json_data=sample_transcript_json_data,
            step_id="test_orpheus_preprocess_fail" # Default is "tts_run"
        )
    
    assert "Transcript preprocessing failed" in str(excinfo.value)
    assert "Preprocessor simulated error." in str(excinfo.value)
    assert mock_subprocess.call_count == 1


def test_wrapper_orpheus_tts_empty_texts_from_preprocess(mock_subprocess, tmp_path, sample_transcript_json_data, capsys):
    """Test Orpheus TTS wrapper when preprocessing results in no texts (e.g. only whitespace)."""
    mock_preprocessed_data_empty = {
        "slides": [
            {"index": 0, "merged_text": " ", "tts_texts": [" "]}, 
            {"index": 1, "merged_text": "", "tts_texts": [""]}    
        ],
        "preprocessing_applied": {"chunk_mode": "slide"}
    }
    preprocess_mock_empty = Mock(spec=subprocess.CompletedProcess)
    preprocess_mock_empty.stdout = json.dumps(mock_preprocessed_data_empty)
    preprocess_mock_empty.stderr = ""
    preprocess_mock_empty.returncode = 0

    # Orpheus CLI with --texts-file and an empty list should return an empty list JSON `[]` and exit 0
    mock_orpheus_cli_empty_output = [] 
    orpheus_cli_mock_empty = Mock(spec=subprocess.CompletedProcess)
    orpheus_cli_mock_empty.stdout = json.dumps(mock_orpheus_cli_empty_output)
    orpheus_cli_mock_empty.stderr = ""
    orpheus_cli_mock_empty.returncode = 0 # Orpheus CLI should succeed with empty list
    
    mock_subprocess.side_effect = [preprocess_mock_empty, orpheus_cli_mock_empty]

    result = orpheus_tts(
        input_json_data=sample_transcript_json_data, 
        step_id="test_orpheus_empty_texts" # Default is "tts_run"
    )

    assert mock_subprocess.call_count == 2
    assert result["number_of_tts_segments_processed"] == 0 # Wrapper filters out empty/whitespace strings
    assert len(result["tts_results"]) == 0 # Orpheus CLI mock returns empty list
    assert result["step_id"] == "test_orpheus_empty_texts"
