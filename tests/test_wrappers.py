import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock
from core.wrappers import check_datasets, extract_transcript

# Mock subprocess.run for check_datasets tests
@patch('core.wrappers.subprocess.run')
def test_check_datasets_success(mock_run):
    """Test check_datasets with successful subprocess execution."""
    mock_result = MagicMock()
    mock_result.stdout = "Datasets checked successfully"
    mock_run.return_value = mock_result

    data_dir = "/fake/data"
    output = check_datasets(data_dir)

    mock_run.assert_called_once_with(
        [sys.executable, 'cli/check_datasets.py', '--data', data_dir, '--step_id', 'check_datasets'],
        capture_output=True,
        text=True,
        check=True
    )
    assert output == "Datasets checked successfully"

@patch('core.wrappers.subprocess.run')
def test_check_datasets_failure(mock_run):
    """Test check_datasets with subprocess failure."""
    mock_run.side_effect = subprocess.CalledProcessError(1, 'fake command', stderr='Error checking datasets')

    data_dir = "/fake/data"
    with pytest.raises(RuntimeError, match="Error checking datasets: Error checking datasets"):
        check_datasets(data_dir)

    mock_run.assert_called_once_with(
        [sys.executable, 'cli/check_datasets.py', '--data', data_dir, '--step_id', 'check_datasets'],
        capture_output=True,
        text=True,
        check=True
    )

@patch('core.wrappers.subprocess.run')
def test_check_datasets_with_optional_args(mock_run):
    """Test check_datasets with optional arguments."""
    mock_result = MagicMock()
    mock_result.stdout = "Datasets checked successfully with optional args"
    mock_run.return_value = mock_result

    data_dir = "/fake/data"
    output_path = "/fake/output.json"
    step_id = "custom_check"
    output = check_datasets(data_dir, output_path=output_path, step_id=step_id)

    mock_run.assert_called_once_with(
        [sys.executable, 'cli/check_datasets.py', '--data', data_dir, '--out', output_path, '--step_id', step_id],
        capture_output=True,
        text=True,
        check=True
    )
    assert output == "Datasets checked successfully with optional args"


# Mock subprocess.run for extract_transcript tests
@patch('core.wrappers.subprocess.run')
def test_extract_transcript_success(mock_run):
    """Test extract_transcript with successful subprocess execution."""
    mock_result = MagicMock()
    mock_result.stdout = "Transcript extracted successfully"
    mock_run.return_value = mock_result

    pptx_path = "/fake/presentation.pptx"
    output = extract_transcript(pptx_path)

    mock_run.assert_called_once_with(
        [sys.executable, 'cli/extract_transcript.py', '--pptx', pptx_path, '--cue', 'transition', '--step_id', 'extract_transcript'],
        capture_output=True,
        text=True,
        check=True
    )
    assert output == "Transcript extracted successfully"

@patch('core.wrappers.subprocess.run')
def test_extract_transcript_failure(mock_run):
    """Test extract_transcript with subprocess failure."""
    mock_run.side_effect = subprocess.CalledProcessError(1, 'fake command', stderr='Error extracting transcript')

    pptx_path = "/fake/presentation.pptx"
    with pytest.raises(RuntimeError, match="Error extracting transcript: Error extracting transcript"):
        extract_transcript(pptx_path)

    mock_run.assert_called_once_with(
        [sys.executable, 'cli/extract_transcript.py', '--pptx', pptx_path, '--cue', 'transition', '--step_id', 'extract_transcript'],
        capture_output=True,
        text=True,
        check=True
    )

@patch('core.wrappers.subprocess.run')
def test_extract_transcript_with_optional_args(mock_run):
    """Test extract_transcript with optional arguments."""
    mock_result = MagicMock()
    mock_result.stdout = "Transcript extracted successfully with optional args"
    mock_run.return_value = mock_result

    pptx_path = "/fake/presentation.pptx"
    output_path = "/fake/transcript.json"
    cue_token = "slide_end"
    step_id = "custom_extract"
    output = extract_transcript(pptx_path, output_path=output_path, cue_token=cue_token, step_id=step_id)

    mock_run.assert_called_once_with(
        [sys.executable, 'cli/extract_transcript.py', '--pptx', pptx_path, '--out', output_path, '--cue', cue_token, '--step_id', step_id],
        capture_output=True,
        text=True,
        check=True
    )
    assert output == "Transcript extracted successfully with optional args"
