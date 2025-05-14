import json
import os
import sys
from pathlib import Path

import pytest

# Add the scripts directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from check_datasets import check_datasets, main


@pytest.fixture
def setup_success_data(tmp_path):
    """Setup temporary test data directory with mock files for success case."""
    data_dir = tmp_path / "data_success"
    data_dir.mkdir()

    # Create mock files for success case
    (data_dir / "test_lecture.pptx").touch()
    (data_dir / "test_lecture.mov").touch()

    return data_dir


@pytest.fixture
def setup_failure_data(tmp_path):
    """Setup temporary test data directory with mock files for failure case."""
    data_dir = tmp_path / "data_failure"
    data_dir.mkdir()

    # Create mock file for missing case
    (data_dir / "test_missing.pptx").touch()

    # Create mock files for a successful pair to ensure it's still included
    (data_dir / "another_lecture.pptx").touch()
    (data_dir / "another_lecture.mov").touch()

    return data_dir


def test_check_datasets_success(setup_success_data):
    """Test check_datasets function with matching files."""
    data_dir = setup_success_data
    manifest = check_datasets(str(data_dir), "test_step")

    assert manifest["step_id"] == "test_step"
    assert manifest["status"] == "success"
    assert len(manifest["pairs"]) == 1

    # Check the success pair
    success_pair = manifest["pairs"][0]
    assert success_pair["stem"] == "test_lecture"
    assert success_pair["pptx"].endswith("test_lecture.pptx")
    assert success_pair["mov"].endswith("test_lecture.mov")


def test_check_datasets_failure(setup_failure_data):
    """Test check_datasets function with a missing MOV file."""
    data_dir = setup_failure_data
    manifest = check_datasets(str(data_dir), "test_step")

    assert manifest["step_id"] == "test_step"
    assert manifest["status"] == "failed"
    assert len(manifest["pairs"]) == 2

    # Check the missing pair
    missing_pair = next(item for item in manifest["pairs"] if item["stem"] == "test_missing")
    assert missing_pair["pptx"].endswith("test_missing.pptx")
    assert missing_pair["mov"] is None

    # Check the successful pair
    success_pair = next(item for item in manifest["pairs"] if item["stem"] == "another_lecture")
    assert success_pair["pptx"].endswith("another_lecture.pptx")
    assert success_pair["mov"].endswith("another_lecture.mov")


def test_main_success_default_step_id(setup_success_data, capsys, monkeypatch):
    """Test main function with matching files and default step_id."""
    data_dir = setup_success_data
    output_file = setup_success_data / "manifest.json"

    # Mock command-line arguments without --step_id
    monkeypatch.setattr(sys, "argv", ["check_datasets.py", "--data", str(data_dir), "--out", str(output_file)])

    # Run the main function and check exit code
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0

    # Check stdout
    captured = capsys.readouterr()
    manifest_stdout = json.loads(captured.out)
    assert manifest_stdout["step_id"] == "check_datasets"
    assert manifest_stdout["status"] == "success"

    # Check output file
    with open(output_file, "r") as f:
        manifest_file = json.load(f)
    assert manifest_file["step_id"] == "check_datasets"
    assert manifest_file["status"] == "success"
    
    # Check default output file
    default_output_file = Path("workspace/00_pairs/pairs_manifest.json")
    with open(default_output_file, "r") as f:
        default_manifest_file = json.load(f)
    assert default_manifest_file["step_id"] == "check_datasets"
    assert default_manifest_file["status"] == "success"


def test_main_success_custom_step_id(setup_success_data, capsys, monkeypatch):
    """Test main function with matching files and a custom step_id."""
    data_dir = setup_success_data
    output_file = setup_success_data / "manifest.json"

    # Mock command-line arguments with --step_id
    monkeypatch.setattr(sys, "argv", ["check_datasets.py", "--data", str(data_dir), "--out", str(output_file), "--step_id", "my_custom_step"])

    # Run the main function and check exit code
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0

    # Check stdout
    captured = capsys.readouterr()
    manifest_stdout = json.loads(captured.out)
    assert manifest_stdout["step_id"] == "my_custom_step"
    assert manifest_stdout["status"] == "success"

    # Check output file
    with open(output_file, "r") as f:
        manifest_file = json.load(f)
    assert manifest_file["step_id"] == "my_custom_step"
    assert manifest_file["status"] == "success"
    
    # Check default output file
    default_output_file = Path("workspace/00_pairs/pairs_manifest.json")
    with open(default_output_file, "r") as f:
        default_manifest_file = json.load(f)
    assert default_manifest_file["step_id"] == "my_custom_step"
    assert default_manifest_file["status"] == "success"


def test_main_failure(setup_failure_data, capsys, monkeypatch):
    """Test main function with a missing MOV file and check exit code."""
    data_dir = setup_failure_data

    # Mock command-line arguments
    monkeypatch.setattr(sys, "argv", ["check_datasets.py", "--data", str(data_dir)])

    # Run the main function and check exit code
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1

    # Check stdout
    captured = capsys.readouterr()
    manifest_stdout = json.loads(captured.out)
    assert manifest_stdout["step_id"] == "check_datasets"
    assert manifest_stdout["status"] == "failed"
    
    # Check default output file
    default_output_file = Path("workspace/00_pairs/pairs_manifest.json")
    with open(default_output_file, "r") as f:
        default_manifest_file = json.load(f)
    assert default_manifest_file["step_id"] == "check_datasets"
    assert default_manifest_file["status"] == "failed"
