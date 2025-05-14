import pytest
import subprocess
import json
import os
from pptx import Presentation

# Define the path for the dummy PPTX file
DUMMY_PPTX_PATH = "tests/fixtures/dummy_presentation.pptx"
OUTPUT_MANIFEST_PATH = "workspace/01_notes/dummy_presentation_manifest.json"

@pytest.fixture(scope="module")
def create_dummy_pptx():
    """Creates a dummy PPTX file with speaker notes for testing."""
    prs = Presentation()

    # Slide 1
    slide1 = prs.slides.add_slide(prs.slide_layouts[0])
    title1 = slide1.shapes.title
    title1.text = "Slide 1 Title"
    notes_slide1 = slide1.notes_slide
    notes_slide1.notes_text_frame.text = "This is the first segment. transition This is the second segment."

    # Slide 2
    slide2 = prs.slides.add_slide(prs.slide_layouts[0])
    title2 = slide2.shapes.title
    title2.text = "Slide 2 Title"
    notes_slide2 = slide2.notes_slide
    notes_slide2.notes_text_frame.text = "This is the only segment on slide 2."

    # Ensure the fixtures directory exists
    os.makedirs(os.path.dirname(DUMMY_PPTX_PATH), exist_ok=True)

    prs.save(DUMMY_PPTX_PATH)

    yield DUMMY_PPTX_PATH

    # Clean up the dummy PPTX file and output manifest after tests
    if os.path.exists(DUMMY_PPTX_PATH):
        os.remove(DUMMY_PPTX_PATH)
    if os.path.exists(OUTPUT_MANIFEST_PATH):
        os.remove(OUTPUT_MANIFEST_PATH)
    # Clean up the output directory if it's empty
    output_dir = os.path.dirname(OUTPUT_MANIFEST_PATH)
    if os.path.exists(output_dir) and not os.listdir(output_dir):
        os.rmdir(output_dir)


def test_extract_transcript_success(create_dummy_pptx):
    """Tests the extract_transcript script for successful extraction."""
    pptx_path = create_dummy_pptx
    script_path = "scripts/extract_transcript.py"

    # Ensure the output directory exists before running the script
    os.makedirs(os.path.dirname(OUTPUT_MANIFEST_PATH), exist_ok=True)

    result = subprocess.run(
        [
            "python",
            script_path,
            "--pptx",
            pptx_path,
            "--out",
            OUTPUT_MANIFEST_PATH,
            "--cue",
            "transition",
            "--step_id",
            "test_extraction"
        ],
        capture_output=True,
        text=True
    )

    # Assert exit code is 0
    assert result.returncode == 0

    # Load the output manifest
    with open(OUTPUT_MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # Assert manifest content
    assert manifest["status"] == "success"
    assert len(manifest["slides"]) == 2
    assert manifest["slides"][0]["segments"][1]["cue"] == "transition"
    assert manifest["step_id"] == "test_extraction"
    assert manifest["pptx"] == pptx_path

def test_extract_transcript_no_notes(create_dummy_pptx):
    """Tests the extract_transcript script when a slide has no notes."""
    # Create a presentation with a slide that has no notes
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    title = slide.shapes.title
    title.text = "Slide with no notes"
    
    no_notes_pptx_path = "tests/fixtures/no_notes_presentation.pptx"
    prs.save(no_notes_pptx_path)

    script_path = "scripts/extract_transcript.py"

    result = subprocess.run(
        [
            "python",
            script_path,
            "--pptx",
            no_notes_pptx_path,
        ],
        capture_output=True,
        text=True
    )

    # Assert exit code is 1
    assert result.returncode == 1

    # Load the output manifest from stdout
    manifest = json.loads(result.stdout)

    # Assert manifest content
    assert manifest["status"] == "failure"
    assert "Slide 0 has no speaker notes." in manifest["error"]

    # Clean up the dummy PPTX file
    if os.path.exists(no_notes_pptx_path):
        os.remove(no_notes_pptx_path)
