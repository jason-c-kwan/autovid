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

    # Slide 0 (as per user request - unicode ellipse)
    slide0 = prs.slides.add_slide(prs.slide_layouts[0])
    title0 = slide0.shapes.title
    title0.text = "Slide 0 Title (Unicode Ellipse)"
    notes_slide0 = slide0.notes_slide
    notes_slide0.notes_text_frame.text = "First half …"

    # Slide 1 (as per user request - unicode ellipse with trailing ellipse)
    slide1 = prs.slides.add_slide(prs.slide_layouts[0])
    title1 = slide1.shapes.title
    title1.text = "Slide 1 Title (Unicode Ellipse Trailing)"
    notes_slide1 = slide1.notes_slide
    notes_slide1.notes_text_frame.text = "… second half…"

    # Slide 2 (new - ASCII ellipses)
    slide2 = prs.slides.add_slide(prs.slide_layouts[0])
    title2 = slide2.shapes.title
    title2.text = "Slide 2 Title (ASCII Ellipses)"
    notes_slide2 = slide2.notes_slide
    notes_slide2.notes_text_frame.text = "...third half..."

    # Slide 3 (new - mixed ellipses)
    slide3 = prs.slides.add_slide(prs.slide_layouts[0])
    title3 = slide3.shapes.title
    title3.text = "Slide 3 Title (Mixed Ellipses)"
    notes_slide3 = slide3.notes_slide
    notes_slide3.notes_text_frame.text = "...fourth half…"

    # Slide 4 (original slide 2, now slide 4)
    slide4 = prs.slides.add_slide(prs.slide_layouts[0])
    title4 = slide4.shapes.title
    title4.text = "Slide 4 Title (No Ellipses)"
    notes_slide4 = slide4.notes_slide
    notes_slide4.notes_text_frame.text = "This is the only segment on slide 4."

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
    assert len(manifest["slides"]) == 5 # Now 5 slides
    assert manifest["step_id"] == "test_extraction"
    assert manifest["pptx"] == pptx_path

    # Assert segments for Slide 0 (index 0) - Unicode Ellipse
    slide0_segments = manifest["slides"][0]["segments"]
    assert len(slide0_segments) == 1
    assert slide0_segments[0]["kind"] == "text"
    assert slide0_segments[0]["text"] == "First half"
    assert slide0_segments[0]["continue"] == "end"

    # Assert segments for Slide 1 (index 1) - Unicode Ellipse Trailing
    slide1_segments = manifest["slides"][1]["segments"]
    assert len(slide1_segments) == 1
    assert slide1_segments[0]["kind"] == "text"
    assert slide1_segments[0]["text"] == "second half"
    assert slide1_segments[0]["continue"] == ["start", "end"]

    # Assert segments for Slide 2 (index 2) - ASCII Ellipses
    slide2_segments = manifest["slides"][2]["segments"]
    assert len(slide2_segments) == 1
    assert slide2_segments[0]["kind"] == "text"
    assert slide2_segments[0]["text"] == "third half"
    assert slide2_segments[0]["continue"] == ["start", "end"]

    # Assert segments for Slide 3 (index 3) - Mixed Ellipses
    slide3_segments = manifest["slides"][3]["segments"]
    assert len(slide3_segments) == 1
    assert slide3_segments[0]["kind"] == "text"
    assert slide3_segments[0]["text"] == "fourth half"
    assert slide3_segments[0]["continue"] == ["start", "end"]

    # Assert segments for Slide 4 (index 4) - No Ellipses
    slide4_segments = manifest["slides"][4]["segments"]
    assert len(slide4_segments) == 1
    assert slide4_segments[0]["kind"] == "text"
    assert slide4_segments[0]["text"] == "This is the only segment on slide 4."
    assert "continue" not in slide4_segments[0]

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
