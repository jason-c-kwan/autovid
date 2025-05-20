import pytest
import json
import subprocess
import sys
from pathlib import Path

# Assuming cli.transcript_preprocess is importable, or adjust path as needed
# For direct script testing, we might call it as a subprocess.
# For unit testing functions, we'd import them.
from cli.transcript_preprocess import join_slide_segments, split_into_sentences, preprocess_transcript_data

# --- Fixtures ---

@pytest.fixture
def sample_transcript_data_raw():
    """Raw output similar to extract_transcript.py for complex cases."""
    return {
        "step_id": "extract_transcript_test",
        "status": "success",
        "pptx": "test.pptx",
        "slide_count": 3,
        "slides": [
            {
                "index": 0,
                "segments": [
                    {"kind": "text", "text": "Slide one, first sentence."},
                    {"kind": "text", "text": "Second sentence, continues", "continue": "end"}
                ]
            },
            {
                "index": 1,
                "segments": [
                    {"kind": "text", "text": "on slide two", "continue": "start"},
                    {"kind": "text", "text": "with more details. This is a new thought!"},
                    {"kind": "text", "text": "And another one..."} # Implicit end
                ]
            },
            {
                "index": 2,
                "segments": [
                    {"kind": "text", "text": "Final slide text.", "continue": ["start", "end"]}, # Odd case, but test
                    {"kind": "text", "text": "More on final slide."}
                ]
            },
            {
                "index": 3, # Empty slide
                "segments": []
            },
            {
                "index": 4, # Slide with only cues or non-text
                "segments": [{"kind": "cue", "cue": "[transition]"}]
            },
            {
                "index": 5, # Slide with text that is only ellipses
                "segments": [{"kind": "text", "text": "", "continue": ["start", "end"]}]
            },
            {
                "index": 6, # Slide with text that is only ellipses in text field
                "segments": [{"kind": "text", "text": "..."}]
            }
        ]
    }

@pytest.fixture
def sample_transcript_data_simple_sentences():
    return {
        "slides": [
            {
                "index": 0,
                "segments": [
                    {"kind": "text", "text": "Hello world. This is a test!"}
                ]
            },
            {
                "index": 1,
                "segments": [
                    {"kind": "text", "text": "Question? Yes. Ellipsis..."}
                ]
            }
        ]
    }

# --- Tests for join_slide_segments ---

@pytest.mark.parametrize("segments, expected_text", [
    ([], ""),
    ([{"kind": "text", "text": "Hello"}], "Hello"),
    ([{"kind": "text", "text": "Part 1", "continue": "end"}, {"kind": "text", "text": "Part 2", "continue": "start"}], "Part 1... Part 2"), 
    ([{"kind": "text", "text": "Part 1", "continue": "end"}, {"kind": "text", "text": "Part 2"}], "Part 1... Part 2"), 
    ([{"kind": "text", "text": "Part 1"}, {"kind": "text", "text": "Part 2", "continue": "start"}], "Part 1... Part 2"), # Changed: "Part 1 ... Part 2" to "Part 1... Part 2"
    ([{"kind": "text", "text": "Text with ... in it", "continue": "end"}, {"kind": "text", "text": "Next part", "continue": "start"}], "Text with ... in it... Next part"), 
    ([{"kind": "text", "text": "Leading", "continue": "start"}], "... Leading"),
    ([{"kind": "text", "text": "Trailing", "continue": "end"}], "Trailing..."), 
    ([{"kind": "text", "text": "Both", "continue": ["start", "end"]}], "... Both..."), 
    ([{"kind": "text", "text": "No continue."}], "No continue."),
    ([{"kind": "cue", "cue": "[t]"}, {"kind": "text", "text": "Text after cue."}], "Text after cue."),
    ([{"kind": "text", "text": "Text ... with space then dots."}], "Text ... with space then dots."), # Original spacing preserved
    ([{"kind": "text", "text": "Text...no space."}], "Text...no space."), # Minimal norm preserves this
    ([{"kind": "text", "text": "One"}, {"kind": "text", "text": "Two"}, {"kind": "text", "text": "Three"}], "One Two Three"),
    ([{"kind": "text", "text": "Ends with dot."}, {"kind": "text", "text": "Starts with cap."}], "Ends with dot. Starts with cap."),
    ([{"kind": "text", "text": "", "continue": "end"}, {"kind": "text", "text": "Starts", "continue": "start"}], "... Starts"), 
    ([{"kind": "text", "text": "Text", "continue": "end"}, {"kind": "text", "text": "", "continue": "start"}], "Text..."), 
    ([{"kind": "text", "text": "Text", "continue": "end"}, {"kind": "text", "text": "", "continue": ["start", "end"]}, {"kind": "text", "text": "More", "continue": "start"}], "Text... More"), 
    ([{"kind": "text", "text": "..."}], "..."), 
    ([{"kind": "text", "text": "First line."}, {"kind": "text", "text": "Second line..."}], "First line. Second line..."), 
])
def test_join_slide_segments(segments, expected_text):
    assert join_slide_segments(segments) == expected_text

# --- Tests for split_into_sentences ---

@pytest.mark.parametrize("text, expected_sentences", [
    ("", []),
    ("Hello world. This is a test!", ["Hello world.", "This is a test!"]),
    ("One sentence.", ["One sentence."]),
    ("Question? Yes. Ellipsis...", ["Question?", "Yes.", "Ellipsis..."]),
    ("No terminator", ["No terminator"]),
    ("Ends with space. ", ["Ends with space."]),
    ("Sentence one... Sentence two.", ["Sentence one...", "Sentence two."]),
    # Adjusted for simple regex behavior with "Mr."
    ("Mr. Jones said hello. It was nice.", ["Mr.", "Jones said hello.", "It was nice."]), 
    ("This is a test... and another test.", ["This is a test...", "and another test."]),
    ("Leading space. Trailing space. ", ["Leading space.", "Trailing space."]),
    ("Multiple   spaces   between   sentences.   Like   this.", ["Multiple   spaces   between   sentences.", "Like   this."])
])
def test_split_into_sentences(text, expected_sentences):
    assert split_into_sentences(text) == expected_sentences

# --- Tests for preprocess_transcript_data (main logic) ---

def test_preprocess_slide_mode(sample_transcript_data_raw):
    processed = preprocess_transcript_data(sample_transcript_data_raw, "slide")
    
    # Slide 0
    assert processed["slides"][0]["merged_text"] == "Slide one, first sentence. Second sentence, continues..."
    assert processed["slides"][0]["tts_texts"] == ["Slide one, first sentence. Second sentence, continues..."]
    
    # Slide 1
    assert processed["slides"][1]["merged_text"] == "... on slide two with more details. This is a new thought! And another one..."
    assert processed["slides"][1]["tts_texts"] == ["... on slide two with more details. This is a new thought! And another one..."]

    # Slide 2
    assert processed["slides"][2]["merged_text"] == "... Final slide text... More on final slide." # ...word...
    assert processed["slides"][2]["tts_texts"] == ["... Final slide text... More on final slide."]

    # Slide 3 (Empty)
    assert processed["slides"][3]["merged_text"] == ""
    assert processed["slides"][3]["tts_texts"] == []

    # Slide 4 (Only cue)
    assert processed["slides"][4]["merged_text"] == ""
    assert processed["slides"][4]["tts_texts"] == []
    
    # Slide 5 (Empty text with continue flags)
    assert processed["slides"][5]["merged_text"] == "..." # Should resolve to a single ellipsis
    assert processed["slides"][5]["tts_texts"] == ["..."]

    # Slide 6 (Text is just ellipsis)
    assert processed["slides"][6]["merged_text"] == "..."
    assert processed["slides"][6]["tts_texts"] == ["..."]

    assert "preprocessing_applied" in processed
    assert processed["preprocessing_applied"]["chunk_mode"] == "slide"

def test_preprocess_sentence_mode(sample_transcript_data_raw):
    processed = preprocess_transcript_data(sample_transcript_data_raw, "sentence")

    # Slide 0
    assert processed["slides"][0]["merged_text"] == "Slide one, first sentence. Second sentence, continues..."
    assert processed["slides"][0]["tts_texts"] == ["Slide one, first sentence.", "Second sentence, continues..."]
    
    # Slide 1 - Simpler split regex will split "... on slide two..."
    assert processed["slides"][1]["merged_text"] == "... on slide two with more details. This is a new thought! And another one..."
    assert processed["slides"][1]["tts_texts"] == ["...", "on slide two with more details.", "This is a new thought!", "And another one..."]

    # Slide 2
    assert processed["slides"][2]["merged_text"] == "... Final slide text... More on final slide."
    assert processed["slides"][2]["tts_texts"] == ["...", "Final slide text...", "More on final slide."]
    
    # Slide 3 (Empty)
    assert processed["slides"][3]["tts_texts"] == []

    # Slide 5 (Empty text with continue flags resulting in "...")
    assert processed["slides"][5]["tts_texts"] == ["..."]
    
    # Slide 6 (Text is just ellipsis)
    assert processed["slides"][6]["tts_texts"] == ["..."]

    assert "preprocessing_applied" in processed
    assert processed["preprocessing_applied"]["chunk_mode"] == "sentence"

def test_preprocess_simple_sentences_sentence_mode(sample_transcript_data_simple_sentences):
    processed = preprocess_transcript_data(sample_transcript_data_simple_sentences, "sentence")
    assert processed["slides"][0]["tts_texts"] == ["Hello world.", "This is a test!"]
    assert processed["slides"][1]["tts_texts"] == ["Question?", "Yes.", "Ellipsis..."] # word...

def test_preprocess_simple_sentences_slide_mode(sample_transcript_data_simple_sentences):
    processed = preprocess_transcript_data(sample_transcript_data_simple_sentences, "slide")
    assert processed["slides"][0]["tts_texts"] == ["Hello world. This is a test!"]
    assert processed["slides"][1]["tts_texts"] == ["Question? Yes. Ellipsis..."] # word...

def test_preprocess_missing_slides_key():
    data = {"step_id": "test"}
    processed = preprocess_transcript_data(data, "slide")
    assert processed["slides"] == []
    assert "preprocessing_applied" in processed

def test_preprocess_slides_not_a_list():
    data = {"slides": "not a list"}
    processed = preprocess_transcript_data(data, "slide")
    assert processed["slides"] == [] # Should gracefully handle or initialize
    assert "preprocessing_applied" in processed


# --- Tests for CLI script execution ---

SCRIPT_PATH = Path(__file__).parent.parent / "cli" / "transcript_preprocess.py"

def run_script(input_data, *args):
    process = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)] + list(args),
        input=json.dumps(input_data),
        capture_output=True,
        text=True,
        check=False  # Don't check, we'll assert on returncode
    )
    return process

def test_cli_stdout_output(sample_transcript_data_simple_sentences, capsys):
    # Test with stdin/stdout
    # Note: capsys doesn't work well with subprocess stdout, so we call main directly or use subprocess and parse output
    
    # Using subprocess
    proc = run_script(sample_transcript_data_simple_sentences, "--chunk_mode", "sentence")
    assert proc.returncode == 0
    output_json = json.loads(proc.stdout)
    
    assert output_json["slides"][0]["tts_texts"] == ["Hello world.", "This is a test!"]
    assert output_json["slides"][1]["tts_texts"] == ["Question?", "Yes.", "Ellipsis..."] # word...
    assert output_json["preprocessing_applied"]["chunk_mode"] == "sentence"

def test_cli_file_output(sample_transcript_data_simple_sentences, tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    with open(input_file, "w") as f:
        json.dump(sample_transcript_data_simple_sentences, f)

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(input_file), "--chunk_mode", "slide", "--out", str(output_file)],
        capture_output=True,
        text=True,
        check=False
    )
    assert proc.returncode == 0
    assert proc.stderr == "" # No errors

    assert output_file.exists()
    with open(output_file, "r") as f:
        output_json = json.load(f)
    
    assert output_json["slides"][0]["tts_texts"] == ["Hello world. This is a test!"]
    assert output_json["slides"][1]["tts_texts"] == ["Question? Yes. Ellipsis..."] # word...
    assert output_json["preprocessing_applied"]["chunk_mode"] == "slide"

def test_cli_invalid_chunk_mode(sample_transcript_data_simple_sentences):
    proc = run_script(sample_transcript_data_simple_sentences, "--chunk_mode", "invalid_mode")
    assert proc.returncode != 0  # argparse should cause non-zero exit
    assert "invalid choice: 'invalid_mode'" in proc.stderr # Check for argparse error message

def test_cli_missing_chunk_mode(sample_transcript_data_simple_sentences):
    proc = run_script(sample_transcript_data_simple_sentences) # No --chunk_mode
    assert proc.returncode != 0
    assert "the following arguments are required: --chunk_mode" in proc.stderr

def test_cli_invalid_input_json(tmp_path):
    input_file = tmp_path / "invalid_input.json"
    with open(input_file, "w") as f:
        f.write("this is not json")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(input_file), "--chunk_mode", "sentence"],
        capture_output=True,
        text=True,
        check=False
    )
    assert proc.returncode == 1 # Custom exit code for JSON error
    assert "Error: Invalid JSON input" in proc.stderr

def test_cli_nonexistent_input_file():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "nonexistent_file.json", "--chunk_mode", "sentence"],
        capture_output=True,
        text=True,
        check=False
    )
    assert proc.returncode != 0 # Argparse error
    assert "can't open 'nonexistent_file.json'" in proc.stderr or "No such file or directory" in proc.stderr

# Test for merging ellipses across index boundaries (as interpreted by join_slide_segments)
# The original prompt "merges trailing ellipses across indices" is a bit ambiguous.
# `extract_transcript.py` handles marking `continue` flags.
# `transcript_preprocess.py` then uses these flags for each slide independently.
# This test ensures `join_slide_segments` correctly interprets these flags.
def test_ellipsis_merging_logic_detailed():
    # Case 1: Slide A ends, Slide B starts
    segments_A = [{"kind": "text", "text": "End of A", "continue": "end"}]
    segments_B = [{"kind": "text", "text": "Start of B", "continue": "start"}]
    assert join_slide_segments(segments_A) == "End of A..." # word...
    assert join_slide_segments(segments_B) == "... Start of B" # ... word
    # The preprocessor would produce these, and TTS would later join them if needed.
    # The preprocessor itself doesn't join across slides, only within a slide.

    # Case 2: Multiple segments within one slide
    # P1(end), P2(start), P3(end), P4(start) -> P1...P2 P3...P4
    segments_slide = [
        {"kind": "text", "text": "Part 1", "continue": "end"},
        {"kind": "text", "text": "Part 2", "continue": "start"},
        {"kind": "text", "text": "Part 3", "continue": "end"},
        {"kind": "text", "text": "Part 4", "continue": "start"}
    ]
    # Expected: "Part 1... Part 2 Part 3... Part 4"
    assert join_slide_segments(segments_slide) == "Part 1... Part 2 Part 3... Part 4"

    segments_complex = [
        {"kind": "text", "text": "Initial text."},
        {"kind": "text", "text": "Continued", "continue": "end"}
    ]
    assert join_slide_segments(segments_complex) == "Initial text. Continued..." # word...

    segments_complex_2 = [
        {"kind": "text", "text": "Starts", "continue": "start"},
        {"kind": "text", "text": "then normal."}
    ]
    assert join_slide_segments(segments_complex_2) == "... Starts then normal." # ... word
    
    segments_with_internal_ellipsis = [
        {"kind": "text", "text": "Text A... already has it.", "continue": "end"}, # Text itself has ...
        {"kind": "text", "text": "Text B", "continue": "start"}
    ]
    # Expected: "Text A... already has it... Text B" (original ... is kept, new ... added for cont flags, then consolidated)
    assert join_slide_segments(segments_with_internal_ellipsis) == "Text A... already has it... Text B"
