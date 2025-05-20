import argparse
import json
import sys
import os
import re

def join_slide_segments(segments):
    """
    Joins text segments from a slide, interpreting 'continue' flags for ellipses.
    """
    if not segments:
        return ""

    elements = []
    for i, segment in enumerate(segments):
        if segment.get("kind") != "text":
            continue

        text = segment.get("text", "")
        s_cont = segment.get("continue", [])
        is_list_cont = isinstance(s_cont, list)
        
        leading_ellipse = (s_cont == "start") or (is_list_cont and "start" in s_cont)
        trailing_ellipse = (s_cont == "end") or (is_list_cont and "end" in s_cont)

        if leading_ellipse:
            if not elements or elements[-1] != "...":
                elements.append("...")
        
        if text:  # Only add non-empty text
            elements.append(text)
            
        if trailing_ellipse:
            # Avoid adding "..." if the text element itself already ends with "..."
            # or if the last element added was already "..." (e.g. empty text segment with continue flags)
            if elements and elements[-1] == text and text.endswith("..."):
                pass  # Text itself contains the trailing ellipsis
            elif not elements or elements[-1] != "...":
                elements.append("...")
    
    if not elements:
        return ""

    # Join elements with appropriate spacing
    final_str_parts = []
    for i, el in enumerate(elements):
        if el == "...":
            if final_str_parts and final_str_parts[-1].endswith(" "):
                # If "text ", current is "...", make it "text..."
                final_str_parts[-1] = final_str_parts[-1][:-1] + "..."
            else:
                final_str_parts.append("...")
        else:  # It's text
            if final_str_parts:
                last_part = final_str_parts[-1]
                if not last_part.endswith(" ") and not last_part.endswith("..."):
                    final_str_parts.append(" ")  # Space before text if needed
                elif last_part.endswith("..."):
                    final_str_parts.append(" ")  # Space after ... before text
            final_str_parts.append(el)
            
    final_str = "".join(final_str_parts)

    # Minimal Normalization strategy:
    # 1. Replace any sequence of 3+ dots with a single "..." (e.g., "...." -> "...")
    final_str = re.sub(r'\.{4,}', '...', final_str) # Keep "..." as is, change "...." or more
    # 2. Consolidate adjacent ellipses if they were formed (e.g., "... ...", "...    ...") into a single "..."
    final_str = re.sub(r'(\.\.\.)(\s*\.\.\.)+', r'\1', final_str)
    # 3. General cleanup of multiple spaces and leading/trailing whitespace.
    final_str = re.sub(r'\s{2,}', ' ', final_str).strip()

    return final_str

def split_into_sentences(text):
    """
    Splits text into sentences. Attempts to handle '.', '!', '?', and '...' as terminators.
    """
    if not text:
        return []
    
    # Standard simple regex for splitting sentences.
    # Splits after ., !, ?, or ... when followed by whitespace or end of string.
    sentences = re.split(r'(?<=[.!?])(?:\s+|$)|(?<=\.\.\.)(?:\s+|$)', text)
    
    # Filter out any empty strings that might result from the split (e.g., multiple spaces)
    return [s.strip() for s in sentences if s and s.strip()]

def preprocess_transcript_data(transcript_data, chunk_mode):
    """
    Processes transcript data to join segments and chunk text.
    """
    processed_slides = []
    if "slides" not in transcript_data or not isinstance(transcript_data["slides"], list):
        # Handle cases where 'slides' might be missing or not a list
        # Or raise an error, depending on expected robustness
        transcript_data["slides"] = [] # Ensure it's a list for iteration

    for slide_data in transcript_data.get("slides", []):
        # Create a copy to modify; preserve original fields like 'index'
        new_slide_data = {key: value for key, value in slide_data.items() if key != "segments"}

        segments = slide_data.get("segments", [])
        merged_text_for_slide = join_slide_segments(segments)
        new_slide_data["merged_text"] = merged_text_for_slide

        tts_texts = []
        if merged_text_for_slide: # Only process if there's text
            if chunk_mode == "slide":
                tts_texts.append(merged_text_for_slide)
            elif chunk_mode == "sentence":
                sentences = split_into_sentences(merged_text_for_slide)
                tts_texts.extend(s for s in sentences if s) # Filter out empty strings again
        
        new_slide_data["tts_texts"] = tts_texts
        processed_slides.append(new_slide_data)

    output_data = transcript_data.copy()
    output_data["slides"] = processed_slides
    output_data["preprocessing_applied"] = {"chunk_mode": chunk_mode}
    return output_data

def main():
    parser = argparse.ArgumentParser(
        description="Preprocesses transcript JSON data from extract_transcript.py. "\
                    "Joins text segments, handles ellipses, and chunks text for TTS."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=argparse.FileType('r'),
        default=sys.stdin,
        help="Path to the input JSON file. Reads from stdin if not provided."
    )
    parser.add_argument(
        "--chunk_mode",
        choices=["sentence", "slide"],
        required=True,
        help="Chunking mode: 'sentence' to split by full sentences, 'slide' to group all slide text."
    )
    parser.add_argument(
        "--out",
        type=argparse.FileType('w'),
        default=sys.stdout,
        help="Path to output modified JSON file. Writes to stdout if not provided."
    )

    args = parser.parse_args()

    try:
        input_json = json.load(args.input_file)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input. {e}", file=sys.stderr)
        sys.exit(1) # Exit non-zero
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1) # Exit non-zero
    finally:
        if args.input_file is not sys.stdin:
            args.input_file.close()

    processed_data = preprocess_transcript_data(input_json, args.chunk_mode)

    try:
        json.dump(processed_data, args.out, indent=2)
        if args.out is not sys.stdout:
            # Add a newline to the end of the file if not writing to stdout,
            # for cleaner diffs and POSIX compliance.
            args.out.write("\n")
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1) # Exit non-zero
    finally:
        if args.out is not sys.stdout:
            args.out.close()

if __name__ == "__main__":
    main()
