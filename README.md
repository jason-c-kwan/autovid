# autovid

Uses Autogen to automatically make a video presentation from a PowerPoint presentation (with speaker notes) and the video output from Keynote.

## Workflow

The pipeline converts a speaker-noted PPTX + a 1 s–delayed Keynote video into a fully narrated, captioned MP4 in the user's voice. It relies on python-pptx for note extraction, Piper or Orpheus-TTS for speech, RVC for voice conversion, `stable-ts` for alignment, and FFmpeg scene detection for timing.

The workflow is as follows:

1.  **User supplies pairs of files** in the `data` directory: PowerPoint `.pptx` containing speaker notes and a `.mov` export from Keynote. The pairs must share the same filename stem, and the Keynote video export must have a fixed 1.0 s delay between slide transitions and animation triggers. Speaker notes use `[transition]` to mark when an animation within a slide should fire, and ellipses `...` to indicate sentences spanning slide boundaries.
2.  **Dataset validation**: Confirm each `.pptx` has a corresponding `.mov` in `data`, then iterate through each dataset.
3.  **Note extraction**: Parse each PPTX and attach text (including transition cues and ellipses) to the appropriate slides.
4.  **TTS chunking**: Split text (e.g., by sentence), joining two-part sentences separated by ellipses. Run TTS either via:

    *   Piper ([northern\_english\_male](https://github.com/rhasspy/piper))
    *   Orpheus-TTS (voice “Dan” @ temperature 0.2)
        The output will later be passed through an RVC model, so choice can be refined based on final voice quality.
5.  **TTS QC & phonetic correction**: Detect artifacts or mispronunciations—especially scientific terms—and regenerate as needed. A LLM will:

    1.  Scan the transcript for tokens not in CMUdict
    2.  Generate phonetic spellings and re-synthesize
    3.  Flag any fragment whose cosine similarity to the baseline < 0.85 for manual review
6.  **Voice conversion**: Run QCed TTS through a fine-tuned RVC model to imitate the user's own voice.
7.  **Audio assembly**: Splice RVC outputs into a continuous narration track.
8.  **Video sync**: Edit the Keynote export so transitions align with the audio. Because the 1 s delay between cues may not match actual animation durations, detect movement frame ranges and adjust accordingly. If transition markers in notes are mis-counted, log the mismatch and continue; sync recovers at the next slide boundary.
9.  **Final render**: Output the combined audio+video as an MP4.
10. **Caption export**: Generate an `.srt` file from the narration track.

## Dependencies

*   Agent framework: Microsoft Autogen ≥ 0.4
*   PPTX parsing: python-pptx
*   Local TTS:
    *   [Piper en-GB-northern\_english\_male-medium](https://github.com/rhasspy/piper/tree/master)
    *   [Orpheus-TTS Dan @ temp 0.2](https://github.com/canopyai/Orpheus-TTS)
*   Alignment to SRT: stable-ts 2.x
*   Scene/slide detection: FFmpeg `select='gt(scene,0.4)'` filter
*   Voice conversion: Custom fine-tuned RVC model

## Naming conventions

Input and output files should follow these naming conventions:

```
data/
  <stem>.pptx
  <stem>.mov   # 1 s fixed delay between user-triggered events
out/
  <stem>.wav   # final RVC-processed narration
  <stem>.srt
  <stem>.mp4
  logs/<stem>.json
```

## Running the pipeline

To run the pipeline, use the following command:

```bash
python -m autogen.conductor
```
