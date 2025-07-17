# AutoVid Project Context

## Project Overview
AutoVid is a pipeline that converts PowerPoint presentations with speaker notes and Keynote videos into fully narrated, captioned MP4 videos using AI-generated speech in the user's own voice. The system uses Microsoft AutoGen for orchestration and various AI models for TTS, voice conversion, and quality control.

## Key Architecture Components

### Core Pipeline Flow
1. **Dataset validation** - Validate PPTX/MOV pairs in `data/` directory
2. **Note extraction** - Parse PPTX speaker notes with `[transition]` cues and ellipses
3. **Text preprocessing** - Join segments and prepare for TTS
4. **TTS generation** - Use Piper or Orpheus-TTS for speech synthesis
5. **Audio quality control** - Check MOS scores, WER, and pronunciation
6. **Voice conversion** - Apply RVC model for voice matching (NOT YET IMPLEMENTED)
7. **Audio assembly** - Splice chunks into continuous narration (NOT YET IMPLEMENTED)
8. **Video synchronization** - Align audio with Keynote video (NOT YET IMPLEMENTED)
9. **Final rendering** - Output MP4 with captions (NOT YET IMPLEMENTED)

### Directory Structure
```
autovid/
├── cli/                    # Command-line interface tools
├── core/                   # Core processing functions
├── autogen/               # AutoGen orchestration
├── config/                # YAML configuration files
├── tests/                 # Comprehensive test suite
├── workspace/             # Runtime workspace
└── data/                  # Input PPTX/MOV pairs
```

### Key Files and Functions

#### CLI Tools (all implemented)
- `cli/check_datasets.py` - Dataset validation
- `cli/extract_transcript.py` - PPTX note extraction
- `cli/transcript_preprocess.py` - Text preprocessing
- `cli/piper_tts.py` - Piper TTS generation
- `cli/orpheus_tts_cli.py` - Orpheus TTS generation
- `cli/transcribe_whisperx.py` - WhisperX transcription

#### Core Functions
- `core/qc_audio.py` - Audio quality control (MOS, WER, phoneme injection)
- `core/wrappers.py` - Subprocess wrappers for CLI tools

#### Orchestration
- `autogen/conductor.py` - AutoGen pipeline conductor with Gemini integration

### Configuration
- `config/pipeline.yaml` - Pipeline steps and model configuration
- `config/agents.yaml` - AutoGen agent behavior configuration

### Testing
- Run tests with: `python -m pytest tests/`
- All major components have corresponding test files

### Current Implementation Status
- **COMPLETE**: Dataset validation, PPTX processing, TTS generation, audio QC, transcription
- **MISSING**: Voice conversion (RVC), audio splicing, video sync, final rendering, subtitle generation

### Common Commands
- Check datasets: `python cli/check_datasets.py`
- Extract notes: `python cli/extract_transcript.py <file.pptx>`
- Generate TTS: `python cli/piper_tts.py <transcript.json>`
- Quality control: `python core/qc_audio.py <audio_manifest.json>`
- Run full pipeline: `python autogen/conductor.py`

### Dependencies
- AutoGen ≥ 0.4 for agent framework
- Piper-TTS and Orpheus-TTS for speech synthesis
- WhisperX for transcription
- python-pptx for PowerPoint processing
- Various audio processing libraries (jiwer, speechmetrics, etc.)

### Notes for Future Development
- **IMPORTANT**: Always check off completed tasks in `autovid_tasks.md` as you complete them
- Missing components need FFmpeg integration for video processing
- RVC model integration is critical for voice matching
- The 1-second delay in Keynote exports needs special handling in video sync
- All components use manifest-based architecture for tracking intermediate results

### File Naming Conventions
```
data/<stem>.pptx           # Input presentation
data/<stem>.mov            # Input Keynote video (1s delay)
out/<stem>.wav             # Final narration audio
out/<stem>.srt             # Subtitle file
out/<stem>.mp4             # Final video output
logs/<stem>.json           # Processing logs
```

### Error Handling
- All components have comprehensive error handling
- Manifest files track success/failure status
- Workspace directory contains intermediate results
- Check logs/ directory for detailed error information