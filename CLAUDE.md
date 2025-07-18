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
- `cli/rvc_setup.py` - RVC environment setup and management
- `cli/rvc_convert.py` - RVC voice conversion processing
- `cli/splice_audio.py` - Audio chunk splicing and assembly

#### Core Functions
- `core/qc_audio.py` - Audio quality control (MOS, WER, phoneme injection)
- `core/wrappers.py` - Subprocess wrappers for CLI tools
- `core/rvc_environment.py` - RVC environment management and isolation
- `core/rvc_processing.py` - RVC model validation and processing utilities
- `core/audio_splicing.py` - Audio chunk splicing and assembly

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
- **IMPLEMENTED BUT NEEDS CONFIGURATION**: Voice conversion (RVC) - environment and code ready, needs GPU config fix
- **MISSING**: Audio splicing, video sync, final rendering, subtitle generation

### Environment Setup
AutoVid uses a dual-environment setup to handle dependency conflicts between RVC and other components:

- **Primary Environment**: `autovid` - Used for most pipeline operations
  - **Activation**: `source ~/.bashrc && eval "$(/home/jason/mambaforge/bin/conda shell.bash hook)" && conda activate autovid`
  - **Python Path**: Set `PYTHONPATH=/home/jason/git_repos/autovid` for module imports
  - **GPU Setup**: Uses PyTorch 2.6.0 with CUDA 12.8 support

- **RVC Environment**: `autovid-rvc` - Isolated environment for RVC voice conversion
  - **Activation**: `conda activate autovid-rvc`
  - **Purpose**: Handles RVC-specific dependencies to avoid conflicts
  - **GPU Setup**: Uses PyTorch 2.0.0 with CUDA 12.8 support
  - **Key Dependencies**: faiss-cpu, librosa, soundfile, praat-parselmouth, pyworld, torchcrepe

- **GPU Configuration**: 
  - **Available GPUs**: 5 GPUs (1x RTX 3060 Ti, 4x RTX 3090)
  - **CUDA Version**: 12.8
  - **Default GPU**: Set in `config/pipeline.yaml` (should use `cuda:0` or higher for optimal performance)
  - **GPU Selection**: Use `cuda:1` through `cuda:3` (RTX 3090s) for maximum performance

### Common Commands
- Check datasets: `python cli/check_datasets.py`
- Extract notes: `python cli/extract_transcript.py <file.pptx>`
- Generate TTS: `python cli/piper_tts.py <transcript.json>`
- Quality control: `python core/qc_audio.py <audio_manifest.json>`
- Run full pipeline: `PYTHONPATH=/home/jason/git_repos/autovid python autogen/conductor.py`

#### RVC-Specific Commands
- Setup RVC environment: `python cli/rvc_setup.py setup`
- Check RVC status: `python cli/rvc_setup.py status`
- Validate RVC setup: `python cli/rvc_setup.py validate`
- Convert audio with RVC: `python cli/rvc_convert.py --input <tts_manifest.json> --output <output_dir>`
- Download RVC models: `python cli/rvc_setup.py download`

### Dependencies
- AutoGen ≥ 0.4 for agent framework
- Piper-TTS and Orpheus-TTS for speech synthesis
- WhisperX for transcription
- python-pptx for PowerPoint processing
- Various audio processing libraries (jiwer, speechmetrics, etc.)

#### RVC-Specific Dependencies (autovid-rvc environment)
- PyTorch 2.0.0 with CUDA support
- librosa 0.9.1 for audio processing
- soundfile for audio I/O
- faiss-cpu 1.7.2 for feature retrieval
- praat-parselmouth 0.4.2 for phonetic analysis
- pyworld 0.3.2 for vocoder functionality
- torchcrepe for pitch extraction
- ffmpeg-python for audio format conversion

### Notes for Future Development
- **IMPORTANT**: Always check off completed tasks in `autovid_tasks.md` as you complete them
- Missing components need FFmpeg integration for video processing
- RVC model integration is critical for voice matching
- The 1-second delay in Keynote exports needs special handling in video sync
- All components use manifest-based architecture for tracking intermediate results

### RVC Troubleshooting Guide

#### Common Issues and Solutions
1. **RVC Not Using GPU**
   - Check `config/pipeline.yaml` - ensure `device: cuda:0` (not `cpu:0`)
   - Verify GPU availability: `nvidia-smi`
   - Test PyTorch CUDA: `conda run -n autovid-rvc python -c "import torch; print(torch.cuda.is_available())"`

2. **Environment Issues**
   - Ensure autovid-rvc environment exists: `conda env list`
   - Setup environment: `python cli/rvc_setup.py setup`
   - Validate setup: `python cli/rvc_setup.py validate`

3. **Missing Models**
   - Check model files: `models/rvc/jason.pth` and `models/rvc/jason.index`
   - Download pretrained models: `python cli/rvc_setup.py download`
   - Verify model paths in `config/pipeline.yaml`

4. **Memory Issues**
   - Use RTX 3090s (cuda:1-3) for better memory capacity
   - Reduce batch size if OOM errors occur
   - Monitor GPU memory with `nvidia-smi`

5. **Audio Quality Issues**
   - Check f0_method in pipeline config (harvest, crepe, pm)
   - Verify index_rate (0.66 is typical)
   - Test with different protect values (0.33 is default)

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