# AutoVid Pipeline Fixes Implementation Plan

## Executive Summary

This document outlines the implementation plan to fix three critical issues in the AutoVid pipeline:
1. **Missing Quality Control**: TTS audio glitches (word cutoffs) not detected before RVC processing
2. **Scene Detection Failure**: Video analysis detects 2 scenes instead of expected 18 for Keynote slide videos
3. **Sync Plan Empty**: Video synchronization produces 0 segments due to scene/audio mismatch

## Current Pipeline Status

### Working Components ✅
- Dataset validation, transcript extraction, TTS generation (Orpheus/Piper)
- RVC voice conversion (46/46 chunks successful)
- Audio splicing (326-second final narration generated)
- Basic video analysis framework

### Broken Components ❌
- Quality control step (`qc_pronounce` defined but not implemented)
- Scene detection (finds 2 scenes vs 18 expected transition cues)
- Video synchronization (empty sync plan with 0 segments)

## Issue Analysis

### Issue 1: TTS Audio Glitches
**Root Cause**: Orpheus TTS has 768-token input limit causing word cutoffs when slide text exceeds limit.

**Evidence**: 
- User reported word cutoff glitch in first few seconds
- `cli/orpheus_tts_cli.py` sets `max_model_len=768` 
- Long slides get truncated mid-sentence

**Current QC Status**:
- `core/qc_audio.py` has MOS scoring and WER detection functions
- `config/pipeline.yaml` defines `qc_pronounce` step with thresholds
- `autogen/conductor.py` has NO implementation for `qc_pronounce` step

### Issue 2: Scene Detection for Slide-Only Videos
**Root Cause**: FFmpeg scene detection optimized for cinematic cuts, not presentation slide transitions.

**Key Context**: 
- Video contains ONLY slides (no presenter image/video)
- Keynote export includes 1-second pause before slide transitions/animations
- Expected: 18 transitions based on `[transition]` cues in transcript
- Detected: Only 2 scene changes with current algorithm

**Current Implementation**: 
- Uses FFmpeg `select='gt(scene,0.4)'` filter
- Threshold too high for subtle slide transitions
- No Keynote-specific optimizations

### Issue 3: Video Synchronization Logic
**Root Cause**: Sync plan expects 1:1 mapping between scenes and audio chunks, fails when counts mismatch.

**Evidence**:
- Audio: 46 chunks from slide-by-slide narration
- Video: 2 detected scenes vs 18 expected
- Result: Empty sync plan (`"segments": []`)

## Implementation Plan

## Part 1: Implement Audio Quality Control (Priority: HIGH)

### 1.1 Add QC Step to Pipeline Conductor
**File**: `autogen/conductor.py`
**Location**: Insert after line 266 (between `tts_run` and `apply_rvc`)

```python
elif step_id == "qc_pronounce":
    if 'all_tts_manifests' not in locals() or not all_tts_manifests:
        logging.warning(f"Skipping QC step '{step_id}': No TTS manifests available.")
        continue
    
    step_params = step.get("parameters", {})
    mos_threshold = step_params.get("mos_threshold", 3.5)
    wer_threshold = step_params.get("wer_threshold", 0.10)
    max_attempts = step_params.get("max_attempts", 3)
    
    qc_output_dir = os.path.join(workspace_root, "02_qc_audio")
    os.makedirs(qc_output_dir, exist_ok=True)
    
    # Process each TTS manifest through QC
    validated_manifests = []
    for idx, tts_manifest in enumerate(all_tts_manifests):
        qc_result = core.wrappers.run_audio_qc(
            input_manifest=tts_manifest.get("manifest_path"),
            output_dir=qc_output_dir,
            mos_threshold=mos_threshold,
            wer_threshold=wer_threshold,
            max_attempts=max_attempts,
            step_id=step_id
        )
        validated_manifests.append(qc_result)
    
    # Update all_tts_manifests with QC-validated versions
    all_tts_manifests = validated_manifests
    logging.info(f"QC step completed. Validated {len(validated_manifests)} manifests.")
```

### 1.2 Create QC Wrapper Function
**File**: `core/wrappers.py`
**Add new function**:

```python
def run_audio_qc(
    input_manifest: str,
    output_dir: str,
    mos_threshold: float = 3.5,
    wer_threshold: float = 0.10,
    max_attempts: int = 3,
    step_id: str = "qc_pronounce"
) -> Dict[str, Any]:
    """
    Quality control wrapper for TTS audio validation and fixing.
    
    Args:
        input_manifest: Path to TTS audio manifest
        output_dir: Output directory for QC results
        mos_threshold: Minimum MOS score threshold
        wer_threshold: Maximum WER threshold
        max_attempts: Maximum re-synthesis attempts
        step_id: Step identifier
        
    Returns:
        Dict: QC results and updated manifest
    """
    cmd = [
        sys.executable, "cli/qc_audio.py",
        "--input", input_manifest,
        "--output", output_dir,
        "--mos-threshold", str(mos_threshold),
        "--wer-threshold", str(wer_threshold),
        "--max-attempts", str(max_attempts)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Load QC manifest
        qc_manifest_path = Path(output_dir) / f"qc_manifest_{Path(input_manifest).stem}.json"
        if qc_manifest_path.exists():
            with open(qc_manifest_path, 'r') as f:
                return json.load(f)
        else:
            raise RuntimeError(f"QC manifest not found: {qc_manifest_path}")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio QC failed: {e.stderr}") from e
```

### 1.3 Create QC CLI Tool
**File**: `cli/qc_audio.py` (new file)

```python
#!/usr/bin/env python3
"""
Audio Quality Control CLI for detecting and fixing TTS glitches.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.qc_audio import score_mos, is_bad_alignment
from core.wrappers import orpheus_tts, piper_tts

def main():
    parser = argparse.ArgumentParser(description="Audio Quality Control")
    parser.add_argument("--input", required=True, help="Input TTS manifest")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--mos-threshold", type=float, default=3.5)
    parser.add_argument("--wer-threshold", type=float, default=0.10)
    parser.add_argument("--max-attempts", type=int, default=3)
    
    args = parser.parse_args()
    
    # Load TTS manifest
    with open(args.input, 'r') as f:
        tts_manifest = json.load(f)
    
    qc_results = []
    fixed_chunks = 0
    
    # Process each audio chunk
    for chunk in tts_manifest.get("tts_results", []):
        wav_path = chunk.get("wav_path")
        text = chunk.get("text")
        
        if not wav_path or not text:
            continue
            
        # Run quality checks
        mos_score = score_mos(wav_path)
        # TODO: Add WhisperX transcription and WER calculation
        
        chunk_qc = {
            "chunk_id": chunk.get("id"),
            "original_path": wav_path,
            "mos_score": mos_score,
            "mos_pass": mos_score >= args.mos_threshold,
            "text": text,
            "attempts": 1,
            "status": "pass" if mos_score >= args.mos_threshold else "fail"
        }
        
        # If QC fails, attempt re-synthesis
        if not chunk_qc["mos_pass"] and chunk_qc["attempts"] < args.max_attempts:
            # TODO: Implement re-synthesis logic
            pass
            
        qc_results.append(chunk_qc)
    
    # Save QC manifest
    qc_manifest = {
        "step_id": "qc_pronounce",
        "input_manifest": args.input,
        "mos_threshold": args.mos_threshold,
        "wer_threshold": args.wer_threshold,
        "chunks_processed": len(qc_results),
        "chunks_fixed": fixed_chunks,
        "qc_results": qc_results
    }
    
    output_path = Path(args.output) / f"qc_manifest_{Path(args.input).stem}.json"
    with open(output_path, 'w') as f:
        json.dump(qc_manifest, f, indent=2)
    
    print(f"QC completed: {len(qc_results)} chunks processed, {fixed_chunks} fixed")

if __name__ == "__main__":
    main()
```

## Part 2: Fix Scene Detection for Slide-Only Videos (Priority: HIGH)

### 2.1 Enhanced Scene Detection Algorithm
**File**: `core/video_analysis.py`
**Function**: Modify `detect_scene_changes()`

Key changes for slide-only videos:
- **Lower threshold**: Use 0.1 instead of 0.4 for subtle slide transitions
- **Frame differencing**: Detect static periods (1s pauses) between slides
- **Multiple detection passes**: Combine different algorithms

```python
def detect_keynote_scenes(
    video_path: str, 
    threshold: float = 0.1,
    keynote_delay: float = 1.0,
    min_scene_duration: float = 0.5
) -> List[float]:
    """
    Keynote-optimized scene detection for slide-only videos.
    
    Detects:
    1. Static frame periods (1s pauses before transitions)
    2. Content changes between slides
    3. Animation triggers within slides
    """
    
    # Method 1: Ultra-sensitive scene detection
    scenes_method1 = _detect_scenes_sensitive(video_path, threshold=0.05)
    
    # Method 2: Frame difference analysis for static periods
    scenes_method2 = _detect_static_transitions(video_path, pause_duration=1.0)
    
    # Method 3: Content-based detection with histogram comparison
    scenes_method3 = _detect_content_changes(video_path, threshold=0.15)
    
    # Combine and validate results
    combined_scenes = _merge_scene_detections([scenes_method1, scenes_method2, scenes_method3])
    
    # Apply Keynote delay compensation
    adjusted_scenes = [max(0, t - keynote_delay) for t in combined_scenes]
    
    # Filter minimum duration
    filtered_scenes = _filter_min_duration(adjusted_scenes, min_scene_duration)
    
    return sorted(filtered_scenes)
```

### 2.2 Add Frame Difference Detection
**File**: `core/video_analysis.py`
**Add new function**:

```python
def _detect_static_transitions(video_path: str, pause_duration: float = 1.0) -> List[float]:
    """
    Detect scene transitions by finding static frame periods.
    
    Keynote videos have 1s pauses before slide transitions.
    This function detects when frames become identical for 1+ seconds.
    """
    
    cmd = [
        'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
        '-show_entries', 'frame=pkt_pts_time',
        '-of', 'csv=p=0', video_path
    ]
    
    # Get frame timestamps
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    frame_times = [float(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
    
    # Extract frames for comparison
    static_periods = []
    # TODO: Implement frame extraction and comparison logic
    
    return static_periods
```

### 2.3 Validation Against Expected Transitions
**File**: `core/video_analysis.py`
**Enhance**: `validate_transition_count()`

```python
def validate_transition_count(
    detected_count: int, 
    expected_count: int,
    tolerance: float = 0.3
) -> Tuple[str, str]:
    """
    Validate detected scenes against expected transition cues.
    
    For Keynote videos, we expect close match between detected scenes
    and [transition] cues in transcript.
    """
    
    if expected_count == 0:
        return "SKIP", "No expected transitions provided"
    
    ratio = detected_count / expected_count
    
    if 0.7 <= ratio <= 1.3:  # Within 30%
        return "PASS", f"Good match: {detected_count}/{expected_count} scenes"
    elif 0.4 <= ratio < 0.7:
        return "WARN", f"Under-detection: {detected_count}/{expected_count} scenes"
    elif ratio < 0.4:
        return "FAIL", f"Severe under-detection: {detected_count}/{expected_count} scenes"
    else:
        return "WARN", f"Over-detection: {detected_count}/{expected_count} scenes"
```

## Part 3: Fix Video Synchronization Logic (Priority: MEDIUM)

### 3.1 Robust Sync Plan Generation
**File**: `core/slide_sync.py`
**Function**: Modify `create_sync_plan()`

```python
def create_sync_plan(
    video_analysis_manifest: Dict[str, Any],
    audio_splice_manifest: Dict[str, Any],
    transcript_manifest: Dict[str, Any],
    assembly_method: str = "intelligent"
) -> Dict[str, Any]:
    """
    Create video synchronization plan with intelligent mapping strategies.
    
    Handles cases where scene count != audio chunk count.
    """
    
    # Extract data
    detected_scenes = _extract_transition_points(video_analysis_manifest)
    narration_segments = _extract_narration_segments(audio_splice_manifest)
    transcript_cues = _extract_transcript_cues(transcript_manifest)
    
    # Choose sync strategy based on data quality
    if len(detected_scenes) == len(narration_segments):
        # Perfect match - use direct mapping
        segments = _create_direct_mapping(detected_scenes, narration_segments)
    elif len(transcript_cues) > 0:
        # Use transcript cues as ground truth
        segments = _create_transcript_guided_mapping(
            detected_scenes, narration_segments, transcript_cues
        )
    else:
        # Fallback to interpolation
        segments = _create_interpolated_mapping(detected_scenes, narration_segments)
    
    return {
        "video_path": video_analysis_manifest.get("input_file", ""),
        "audio_path": audio_splice_manifest.get("output_file", ""),
        "segments": segments,
        "sync_strategy": assembly_method,
        "scene_count": len(detected_scenes),
        "audio_chunk_count": len(narration_segments),
        "transcript_cue_count": len(transcript_cues)
    }
```

### 3.2 Transcript-Guided Mapping
**File**: `core/slide_sync.py`
**Add new function**:

```python
def _create_transcript_guided_mapping(
    detected_scenes: List[float],
    narration_segments: List[Dict],
    transcript_cues: List[Dict]
) -> List[Dict[str, Any]]:
    """
    Create sync mapping using transcript [transition] cues as ground truth.
    
    When scene detection fails, use transcript slide numbers and cues
    to create logical audio-video mappings.
    """
    
    segments = []
    
    # Group narration by slide number from transcript
    slide_groups = {}
    for segment in narration_segments:
        slide_num = segment.get("slide_number", 0)
        if slide_num not in slide_groups:
            slide_groups[slide_num] = []
        slide_groups[slide_num].append(segment)
    
    # Create segments for each slide
    current_time = 0.0
    for slide_num in sorted(slide_groups.keys()):
        slide_segments = slide_groups[slide_num]
        slide_duration = sum(seg.get("duration", 0) for seg in slide_segments)
        
        segment = {
            "video_start": current_time,
            "video_end": current_time + slide_duration,
            "audio_segments": slide_segments,
            "slide_number": slide_num,
            "gap_needed": 0.0  # Calculate based on original video timing
        }
        
        segments.append(segment)
        current_time += slide_duration
    
    return segments
```

## Part 4: Configuration Updates

### 4.1 Pipeline Configuration
**File**: `config/pipeline.yaml`
**Modifications**:

```yaml
steps:
  - id: check_datasets
  - id: extract_transcript
    parameters:
      cue: "[transition]"
  - id: tts_run
    parameters:
      engine: orpheus
      orpheus_model: canopylabs/orpheus-tts-0.1-finetune-prod
      orpheus_voice: dan
      orpheus_temperature: 0.2
      # NEW: Token limit handling
      max_tokens_per_chunk: 512  # Reduced from 768
      enable_sentence_splitting: true
  
  # NEW: Quality control step
  - id: qc_pronounce
    parameters:
      mos_threshold: 3.5
      wer_threshold: 0.10
      max_attempts: 3
      enable_transcription: true
      
  - id: apply_rvc
  - id: splice_audio
  
  - id: analyze_video
    parameters:
      # NEW: Keynote-optimized settings
      scene_threshold: 0.1          # Much more sensitive
      movement_threshold: 0.05
      keynote_delay: 1.0
      presentation_mode: true       # Enable Keynote optimizations
      enable_static_detection: true # Detect 1s pause periods
      validate_against_transcript: true
      
  - id: sync_slides
    parameters:
      assembly_method: intelligent  # Use smart mapping strategies
      optimize_gaps: true
      create_preview: false
```

### 4.2 Enhanced Video Analysis Config
```yaml
# Keynote-specific video analysis
video_analysis:
  scene_threshold: 0.1              # Sensitive for slide transitions
  movement_threshold: 0.05          # Detect subtle animations
  keynote_delay: 1.0               # Compensation for export delay
  min_scene_duration: 0.5          # Filter rapid false positives
  presentation_mode: true          # Enable slide-optimized algorithms
  
  # Multiple detection methods
  enable_static_detection: true    # Detect 1s static periods
  enable_content_analysis: true    # Histogram-based detection
  enable_sensitive_scene: true     # Ultra-low threshold detection
  
  # Validation
  validate_transitions: true       # Check against transcript cues
  expected_cue_token: "[transition]"
```

## Implementation Priority

### Phase 1: Critical Fixes (Week 1)
1. **Implement QC step in conductor** - Prevents processing broken audio
2. **Add basic Keynote scene detection** - Lower threshold, static detection
3. **Fix token limit in Orpheus** - Split long chunks to prevent cutoffs

### Phase 2: Advanced Features (Week 2)
1. **Enhanced scene detection** - Multiple algorithms, validation
2. **Intelligent sync mapping** - Transcript-guided fallbacks
3. **QC with re-synthesis** - Automatic fixing of failed chunks

### Phase 3: Testing & Polish (Week 3)
1. **Comprehensive testing** - Multiple presentation styles
2. **Performance optimization** - Faster processing
3. **Error handling** - Graceful degradation

## Testing Strategy

### Test Cases
1. **Lecture-01_how_to_study** (current): 46 slides, 18 transitions
2. **Short presentation**: 5-10 slides
3. **Animation-heavy**: Slides with complex transitions
4. **Long presentation**: 100+ slides

### Success Criteria
- **QC**: 95%+ audio chunks pass quality thresholds
- **Scene Detection**: Detect within 20% of expected transition count
- **Sync**: Generate non-empty sync plans with proper audio-video mapping
- **Final Output**: Playable MP4 with synchronized narration

## Key Dependencies

### Python Packages
- `speechmetrics` - MOS scoring
- `jiwer` - WER calculation  
- `whisperx` - Audio transcription for QC
- `ffmpeg-python` - Enhanced video analysis

### External Tools
- `ffmpeg/ffprobe` - Video processing (already installed)
- WhisperX models - Audio transcription
- NLTK/spaCy - Text segmentation improvements

## Risk Mitigation

### High Risk Items
1. **Scene detection accuracy** - Keynote videos may have subtle transitions
   - *Mitigation*: Multiple detection algorithms, manual validation tools
2. **QC re-synthesis performance** - May slow pipeline significantly  
   - *Mitigation*: Configurable max attempts, async processing
3. **Sync mapping complexity** - Edge cases with mismatched counts
   - *Mitigation*: Fallback strategies, manual override options

### Testing Validation
- Run on multiple presentation styles before deployment
- Implement dry-run mode for validation without full processing
- Add comprehensive logging for debugging complex cases

This implementation plan provides the foundation for fixing all three critical issues while maintaining pipeline robustness and performance.