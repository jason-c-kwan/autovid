# AutoVid Pipeline Completion Plan

## Current Status

The AutoVid pipeline currently runs successfully through the RVC step, producing individual RVC-converted audio chunks. However, several critical steps are missing or misconfigured to complete the full pipeline execution.

## Pipeline Analysis

### Implemented and Working ✅
- `check_datasets` - Dataset validation  
- `extract_transcript` - PowerPoint note extraction
- `tts_run` - Text-to-speech generation (Piper/Orpheus)
- `qc_pronounce` - Audio quality control (newly implemented)
- `apply_rvc` - Voice conversion (produces individual chunks)

### Implemented but Not Invoked ❌
- `splice_audio` - Audio chunk assembly (wrapper exists, step implemented)
- `analyze_video` - Video scene detection (wrapper exists, step implemented) 
- Video synchronization step (wrapper exists but step name mismatch)

### Missing Implementation ❌
- `make_srt` - Subtitle generation (configured but not implemented)

## Critical Issues Blocking Full Pipeline

### Issue 1: Step Name Mismatch
**Problem**: `config/pipeline.yaml` defines `sync_slides` step, but `autogen/conductor.py` implements `sync_video` step.

**Files affected**:
- `config/pipeline.yaml:283` - Defines `sync_slides` step
- `autogen/conductor.py:561` - Implements `sync_video` step
- `core/wrappers.py:763` - Has `sync_video()` wrapper function

### Issue 2: Missing sync_slides Step Implementation  
**Problem**: Conductor has no handler for `sync_slides` step, which is what's actually configured.

### Issue 3: Incorrect RVC Manifest Path Handling
**Problem**: `splice_audio` step uses hardcoded manifest path that doesn't match RVC output structure.

**Location**: `autogen/conductor.py:436`

### Issue 4: Missing Subtitle Generation
**Problem**: `make_srt` step is configured but not implemented in conductor.

## Implementation Checklist

### Phase 1: Critical Fixes (Immediate Priority)

#### Task 1: Fix Step Name Mismatch
- [ ] **Option A**: Change `config/pipeline.yaml` line 283 from `sync_slides` to `sync_video`
- [x] **Option B**: Change `autogen/conductor.py` line 561 from `sync_video` to `sync_slides`
- [x] **Decision**: Choose Option B to maintain config consistency

#### Task 2: Implement sync_slides Step Handler
- [ ] Add `elif step_id == "sync_slides":` block in `autogen/conductor.py` after line 560
- [ ] Implement input validation (splice manifests, video analyses, stems)
- [ ] Create output directory `workspace/06_synchronized_videos`
- [ ] Add loop to process each stem with corresponding manifests
- [ ] Call `core.wrappers.sync_slides()` wrapper function
- [ ] Add logging for step completion

#### Task 3: Create sync_slides Wrapper Function
- [ ] Add `sync_slides()` function to `core/wrappers.py`
- [ ] Define function signature with all required parameters
- [ ] Build command for `cli/sync_slides.py`
- [ ] Add video analysis parameter handling
- [ ] Implement subprocess call with error handling
- [ ] Return structured result dictionary

#### Task 4: Fix RVC Manifest Path Issue
- [x] Replace hardcoded path in `autogen/conductor.py` line 436
- [x] Use `rvc_manifest.get("manifest_path")` from `all_rvc_manifests`
- [x] Add null check and error handling for missing paths
- [x] Test with actual RVC output structure

#### Task 5: Remove Undefined make_srt Step
- [ ] Remove `- id: make_srt` from `config/pipeline.yaml` line 307
- [ ] Verify no other references to make_srt in pipeline
- [ ] Document removal in commit message

### Phase 2: Testing and Validation

#### Task 6: Integration Testing
- [ ] Run full pipeline with `python autogen/conductor.py`
- [ ] Verify all steps execute without errors
- [ ] Check that each step produces expected output files
- [ ] Validate workspace directory structure

#### Task 7: Output Verification
- [ ] Confirm synchronized videos exist in `workspace/06_synchronized_videos/`
- [ ] Test video playback functionality
- [ ] Verify audio-video synchronization quality
- [ ] Check file naming conventions match stems

#### Task 8: Error Handling Validation
- [ ] Test pipeline with missing input files
- [ ] Verify graceful failure modes
- [ ] Check logging output for debugging information
- [ ] Test edge cases (empty manifests, corrupted files)

### Phase 3: Documentation and Cleanup

#### Task 9: Update Documentation
- [ ] Update `CLAUDE.md` with new pipeline status
- [ ] Document new wrapper function in code comments
- [ ] Add example usage for sync_slides step
- [ ] Update troubleshooting guide

#### Task 10: Code Quality
- [ ] Review all modified files for consistency
- [ ] Add proper type hints to new functions
- [ ] Ensure error messages are descriptive
- [ ] Verify import statements are correct

## Detailed Implementation Code

### sync_slides Step Implementation (Task 2)
```python
elif step_id == "sync_slides":
    # Check required inputs
    if 'all_splice_manifests' not in locals() or not all_splice_manifests:
        logging.warning(f"Skipping sync step '{step_id}': No splice manifests available.")
        continue
    
    if 'all_video_analyses' not in locals() or not all_video_analyses:
        logging.warning(f"Skipping sync step '{step_id}': No video analyses available.")
        continue
    
    if 'stems' not in locals() or not stems:
        logging.warning(f"Skipping sync step '{step_id}': No stems available.")
        continue
    
    # Implement slide synchronization
    all_sync_manifests = []
    sync_output_dir = os.path.join(workspace_root, "06_synchronized_videos")
    os.makedirs(sync_output_dir, exist_ok=True)
    
    for idx, stem in enumerate(stems):
        # Find corresponding manifests and files
        splice_manifest = all_splice_manifests[idx] if idx < len(all_splice_manifests) else None
        video_analysis = all_video_analyses[idx] if idx < len(all_video_analyses) else None
        
        # Find video file
        video_path = None
        for ext in ['.mov', '.mp4']:
            candidate_path = os.path.join(data_dir, f"{stem}{ext}")
            if os.path.exists(candidate_path):
                video_path = candidate_path
                break
        
        if not video_path or not splice_manifest or not video_analysis:
            logging.warning(f"Missing inputs for {stem} sync, skipping")
            continue
        
        # Use the sync_slides wrapper
        sync_manifest = core.wrappers.sync_slides(
            video_path=video_path,
            splice_manifest=splice_manifest.get("manifest_path"),
            video_analysis=video_analysis.get("manifest_path") if isinstance(video_analysis, dict) else None,
            output_dir=sync_output_dir,
            stem=stem,
            config_path=args.pipeline_config,
            step_id=step_id
        )
        
        all_sync_manifests.append(sync_manifest)
        logging.info(f"Slide sync completed for {stem}")
    
    logging.info(f"Sync step '{step_id}' completed. Generated {len(all_sync_manifests)} synchronized videos.")
```

### sync_slides Wrapper Function (Task 3)
```python
def sync_slides(
    video_path: str,
    splice_manifest: str,
    video_analysis: str = None,
    output_dir: str = ".",
    stem: str = "output",
    config_path: str = "config/pipeline.yaml",
    step_id: str = "sync_slides"
) -> Dict[str, Any]:
    """
    Wrapper for slide synchronization CLI.
    
    Args:
        video_path: Path to input video file
        splice_manifest: Path to splice manifest JSON
        video_analysis: Path to video analysis manifest
        output_dir: Output directory for synchronized video
        stem: Stem name for output file
        config_path: Path to pipeline configuration
        step_id: Step identifier
        
    Returns:
        Dict: Synchronization results
        
    Raises:
        RuntimeError: If synchronization fails
    """
    output_video = os.path.join(output_dir, f"{stem}_synchronized.mp4")
    
    cmd = [
        sys.executable, "cli/sync_slides.py",
        "--video", video_path,
        "--splice-manifest", splice_manifest,
        "--output", output_video,
        "--config", config_path
    ]
    
    if video_analysis:
        cmd.extend(["--video-analysis", video_analysis])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        return {
            "step_id": step_id,
            "video_path": video_path,
            "output_video": output_video,
            "splice_manifest": splice_manifest,
            "video_analysis": video_analysis,
            "status": "success"
        }
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Slide synchronization failed: {e.stderr}") from e
```

### RVC Manifest Path Fix (Task 4)
```python
# BEFORE (incorrect):
rvc_manifest_path = os.path.join(workspace_root, "03_rvc_audio", "rvc_conversion_manifest.json")

# AFTER (correct):
rvc_manifest_path = rvc_manifest.get("manifest_path")
if not rvc_manifest_path:
    logging.warning(f"No manifest path in RVC manifest {idx+1}, skipping splice")
    continue
```

## Expected Final Pipeline Flow

After completing all tasks:

1. ✅ `check_datasets` - Validates PPTX/MOV pairs
2. ✅ `extract_transcript` - Extracts PowerPoint notes  
3. ✅ `tts_run` - Generates speech audio
4. ✅ `qc_pronounce` - Quality control validation
5. ✅ `apply_rvc` - Voice conversion
6. ✅ `splice_audio` - Assembles continuous narration
7. ✅ `analyze_video` - Detects scene transitions
8. ✅ `sync_slides` - Synchronizes audio with video
9. ✅ **Output**: Final MP4 files in `workspace/06_synchronized_videos/`

## File Modifications Summary

- **`autogen/conductor.py`**: Add sync_slides step, fix RVC paths
- **`core/wrappers.py`**: Add sync_slides wrapper function  
- **`config/pipeline.yaml`**: Remove make_srt step

## Success Criteria

- [ ] Pipeline runs from start to finish without errors
- [ ] All configured steps execute successfully
- [ ] Final video files are generated and playable
- [ ] Audio-video synchronization is functional
- [ ] No hanging or undefined pipeline steps

This checklist-based plan allows for incremental progress tracking and ensures all critical issues are addressed systematically.