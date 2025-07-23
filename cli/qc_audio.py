#!/usr/bin/env python3
"""
Audio Quality Control CLI for detecting and fixing TTS glitches.

This tool processes TTS audio manifests and:
1. Scores audio quality using MOS (Mean Opinion Score)
2. Transcribes audio using WhisperX for WER calculation
3. Identifies chunks that fail quality thresholds
4. Optionally re-synthesizes failed chunks up to max_attempts
5. Outputs updated manifest with QC results
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.qc_audio import score_mos, is_bad_alignment
from core.wrappers import whisper_transcribe
import librosa
import numpy as np

def transcribe_audio_chunk(wav_path: str, model: str = "large-v2") -> Dict[str, Any]:
    """
    Transcribe audio chunk using WhisperX for WER calculation.
    
    Args:
        wav_path: Path to audio file
        model: WhisperX model to use
        
    Returns:
        Dict with transcription text and confidence
    """
    try:
        # Use WhisperX via whisper_transcribe wrapper
        # The wrapper expects a list of file paths or dict format
        result = whisper_transcribe(
            in_data=[wav_path],
            model=model
        )
        
        # Extract text from result dict
        if result and isinstance(result, dict):
            # Result is a dict mapping file paths to transcriptions
            transcription = result.get(wav_path, '')
            return {
                'text': transcription.strip() if transcription else '',
                'confidence': 1.0 if transcription else 0.0
            }
        else:
            return {'text': '', 'confidence': 0.0}
            
    except Exception as e:
        logging.warning(f"Transcription failed for {wav_path}: {e}")
        return {'text': '', 'confidence': 0.0}

def process_audio_chunk(chunk: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Process individual audio chunk through QC pipeline.
    
    Args:
        chunk: TTS chunk from manifest
        args: CLI arguments with thresholds and settings
        
    Returns:
        QC result dictionary for this chunk
    """
    chunk_id = chunk.get("id", "unknown")
    wav_path = chunk.get("wav_path")
    original_text = chunk.get("text", "")
    
    if not wav_path or not Path(wav_path).exists():
        return {
            "chunk_id": chunk_id,
            "status": "error",
            "error": f"Audio file not found: {wav_path}",
            "mos_score": 0.0,
            "wer_score": 1.0,
            "attempts": 0
        }
    
    # Calculate MOS score
    mos_score = score_mos(wav_path)
    mos_pass = mos_score >= args.mos_threshold
    
    # Transcribe and calculate WER if enabled
    wer_score = 0.0
    wer_pass = True
    transcribed_text = ""
    
    if args.enable_transcription:
        transcription_result = transcribe_audio_chunk(wav_path, args.whisperx_model)
        transcribed_text = transcription_result['text']
        
        if transcribed_text and original_text:
            wer_pass = not is_bad_alignment(
                ref=original_text,
                hyp=transcribed_text,
                wer_th=args.wer_threshold
            )
            # Calculate actual WER score for reporting
            try:
                import jiwer
                wer_score = jiwer.wer(reference=original_text, hypothesis=transcribed_text)
            except:
                wer_score = 0.0 if wer_pass else 1.0
    
    # Additional quality checks
    clipping_result = {}
    silence_result = {}
    duration_result = {}
    
    if args.detect_clipping:
        clipping_result = detect_clipping(wav_path, threshold=0.95)
    
    if args.detect_silence:
        silence_result = detect_silence(wav_path, args.silence_threshold, min_duration=0.1)
    
    # Duration validation
    duration_result = validate_duration(wav_path, args.min_chunk_duration, args.max_chunk_duration)
    
    # Additional pass/fail criteria
    clipping_pass = not clipping_result.get("has_clipping", False) if args.detect_clipping else True
    silence_pass = not silence_result.get("has_unexpected_silence", False) if args.detect_silence else True
    duration_pass = duration_result.get("duration_valid", True)
    
    # Determine overall pass/fail
    overall_pass = mos_pass and wer_pass and clipping_pass and silence_pass and duration_pass
    
    qc_result = {
        "chunk_id": chunk_id,
        "original_path": wav_path,
        "original_text": original_text,
        "transcribed_text": transcribed_text,
        "mos_score": mos_score,
        "mos_pass": mos_pass,
        "wer_score": wer_score,
        "wer_pass": wer_pass,
        "clipping_analysis": clipping_result,
        "clipping_pass": clipping_pass,
        "silence_analysis": silence_result,
        "silence_pass": silence_pass,
        "duration_analysis": duration_result,
        "duration_pass": duration_pass,
        "overall_pass": overall_pass,
        "attempts": 1,
        "status": "pass" if overall_pass else "fail"
    }
    
    # Attempt re-synthesis if QC fails and we haven't exceeded max attempts
    if not overall_pass and args.max_attempts > 1:
        qc_result = attempt_resynthesis(chunk, qc_result, args)
    
    return qc_result

def attempt_resynthesis(original_chunk: Dict[str, Any], qc_result: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Attempt to re-synthesize failed audio chunk.
    
    This is a placeholder for Phase 2 implementation.
    Currently just logs the attempt and returns original result.
    """
    logging.info(f"Re-synthesis needed for chunk {qc_result['chunk_id']} (MOS: {qc_result['mos_score']:.2f}, WER: {qc_result['wer_score']:.2f})")
    
    # TODO: Implement re-synthesis logic
    # 1. Apply phoneme injection if available
    # 2. Re-run TTS with modified text
    # 3. Re-evaluate QC metrics
    # 4. Update qc_result with new attempt
    
    return qc_result


def detect_clipping(wav_path: str, threshold: float = 0.95) -> Dict[str, Any]:
    """
    Detect audio clipping by analyzing peak amplitudes.
    
    Args:
        wav_path: Path to audio file
        threshold: Amplitude threshold for clipping detection (0.0-1.0)
        
    Returns:
        Dict with clipping analysis results
    """
    try:
        audio, sr = librosa.load(wav_path, sr=None)
        
        # Find peaks above threshold
        clipping_samples = np.abs(audio) >= threshold
        clipping_count = np.sum(clipping_samples)
        clipping_percentage = (clipping_count / len(audio)) * 100
        
        return {
            "has_clipping": clipping_count > 0,
            "clipping_percentage": clipping_percentage,
            "clipping_samples": int(clipping_count),
            "total_samples": len(audio),
            "threshold_used": threshold
        }
    except Exception as e:
        logging.error(f"Error detecting clipping in {wav_path}: {e}")
        return {
            "has_clipping": False,
            "clipping_percentage": 0.0,
            "error": str(e)
        }


def detect_silence(wav_path: str, silence_threshold: float = -40, min_duration: float = 0.1) -> Dict[str, Any]:
    """
    Detect unexpected silence periods in audio.
    
    Args:
        wav_path: Path to audio file
        silence_threshold: dB threshold for silence detection
        min_duration: Minimum duration to consider as silence (seconds)
        
    Returns:
        Dict with silence analysis results
    """
    try:
        audio, sr = librosa.load(wav_path, sr=None)
        
        # Convert to dB
        audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
        
        # Find silent samples
        silent_samples = audio_db <= silence_threshold
        
        # Find continuous silent periods
        silent_periods = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silent_samples):
            if is_silent and not in_silence:
                silence_start = i
                in_silence = True
            elif not is_silent and in_silence:
                silence_duration = (i - silence_start) / sr
                if silence_duration >= min_duration:
                    silent_periods.append({
                        "start_time": silence_start / sr,
                        "end_time": i / sr,
                        "duration": silence_duration
                    })
                in_silence = False
        
        # Handle silence at end of file
        if in_silence:
            silence_duration = (len(silent_samples) - silence_start) / sr
            if silence_duration >= min_duration:
                silent_periods.append({
                    "start_time": silence_start / sr,
                    "end_time": len(silent_samples) / sr,
                    "duration": silence_duration
                })
        
        total_silence_duration = sum(period["duration"] for period in silent_periods)
        silence_percentage = (total_silence_duration / (len(audio) / sr)) * 100
        
        return {
            "has_unexpected_silence": len(silent_periods) > 0,
            "silent_periods": silent_periods,
            "total_silence_duration": total_silence_duration,
            "silence_percentage": silence_percentage,
            "threshold_used": silence_threshold
        }
    except Exception as e:
        logging.error(f"Error detecting silence in {wav_path}: {e}")
        return {
            "has_unexpected_silence": False,
            "silent_periods": [],
            "error": str(e)
        }


def validate_duration(wav_path: str, min_duration: float, max_duration: float) -> Dict[str, Any]:
    """
    Validate audio chunk duration is within expected range.
    
    Args:
        wav_path: Path to audio file
        min_duration: Minimum expected duration (seconds)
        max_duration: Maximum expected duration (seconds)
        
    Returns:
        Dict with duration validation results
    """
    try:
        audio, sr = librosa.load(wav_path, sr=None)
        duration = len(audio) / sr
        
        is_valid = min_duration <= duration <= max_duration
        
        return {
            "duration_valid": is_valid,
            "actual_duration": duration,
            "min_expected": min_duration,
            "max_expected": max_duration,
            "duration_issue": None if is_valid else (
                "too_short" if duration < min_duration else "too_long"
            )
        }
    except Exception as e:
        logging.error(f"Error validating duration for {wav_path}: {e}")
        return {
            "duration_valid": False,
            "actual_duration": 0.0,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Audio Quality Control for TTS with comprehensive features")
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Input TTS manifest JSON file")
    parser.add_argument("--output", required=True, help="Output directory for QC results")
    
    # Quality thresholds
    parser.add_argument("--mos-threshold", type=float, default=3.5, help="Minimum MOS score threshold")
    parser.add_argument("--wer-threshold", type=float, default=0.10, help="Maximum WER threshold")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum re-synthesis attempts")
    
    # Transcription settings
    parser.add_argument("--whisper-model", default="large-v3", help="WhisperX model for transcription")
    parser.add_argument("--enable-transcription", action="store_true", help="Enable WER checking via transcription")
    parser.add_argument("--transcription-timeout", type=int, default=30, help="Timeout for transcription in seconds")
    
    # Re-synthesis strategy flags
    parser.add_argument("--retry-with-phonemes", action="store_true", help="Use phoneme hints for failed chunks")
    parser.add_argument("--retry-different-engine", action="store_true", help="Try alternate TTS engine if available")
    parser.add_argument("--preserve-original-on-failure", action="store_true", help="Keep original audio if all retries fail")
    
    # Audio issue detection
    parser.add_argument("--detect-clipping", action="store_true", help="Detect audio clipping")
    parser.add_argument("--detect-silence", action="store_true", help="Detect unexpected silence periods")
    parser.add_argument("--silence-threshold", type=float, default=-40, help="dB threshold for silence detection")
    
    # Duration validation
    parser.add_argument("--min-chunk-duration", type=float, default=0.5, help="Minimum expected chunk duration (seconds)")
    parser.add_argument("--max-chunk-duration", type=float, default=30.0, help="Maximum expected chunk duration (seconds)")
    
    # System settings
    parser.add_argument("--step-id", default="qc_pronounce", help="Step identifier for logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load input manifest
    try:
        with open(args.input, 'r') as f:
            tts_manifest = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load input manifest {args.input}: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each audio chunk
    qc_results = []
    chunks_processed = 0
    chunks_passed = 0
    chunks_fixed = 0
    
    tts_results = tts_manifest.get("tts_results", [])
    if not tts_results:
        logging.warning("No TTS results found in manifest")
        
    for chunk in tts_results:
        qc_result = process_audio_chunk(chunk, args)
        qc_results.append(qc_result)
        
        chunks_processed += 1
        if qc_result["status"] == "pass":
            chunks_passed += 1
        if qc_result["attempts"] > 1:
            chunks_fixed += 1
            
        logging.info(f"Processed chunk {qc_result['chunk_id']}: {qc_result['status']} "
                    f"(MOS: {qc_result['mos_score']:.2f}, WER: {qc_result['wer_score']:.2f})")
    
    # Create QC manifest
    manifest_stem = Path(args.input).stem
    qc_manifest = {
        "step_id": args.step_id,
        "input_manifest": args.input,
        "output_dir": str(output_dir),
        "parameters": {
            "mos_threshold": args.mos_threshold,
            "wer_threshold": args.wer_threshold,
            "max_attempts": args.max_attempts,
            "whisper_model": args.whisper_model,
            "enable_transcription": args.enable_transcription,
            "transcription_timeout": args.transcription_timeout,
            "retry_with_phonemes": args.retry_with_phonemes,
            "retry_different_engine": args.retry_different_engine,
            "preserve_original_on_failure": args.preserve_original_on_failure,
            "detect_clipping": args.detect_clipping,
            "detect_silence": args.detect_silence,
            "silence_threshold": args.silence_threshold,
            "min_chunk_duration": args.min_chunk_duration,
            "max_chunk_duration": args.max_chunk_duration
        },
        "summary": {
            "chunks_processed": chunks_processed,
            "chunks_passed": chunks_passed,
            "chunks_failed": chunks_processed - chunks_passed,
            "chunks_fixed": chunks_fixed,
            "pass_rate": chunks_passed / chunks_processed if chunks_processed > 0 else 0.0
        },
        "qc_results": qc_results
    }
    
    # Save QC manifest
    qc_manifest_path = output_dir / f"qc_manifest_{manifest_stem}.json"
    try:
        with open(qc_manifest_path, 'w') as f:
            json.dump(qc_manifest, f, indent=2)
        logging.info(f"QC manifest saved to {qc_manifest_path}")
    except Exception as e:
        logging.error(f"Failed to save QC manifest: {e}")
        return 1
    
    # Print summary
    print(f"QC Summary:")
    print(f"  Processed: {chunks_processed} chunks")
    print(f"  Passed: {chunks_passed} chunks ({chunks_passed/chunks_processed*100:.1f}%)" if chunks_processed > 0 else "  No chunks processed")
    print(f"  Failed: {chunks_processed - chunks_passed} chunks")
    print(f"  Fixed: {chunks_fixed} chunks")
    print(f"  Output: {qc_manifest_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())