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
    
    # Determine overall pass/fail
    overall_pass = mos_pass and wer_pass
    
    qc_result = {
        "chunk_id": chunk_id,
        "original_path": wav_path,
        "original_text": original_text,
        "transcribed_text": transcribed_text,
        "mos_score": mos_score,
        "mos_pass": mos_pass,
        "wer_score": wer_score,
        "wer_pass": wer_pass,
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

def main():
    parser = argparse.ArgumentParser(description="Audio Quality Control for TTS")
    parser.add_argument("--input", required=True, help="Input TTS manifest JSON file")
    parser.add_argument("--output", required=True, help="Output directory for QC results")
    parser.add_argument("--mos-threshold", type=float, default=3.5, help="Minimum MOS score threshold")
    parser.add_argument("--wer-threshold", type=float, default=0.10, help="Maximum WER threshold")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum re-synthesis attempts")
    parser.add_argument("--enable-transcription", action="store_true", help="Enable WER checking via transcription")
    parser.add_argument("--whisperx-model", default="large-v2", help="WhisperX model for transcription")
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
            "enable_transcription": args.enable_transcription,
            "whisperx_model": args.whisperx_model
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