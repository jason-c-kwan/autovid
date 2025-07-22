"""
Quality control functions for audio data.
"""
import jiwer
import cmudict
import pronouncing
import re
from typing import Dict, Optional, List

# Optional import for speechmetrics - fallback if not available
try:
    import speechmetrics
    SPEECHMETRICS_AVAILABLE = True
except ImportError:
    SPEECHMETRICS_AVAILABLE = False
    print("Warning: speechmetrics not available. Using basic audio analysis for MOS scoring.")

# Import librosa for basic audio analysis
try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available. MOS scoring will use fallback values.")

# --- CMUdict Caching ---
_cmudict_cache: Optional[Dict[str, List[List[str]]]] = None

def get_cmudict() -> Dict[str, List[List[str]]]:
    """
    Loads and caches the CMU Pronouncing Dictionary.

    Returns:
        The CMU dictionary.
    """
    global _cmudict_cache
    if _cmudict_cache is None:
        try:
            _cmudict_cache = cmudict.dict()
        except LookupError:
            # This can happen if the cmudict corpus is not downloaded
            # Attempt to download it if nltk is available and configured for downloads
            try:
                import nltk # type: ignore
                # Check if already downloaded, otherwise download
                try:
                    nltk.data.find('corpora/cmudict')
                except LookupError: # NLTK raises LookupError if resource not found
                    print("CMUdict not found via nltk.data.find, attempting to download with NLTK...")
                    nltk.download('cmudict', quiet=True)
                _cmudict_cache = cmudict.dict() # Try loading again
            except ImportError:
                print("NLTK is not installed. Cannot automatically download CMUdict.")
                print("Please install NLTK and download the CMUdict corpus manually or ensure it's available.")
                _cmudict_cache = {} # Return empty dict to avoid repeated errors
            except Exception as e:
                print(f"An error occurred while trying to download/load CMUdict: {e}")
                _cmudict_cache = {}
    return _cmudict_cache if _cmudict_cache is not None else {}


# --- Audio Analysis Helper Functions ---
def analyze_audio_quality(wav_path: str) -> Dict[str, float]:
    """
    Analyze basic audio quality metrics using librosa.
    
    Args:
        wav_path: Path to WAV audio file
        
    Returns:
        Dict containing quality metrics:
        - snr: Signal-to-noise ratio estimate
        - clipping_rate: Percentage of samples that are clipped
        - rms_level: RMS level in dB
        - silence_ratio: Proportion of silent frames
        - spectral_centroid: Average spectral centroid (brightness)
    """
    if not LIBROSA_AVAILABLE:
        return {
            'snr': 20.0,
            'clipping_rate': 0.0,
            'rms_level': -20.0,
            'silence_ratio': 0.1,
            'spectral_centroid': 2000.0
        }
    
    try:
        # Load audio
        y, sr = librosa.load(wav_path, sr=None)
        
        # 1. Clipping detection
        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) >= clipping_threshold)
        clipping_rate = clipped_samples / len(y)
        
        # 2. RMS level calculation
        rms = librosa.feature.rms(y=y)[0]
        rms_db = 20 * np.log10(np.mean(rms) + 1e-10)  # Avoid log(0)
        
        # 3. Silence detection
        silence_threshold = 0.01
        silent_frames = np.sum(rms < silence_threshold)
        silence_ratio = silent_frames / len(rms)
        
        # 4. SNR estimation using spectral subtraction approach
        # Estimate noise from quiet portions
        noise_frames = rms < np.percentile(rms, 10)  # Bottom 10% as noise
        if np.sum(noise_frames) > 0:
            noise_power = np.mean(rms[noise_frames] ** 2)
            signal_power = np.mean(rms ** 2)
            snr = 10 * np.log10((signal_power - noise_power) / (noise_power + 1e-10))
            snr = max(0, min(40, snr))  # Clamp to reasonable range
        else:
            snr = 20.0  # Default good SNR
        
        # 5. Spectral centroid (brightness indicator)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_spectral_centroid = np.mean(spectral_centroids)
        
        return {
            'snr': float(snr),
            'clipping_rate': float(clipping_rate),
            'rms_level': float(rms_db),
            'silence_ratio': float(silence_ratio),
            'spectral_centroid': float(avg_spectral_centroid)
        }
        
    except Exception as e:
        print(f"Error analyzing audio quality for {wav_path}: {e}")
        # Return neutral values on error
        return {
            'snr': 20.0,
            'clipping_rate': 0.0,
            'rms_level': -20.0,
            'silence_ratio': 0.1,
            'spectral_centroid': 2000.0
        }

def compute_mos_from_audio_metrics(metrics: Dict[str, float]) -> float:
    """
    Compute MOS score from basic audio quality metrics.
    
    Args:
        metrics: Dict of audio quality metrics from analyze_audio_quality()
        
    Returns:
        MOS score (1.0 - 5.0)
    """
    score = 5.0  # Start with perfect score
    
    # Penalize high clipping rate
    if metrics['clipping_rate'] > 0.01:  # More than 1% clipped
        score -= min(2.0, metrics['clipping_rate'] * 200)  # Heavy penalty
    
    # Penalize poor SNR
    snr = metrics['snr']
    if snr < 10:
        score -= (10 - snr) * 0.1  # 0.1 point per dB below 10
    elif snr > 35:
        score -= 0.2  # Slight penalty for unrealistically high SNR
    
    # Penalize inappropriate volume levels
    rms_db = metrics['rms_level']
    if rms_db < -40:  # Too quiet
        score -= min(1.5, (-40 - rms_db) * 0.05)
    elif rms_db > -6:  # Too loud (likely distorted)
        score -= min(1.5, (rms_db + 6) * 0.1)
    
    # Penalize excessive silence
    if metrics['silence_ratio'] > 0.3:  # More than 30% silence
        score -= min(1.0, (metrics['silence_ratio'] - 0.3) * 2)
    
    # Penalize unnatural spectral characteristics
    centroid = metrics['spectral_centroid']
    if centroid < 500 or centroid > 8000:  # Outside normal speech range
        score -= 0.3
    
    # Clamp to valid MOS range
    return max(1.0, min(5.0, score))

# --- MOS Scoring ---
_mosnet_model = None

def score_mos(wav_path: str) -> float:
    """
    Scores the Mean Opinion Score (MOS) of a WAV file.
    
    Uses SpeechMetrics MOSNet if available, otherwise falls back to 
    basic audio analysis using librosa.

    Args:
        wav_path: Path to the WAV audio file.

    Returns:
        A float representing the MOS score (1.0-5.0).
        Uses basic audio analysis fallback if speechmetrics not available.
    """
    # Try speechmetrics first if available
    if SPEECHMETRICS_AVAILABLE:
        global _mosnet_model
        if _mosnet_model is None:
            try:
                _mosnet_model = speechmetrics.load('mosnet')
            except Exception as e:
                print(f"Failed to load MOSNet model: {e}")
                _mosnet_model = False  # Mark as failed

        if _mosnet_model:
            try:
                metrics = _mosnet_model(wav_path)
                # speechmetrics can return a dict with various metrics.
                # We expect 'mosnet' to be a key, and its value to be a list of scores.
                if 'mosnet' in metrics and isinstance(metrics['mosnet'], list) and metrics['mosnet']:
                    return float(metrics['mosnet'][0])
                else:
                    print(f"Unexpected MOSNet output format for {wav_path}: {metrics}")
            except Exception as e:
                print(f"Error scoring MOS with speechmetrics for {wav_path}: {e}")
    
    # Fallback to basic audio analysis
    if LIBROSA_AVAILABLE:
        try:
            audio_metrics = analyze_audio_quality(wav_path)
            return compute_mos_from_audio_metrics(audio_metrics)
        except Exception as e:
            print(f"Error with basic audio analysis for {wav_path}: {e}")
    
    # Final fallback to neutral score
    print(f"Warning: Using fallback MOS score for {wav_path}")
    return 4.0


# --- Word Error Rate and OOV Checking ---
def is_bad_alignment(
    ref: str,
    hyp: str,
    wer_th: float = 0.10,
    oov_dict: Optional[Dict[str, str]] = None
) -> bool:
    """
    Checks if the alignment between reference and hypothesis is bad based on
    Word Error Rate (WER) and Out-Of-Vocabulary (OOV) words.

    Args:
        ref: The reference transcript.
        hyp: The hypothesis transcript.
        wer_th: The Word Error Rate threshold. If WER > wer_th, it's considered bad.
        oov_dict: An optional dictionary of allowed OOV words and their pronunciations.
                  If an OOV word from `hyp` is not in this dictionary, it's considered bad.

    Returns:
        True if the alignment is bad, False otherwise.
    """
    if not ref.strip() and not hyp.strip(): # Both empty, perfect alignment
        return False
    if not ref.strip() and hyp.strip(): # Ref empty, hyp has content -> 100% WER
        return True
    # If hyp is empty but ref is not, jiwer will calculate WER as 1.0

    try:
        wer = jiwer.wer(reference=ref, hypothesis=hyp)
        if wer > wer_th:
            return True
    except ValueError as e:
        # jiwer can raise ValueError if e.g. ref is empty and hyp is not for certain internal calcs
        # or if inputs are not strings.
        # If ref is empty and hyp is not, it's 100% error.
        if not ref.strip() and hyp.strip():
            return True
        print(f"Error calculating WER for ref='{ref}', hyp='{hyp}': {e}")
        return True # Treat calculation errors as bad alignment

    # OOV Check
    # Normalize text for OOV checking: lowercase and remove punctuation
    # A more sophisticated tokenizer might be needed for some languages/cases
    hyp_words = re.findall(r"\b\w+\b", hyp.lower())
    
    cmu_dictionary = get_cmudict()

    for word in hyp_words:
        is_in_cmudict = word in cmu_dictionary
        has_pronouncing_entry = bool(pronouncing.phones_for_word(word))

        if not is_in_cmudict and not has_pronouncing_entry:
            # Word is OOV according to both CMUdict and pronouncing
            if oov_dict is None or word not in oov_dict:
                return True  # OOV word found and not in the allowed list

    return False


# --- Phoneme Injection ---
def inject_phonemes(text: str, ipa_map: Dict[str, str], engine: str) -> str:
    """
    Injects phonemic pronunciations into text for TTS engines.

    Args:
        text: The input text string.
        ipa_map: A dictionary mapping words to their phonemic representations.
                 For 'piper', this map should contain XSAMPA.
                 For 'orpheus', this map should contain IPA.
        engine: The TTS engine ('piper' or 'orpheus').

    Returns:
        The text string with phonemes injected.
    """
    if not text or not ipa_map:
        return text

    # Using regex to split by words while preserving spaces and punctuation
    # This allows more accurate reconstruction of the sentence.
    # It splits by sequences of word characters (\w+) and non-word characters (\W+).
    parts = re.findall(r"(\w+|\W+)", text)
    processed_parts = []

    for part in parts:
        # Check if the part is a word (consists of alphanumeric characters)
        if re.fullmatch(r"\w+", part):
            # Use lowercased part for map lookup, but preserve original case if not found
            # or if map keys are case-sensitive (assuming map keys are lowercase for consistency)
            word_to_lookup = part.lower()
            if word_to_lookup in ipa_map:
                phoneme = ipa_map[word_to_lookup]
                if engine == 'piper':
                    # Assumes ipa_map provides XSAMPA directly for Piper
                    processed_parts.append(phoneme)
                elif engine == 'orpheus':
                    # Formats IPA into Orpheus-specific tag
                    processed_parts.append(f'<phoneme ipa="{phoneme}"></phoneme>')
                else:
                    # Unknown engine, keep original word part
                    processed_parts.append(part)
            else:
                # Word not in ipa_map, keep original word part
                processed_parts.append(part)
        else:
            # Part is not a word (e.g., space, punctuation), keep as is
            processed_parts.append(part)
            
    return "".join(processed_parts)
