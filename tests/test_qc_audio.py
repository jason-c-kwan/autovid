"""
Tests for core.qc_audio functions.
"""
import pytest
from unittest.mock import patch, MagicMock
from core import qc_audio
import nltk.downloader # For DownloadError

# --- Fixtures ---

@pytest.fixture
def sample_ipa_map_xsampa():
    """Sample ipa_map with XSAMPA for Piper."""
    return {
        "hello": "h@loU",
        "world": "w3`ld", # Using ` for r sometimes seen in XSAMPA
        "example": "Ig'z{mp@l"
    }

@pytest.fixture
def sample_ipa_map_ipa():
    """Sample ipa_map with IPA for Orpheus."""
    return {
        "hello": "həˈloʊ",
        "world": "ˈwɜrld",
        "example": "ɪɡˈzæmpəl"
    }

# --- Tests for get_cmudict ---

@patch('cmudict.dict')
def test_get_cmudict_loads_and_caches(mock_cmudict_dict):
    """Test that get_cmudict loads from cmudict.dict() and caches it."""
    mock_dict_content = {"test": [["T", "EH1", "S", "T"]]}
    mock_cmudict_dict.return_value = mock_dict_content
    
    qc_audio._cmudict_cache = None # Ensure cache is clear

    # First call - should load
    res1 = qc_audio.get_cmudict()
    assert res1 == mock_dict_content
    mock_cmudict_dict.assert_called_once()

    # Second call - should use cache
    res2 = qc_audio.get_cmudict()
    assert res2 == mock_dict_content
    mock_cmudict_dict.assert_called_once() # Still called only once

    qc_audio._cmudict_cache = None # Clean up

@patch('cmudict.dict')
@patch('nltk.download')
@patch('nltk.data.find')
def test_get_cmudict_nltk_download_flow(mock_nltk_find, mock_nltk_download, mock_cmudict_dict_load):
    """Test the NLTK download flow if cmudict initially fails."""
    
    # Simulate cmudict.dict() failing first, then succeeding after download
    final_dict_content = {"downloaded": [["D", "L"]]}
    
    # nltk.data.find raises LookupError when a resource is not found.
    mock_nltk_find.side_effect = [LookupError("Simulated NLTK resource not found"), None] 
    
    # Configure mock_cmudict_dict_load to fail first, then succeed
    mock_cmudict_dict_load.side_effect = [
        LookupError("Test CMUdict not found on first try"), # First call fails
        final_dict_content  # Second call (after simulated download) succeeds
    ]
    
    qc_audio._cmudict_cache = None # Ensure cache is clear
    
    result = qc_audio.get_cmudict() # This should trigger the download flow
    
    assert mock_nltk_find.call_count == 1 # nltk.data.find is called to check
    mock_nltk_download.assert_called_once_with('cmudict', quiet=True) # nltk.download is called
    
    # cmudict.dict() should be called twice: once failing, once succeeding
    assert mock_cmudict_dict_load.call_count == 2 
    assert result == final_dict_content
    
    qc_audio._cmudict_cache = None # Clean up


# --- Tests for score_mos ---

@patch('speechmetrics.load')
def test_score_mos_success(mock_speechmetrics_load):
    """Test score_mos returns float when speechmetrics succeeds."""
    mock_model_instance = MagicMock()
    mock_model_instance.return_value = {'mosnet': [3.75]} # Simulate model call
    mock_speechmetrics_load.return_value = mock_model_instance
    
    qc_audio._mosnet_model = None # Ensure model is reloaded

    result = qc_audio.score_mos("dummy.wav")

    mock_speechmetrics_load.assert_called_once_with('mosnet')
    mock_model_instance.assert_called_once_with("dummy.wav")
    assert isinstance(result, float)
    assert result == 3.75
    
    # Test caching of model
    qc_audio.score_mos("dummy2.wav")
    mock_speechmetrics_load.assert_called_once() # Still called only once
    qc_audio._mosnet_model = None # Clean up

@patch('speechmetrics.load', side_effect=Exception("Model load failed"))
def test_score_mos_model_load_failure(mock_speechmetrics_load_fail):
    """Test score_mos returns 0.0 if model loading fails."""
    qc_audio._mosnet_model = None
    result = qc_audio.score_mos("dummy.wav")
    assert result == 0.0
    mock_speechmetrics_load_fail.assert_called_once_with('mosnet')
    qc_audio._mosnet_model = None

@patch('speechmetrics.load')
def test_score_mos_scoring_failure(mock_speechmetrics_load_scoring_fail):
    """Test score_mos returns 0.0 if scoring call fails."""
    mock_model_instance = MagicMock(side_effect=Exception("Scoring failed"))
    mock_speechmetrics_load_scoring_fail.return_value = mock_model_instance
    qc_audio._mosnet_model = None

    result = qc_audio.score_mos("dummy.wav")
    assert result == 0.0
    qc_audio._mosnet_model = None

@patch('speechmetrics.load')
def test_score_mos_unexpected_format(mock_speechmetrics_load_bad_format):
    """Test score_mos returns 0.0 if speechmetrics returns unexpected format."""
    mock_model_instance = MagicMock()
    mock_model_instance.return_value = {'something_else': [1.0]} # Wrong key
    mock_speechmetrics_load_bad_format.return_value = mock_model_instance
    qc_audio._mosnet_model = None

    result = qc_audio.score_mos("dummy.wav")
    assert result == 0.0
    qc_audio._mosnet_model = None


# --- Tests for is_bad_alignment ---

@patch('core.qc_audio.get_cmudict')
@patch('pronouncing.phones_for_word')
def test_is_bad_alignment_wer_threshold(mock_phones_for_word, mock_get_cmu):
    """Test WER threshold logic."""
    mock_get_cmu.return_value = {"word": [["W", "ER0", "D"]]} # Dummy CMU dict
    mock_phones_for_word.return_value = ["P", "R", "O", "N"] # Dummy pronouncing

    # WER = 0.5 (2 substitutions / 4 words in ref)
    assert qc_audio.is_bad_alignment("this is a test", "this was a task", wer_th=0.10) == True
    # WER = 0.25 (1 substitution / 4 words in ref)
    assert qc_audio.is_bad_alignment("this is a test", "this is a task", wer_th=0.30) == False
    assert qc_audio.is_bad_alignment("this is a test", "this is a task", wer_th=0.20) == True # Edge case for >
    
    # Perfect match
    assert qc_audio.is_bad_alignment("test sentence", "test sentence", wer_th=0.05) == False
    
    # Empty strings
    assert qc_audio.is_bad_alignment("", "", wer_th=0.10) == False
    assert qc_audio.is_bad_alignment("not empty", "", wer_th=0.10) == True # WER = 1.0
    assert qc_audio.is_bad_alignment("", "not empty", wer_th=0.10) == True # WER = 1.0

@patch('core.qc_audio.get_cmudict')
@patch('pronouncing.phones_for_word')
def test_is_bad_alignment_oov_logic(mock_phones_for_word, mock_get_cmu):
    """Test OOV detection logic."""
    # Setup: 'oovword' is not in CMU or pronouncing
    # Ensure 'a' and 'word' are considered known to avoid false positives from them
    mock_get_cmu.return_value = {
        "a": [["AH0"]],
        "known": [["K", "N", "O", "W", "N"]],
        "word": [["W", "ER0", "D"]]
    }
    
    def phones_side_effect(token):
        if token == "a": return [["AH0"]]
        if token == "known": return [["K", "N", "O", "W", "N"]]
        if token == "word": return [["W", "ER0", "D"]]
        return [] # For 'oovword' or any other OOV
    mock_phones_for_word.side_effect = phones_side_effect

    # Case 1: OOV word 'oovword' present, no oov_dict -> bad
    # "a known word" vs "a known oovword" -> WER is 1/3 = 0.33
    assert qc_audio.is_bad_alignment("a known word", "a known oovword", wer_th=0.50) == True 
    
    # Case 2: OOV word 'oovword' present, but in oov_dict, WER is fine -> not bad
    assert qc_audio.is_bad_alignment("a known word", "a known oovword", wer_th=0.50, oov_dict={"oovword": "pron"}) == False
    
    # Case 3: OOV word 'oovword' present, in oov_dict, but WER is too high -> bad
    assert qc_audio.is_bad_alignment("a known word", "a known oovword", wer_th=0.10, oov_dict={"oovword": "pron"}) == True

    # Case 4: No OOV words, WER is fine -> not bad
    assert qc_audio.is_bad_alignment("a known word", "a known word", wer_th=0.10) == False

    # Case 5: OOV word 'xyzabc' (not in CMU, not in pronouncing, not in oov_dict)
    # "test" vs "test xyzabc" -> WER is 1/2 = 0.5
    # Make "test" known for this specific assertion
    mock_get_cmu.return_value.update({"test": [["T", "EH1", "S", "T"]]})
    def phones_side_effect_test(token):
        if token == "test": return [["T", "EH1", "S", "T"]]
        return []
    mock_phones_for_word.side_effect = phones_side_effect_test
    assert qc_audio.is_bad_alignment("test", "test xyzabc", wer_th=0.80) == True


# --- Tests for inject_phonemes ---

def test_inject_phonemes_piper(sample_ipa_map_xsampa):
    """Test inject_phonemes for Piper engine (XSAMPA)."""
    text = "Hello world, this is an example."
    expected = "h@loU w3`ld, this is an Ig'z{mp@l."
    # Assuming map keys are lowercase
    lowercase_map = {k.lower(): v for k,v in sample_ipa_map_xsampa.items()}
    result = qc_audio.inject_phonemes(text, lowercase_map, "piper")
    assert result == expected

def test_inject_phonemes_orpheus(sample_ipa_map_ipa):
    """Test inject_phonemes for Orpheus engine (IPA tags)."""
    text = "Hello world, this is an example."
    expected = '<phoneme ipa="həˈloʊ"></phoneme> <phoneme ipa="ˈwɜrld"></phoneme>, this is an <phoneme ipa="ɪɡˈzæmpəl"></phoneme>.'
    lowercase_map = {k.lower(): v for k,v in sample_ipa_map_ipa.items()}
    result = qc_audio.inject_phonemes(text, lowercase_map, "orpheus")
    assert result == expected

def test_inject_phonemes_unknown_engine(sample_ipa_map_ipa):
    """Test inject_phonemes with an unknown engine keeps original text for mapped words."""
    text = "Hello world example"
    lowercase_map = {k.lower(): v for k,v in sample_ipa_map_ipa.items()}
    result = qc_audio.inject_phonemes(text, lowercase_map, "unknown_engine")
    assert result == "Hello world example" # Words from map are not replaced

def test_inject_phonemes_word_not_in_map(sample_ipa_map_ipa):
    """Test words not in the map remain unchanged."""
    text = "Hello unknownword example"
    expected_orpheus = '<phoneme ipa="həˈloʊ"></phoneme> unknownword <phoneme ipa="ɪɡˈzæmpəl"></phoneme>'
    lowercase_map = {k.lower(): v for k,v in sample_ipa_map_ipa.items()}
    result = qc_audio.inject_phonemes(text, lowercase_map, "orpheus")
    assert result == expected_orpheus

def test_inject_phonemes_empty_text():
    """Test with empty text."""
    assert qc_audio.inject_phonemes("", {"hello": "həloʊ"}, "orpheus") == ""

def test_inject_phonemes_empty_map():
    """Test with empty ipa_map."""
    assert qc_audio.inject_phonemes("Hello world", {}, "orpheus") == "Hello world"

def test_inject_phonemes_preserves_punctuation_and_spacing(sample_ipa_map_ipa):
    """Test that punctuation and spacing are preserved."""
    text = "Hello, world! This is an example sentence."
    # Assuming map keys are lowercase
    lowercase_map = {
        "hello": sample_ipa_map_ipa["hello"],
        "world": sample_ipa_map_ipa["world"],
        "example": sample_ipa_map_ipa["example"]
    }
    expected_orpheus = '<phoneme ipa="həˈloʊ"></phoneme>, <phoneme ipa="ˈwɜrld"></phoneme>! This is an <phoneme ipa="ɪɡˈzæmpəl"></phoneme> sentence.'
    result = qc_audio.inject_phonemes(text, lowercase_map, "orpheus")
    assert result == expected_orpheus

    text_piper = "Hello, world! Example."
    xsampa_map = {
        "hello": "h@loU",
        "world": "w3`ld",
        "example": "Ig'z{mp@l"
    }
    expected_piper = "h@loU, w3`ld! Ig'z{mp@l."
    result_piper = qc_audio.inject_phonemes(text_piper, xsampa_map, "piper")
    assert result_piper == expected_piper

def test_inject_phonemes_case_insensitivity_in_text(sample_ipa_map_ipa):
    """Test that word lookup is case-insensitive for words in text, assuming map keys are lowercase."""
    text = "HELLO WoRlD eXaMpLe"
    # Map keys are already lowercase in fixture helper
    lowercase_map = {k.lower(): v for k,v in sample_ipa_map_ipa.items()}
    expected_orpheus = '<phoneme ipa="həˈloʊ"></phoneme> <phoneme ipa="ˈwɜrld"></phoneme> <phoneme ipa="ɪɡˈzæmpəl"></phoneme>'
    result = qc_audio.inject_phonemes(text, lowercase_map, "orpheus")
    assert result == expected_orpheus
