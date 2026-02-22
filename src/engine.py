# File: engine.py
# Core TTS model loading and speech generation logic for KittenTTS v0.8.

import logging
import inspect
import numpy as np
from typing import Any, Dict, Optional, Tuple
from kittentts import KittenTTS
from nlp import TextPreprocessor

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# --- Global Module Variables ---
tts_model: Optional[KittenTTS] = None
MODEL_LOADED: bool = False

# KittenTTS available voices
KITTEN_TTS_VOICES = [
    "Bella", "Jasper", "Luna", "Bruno",
    "Rosie", "Hugo", "Kiki", "Leo"
]

_ALLOWED_PAUSE_STRENGTHS = {"light", "medium", "strong"}
_ALLOWED_SPEAKER_LABEL_MODES = {"drop", "speak"}
_PROFILE_SANITIZED_KEYS = {
    "filter_table_artifacts",
    "filter_reference_artifacts",
    "filter_symbol_noise",
    "remove_punctuation",
    "normalize_pause_punctuation",
    "pause_strength",
    "dialogue_turn_splitting",
    "speaker_label_mode",
    "max_punct_run",
}
_TEXT_PREPROCESSOR_PARAM_NAMES = {
    name
    for name in inspect.signature(TextPreprocessor.__init__).parameters
    if name != "self"
}
_TEXT_OPTION_DEFAULTS: Dict[str, Any] = {
    "filter_table_artifacts": True,
    "filter_reference_artifacts": True,
    "filter_symbol_noise": True,
    "remove_punctuation": False,
    "normalize_pause_punctuation": True,
    "pause_strength": "medium",
    "dialogue_turn_splitting": False,
    "speaker_label_mode": "drop",
    "max_punct_run": 3,
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any, default: int, *, min_value: int, max_value: int) -> int:
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, int_value))


def _sanitize_text_options(raw_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw_options, dict):
        return {}

    sanitized: Dict[str, Any] = {}

    if "remove_punctuation" in raw_options:
        sanitized["remove_punctuation"] = _as_bool(
            raw_options.get("remove_punctuation"),
            default=_TEXT_OPTION_DEFAULTS["remove_punctuation"],
        )
    if "normalize_pause_punctuation" in raw_options:
        sanitized["normalize_pause_punctuation"] = _as_bool(
            raw_options.get("normalize_pause_punctuation"),
            default=_TEXT_OPTION_DEFAULTS["normalize_pause_punctuation"],
        )
    if "dialogue_turn_splitting" in raw_options:
        sanitized["dialogue_turn_splitting"] = _as_bool(
            raw_options.get("dialogue_turn_splitting"),
            default=_TEXT_OPTION_DEFAULTS["dialogue_turn_splitting"],
        )
    if "pause_strength" in raw_options:
        pause_strength = str(raw_options.get("pause_strength") or "").strip().lower()
        if pause_strength in _ALLOWED_PAUSE_STRENGTHS:
            sanitized["pause_strength"] = pause_strength
        else:
            logger.warning(
                "Invalid pause_strength '%s'. Falling back to default '%s'.",
                raw_options.get("pause_strength"),
                _TEXT_OPTION_DEFAULTS["pause_strength"],
            )
    if "speaker_label_mode" in raw_options:
        speaker_label_mode = str(raw_options.get("speaker_label_mode") or "").strip().lower()
        if speaker_label_mode in _ALLOWED_SPEAKER_LABEL_MODES:
            sanitized["speaker_label_mode"] = speaker_label_mode
        else:
            logger.warning(
                "Invalid speaker_label_mode '%s'. Falling back to default '%s'.",
                raw_options.get("speaker_label_mode"),
                _TEXT_OPTION_DEFAULTS["speaker_label_mode"],
            )
    if "max_punct_run" in raw_options:
        sanitized["max_punct_run"] = _as_int(
            raw_options.get("max_punct_run"),
            _TEXT_OPTION_DEFAULTS["max_punct_run"],
            min_value=1,
            max_value=6,
        )

    return sanitized


def _sanitize_profile_filter_options(raw_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw_options, dict):
        return {}

    sanitized: Dict[str, Any] = {}
    if "filter_table_artifacts" in raw_options:
        sanitized["filter_table_artifacts"] = _as_bool(
            raw_options.get("filter_table_artifacts"),
            default=_TEXT_OPTION_DEFAULTS["filter_table_artifacts"],
        )
    if "filter_reference_artifacts" in raw_options:
        sanitized["filter_reference_artifacts"] = _as_bool(
            raw_options.get("filter_reference_artifacts"),
            default=_TEXT_OPTION_DEFAULTS["filter_reference_artifacts"],
        )
    if "filter_symbol_noise" in raw_options:
        sanitized["filter_symbol_noise"] = _as_bool(
            raw_options.get("filter_symbol_noise"),
            default=_TEXT_OPTION_DEFAULTS["filter_symbol_noise"],
        )
    return sanitized


def resolve_text_options(request_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    text_processing_cfg = config_manager.get("text_processing", {})
    if not isinstance(text_processing_cfg, dict):
        text_processing_cfg = {}

    profiles = text_processing_cfg.get("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}

    active_profile = str(text_processing_cfg.get("active_profile") or "balanced").strip().lower()
    requested_profile = None
    if isinstance(request_overrides, dict) and request_overrides.get("profile"):
        requested_profile = str(request_overrides["profile"]).strip().lower()

    selected_profile_name = requested_profile or active_profile or "balanced"
    selected_profile = profiles.get(selected_profile_name, {})
    if not isinstance(selected_profile, dict):
        selected_profile = {}

    if not selected_profile:
        fallback_profile_name = active_profile if isinstance(profiles.get(active_profile), dict) else "balanced"
        fallback_profile = profiles.get(fallback_profile_name, {})
        if not isinstance(fallback_profile, dict):
            fallback_profile = {}
        if selected_profile_name != fallback_profile_name:
            logger.warning(
                "Unknown text profile '%s'. Falling back to '%s'.",
                selected_profile_name,
                fallback_profile_name,
            )
        selected_profile_name = fallback_profile_name
        selected_profile = fallback_profile

    effective_options = dict(_TEXT_OPTION_DEFAULTS)
    for key, value in selected_profile.items():
        if key not in _PROFILE_SANITIZED_KEYS:
            effective_options[key] = value
    effective_options.update(_sanitize_profile_filter_options(selected_profile))
    effective_options.update(_sanitize_text_options(selected_profile))

    if isinstance(request_overrides, dict):
        effective_options.update(_sanitize_text_options(request_overrides))

    effective_options["profile"] = selected_profile_name
    return effective_options


def _build_runtime_preprocessor(
    effective_text_options: Optional[Dict[str, Any]] = None,
) -> TextPreprocessor:
    """
    Build the server-level text preprocessor using runtime config toggles.
    """
    options = effective_text_options or resolve_text_options()
    preprocessor_options = {
        key: value for key, value in options.items() if key in _TEXT_PREPROCESSOR_PARAM_NAMES
    }
    return TextPreprocessor(**preprocessor_options)


def is_speakable_text(text: str) -> bool:
    if not text:
        return False
    return any(ch.isalnum() for ch in text)


def preprocess_text(
    text: str,
    text_options: Optional[Dict[str, Any]] = None,
    effective_text_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Centralized text preprocessing entrypoint for server request handling.
    Returns cleaned text and metadata useful for observability.
    """
    options = (
        effective_text_options
        if effective_text_options is not None
        else resolve_text_options(text_options)
    )
    preprocessor = _build_runtime_preprocessor(effective_text_options=options)
    cleaned_text, metadata = preprocessor.process_with_metadata(text)
    return cleaned_text, metadata


def load_model() -> bool:
    """
    Loads the KittenTTS model and initializes it.
    Updates global variables for model components.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global tts_model, MODEL_LOADED

    if MODEL_LOADED:
        logger.info("KittenTTS model is already loaded.")
        return True

    try:
        # Get model repository and cache path from config
        model_repo_id = config_manager.get_string(
            "model.repo_id", "KittenML/kitten-tts-nano-0.8-fp32"
        )
        model_device = config_manager.get_string("tts_engine.device", "auto")
        model_cache_path = config_manager.get_path(
            "paths.model_cache", "./model_cache", ensure_absolute=True
        )
        
        logger.info(f"Loading KittenTTS model from: {model_repo_id}")
        logger.info(f"Requested inference device: {model_device}")
        logger.info(f"Using cache directory: {model_cache_path}")

        # The KittenTTS package handles phonemizer, session, 
        # downloading, etc internally.
        tts_model = KittenTTS(
            model_repo_id,
            cache_dir=str(model_cache_path),
            device=model_device,
        )

        MODEL_LOADED = True
        logger.info("KittenTTS model loaded successfully.")
        return True

    except Exception as e:
        logger.error(f"Error loading KittenTTS model: {e}", exc_info=True)
        tts_model = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str, voice: str, speed: float = 1.0, clean_text: bool = True
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Synthesizes audio from text using the loaded KittenTTS model.

    Args:
        text: The text to synthesize.
        voice: Voice identifier (e.g., 'Jasper').
        speed: Speech speed factor (1.0 is normal speed).

    Returns:
        A tuple containing the audio waveform (numpy array) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global tts_model

    if not MODEL_LOADED or tts_model is None:
        logger.error("KittenTTS model is not loaded. Cannot synthesize audio.")
        return None, None

    if voice not in KITTEN_TTS_VOICES:
        logger.error(f"Voice '{voice}' not available. Available voices: {KITTEN_TTS_VOICES}")
        return None, None

    try:
        logger.debug(f"Synthesizing with voice='{voice}', speed={speed}")
        logger.debug(f"Input text (first 100 chars): '{text[:100]}...'")

        # Generate audio using KittenTTS package.
        audio = tts_model.generate(text, voice=voice, speed=speed, clean_text=clean_text)

        # KittenTTS uses 24kHz sample rate natively
        sample_rate = 24000

        logger.info(f"Successfully generated {len(audio)} audio samples at {sample_rate}Hz")
        return audio, sample_rate

    except ValueError as e:
        logger.warning("Text could not be synthesized cleanly: %s", e)
        return None, None
    except Exception as e:
        logger.error(f"Error during KittenTTS synthesis: {e}", exc_info=True)
        return None, None

# --- End File: engine.py ---
