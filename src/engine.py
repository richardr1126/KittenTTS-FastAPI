# File: engine.py
# Core TTS model loading and speech generation logic for KittenTTS v0.8.

import logging
import numpy as np
from typing import Dict, Optional, Tuple
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

def _build_runtime_preprocessor() -> TextPreprocessor:
    """
    Build the server-level text preprocessor using runtime config toggles.
    """
    return TextPreprocessor(
        filter_table_artifacts=config_manager.get_bool(
            "text_processing.filter_table_artifacts", True
        ),
        filter_reference_artifacts=config_manager.get_bool(
            "text_processing.filter_reference_artifacts", True
        ),
        filter_symbol_noise=config_manager.get_bool(
            "text_processing.filter_symbol_noise", True
        ),
    )


def is_speakable_text(text: str) -> bool:
    if not text:
        return False
    return any(ch.isalnum() for ch in text)


def preprocess_text(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Centralized text preprocessing entrypoint for server request handling.
    Returns cleaned text and metadata useful for observability.
    """
    preprocessor = _build_runtime_preprocessor()
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
