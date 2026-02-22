# utils.py
# Utility functions for the TTS server application.
# This module includes functions for audio processing, text manipulation,
# file system operations, and performance monitoring.

import os
import logging
import time
import io
import uuid
from pathlib import Path
from typing import Optional, Tuple, List
from pydub import AudioSegment

import numpy as np
import soundfile as sf

# Optional import for librosa (for audio resampling, e.g., Opus encoding and time stretching)
try:
    import librosa

    LIBROSA_AVAILABLE = True
    logger = logging.getLogger(
        __name__
    )  # Initialize logger here if librosa is available
    logger.info(
        "Librosa library found and will be used for audio resampling and time stretching."
    )
except ImportError:
    LIBROSA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "Librosa library not found. Advanced audio resampling features (e.g., for Opus encoding) "
        "and pitch-preserving speed adjustment will be limited. Speed adjustment will fall back to basic method if enabled."
    )


# --- Filename Sanitization ---
def sanitize_filename(filename: str) -> str:
    """
    Removes potentially unsafe characters and path components from a filename
    to make it safe for use in file paths. Replaces unsafe sequences with underscores.

    Args:
        filename: The original filename string.

    Returns:
        A sanitized filename string, ensuring it's not empty and reasonably short.
    """
    if not filename:
        # Generate a unique name if the input is empty.
        return f"unnamed_file_{uuid.uuid4().hex[:8]}"

    # Remove directory separators and leading/trailing whitespace.
    base_filename = Path(filename).name.strip()
    if not base_filename:
        return f"empty_basename_{uuid.uuid4().hex[:8]}"

    # Define a set of allowed characters (alphanumeric, underscore, hyphen, dot, space).
    # Spaces will be replaced by underscores later.
    safe_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- "
    )
    sanitized_list = []
    last_char_was_underscore = False

    for char in base_filename:
        if char in safe_chars:
            # Replace spaces with underscores.
            sanitized_list.append("_" if char == " " else char)
            last_char_was_underscore = char == " "
        elif not last_char_was_underscore:
            # Replace any disallowed character sequence with a single underscore.
            sanitized_list.append("_")
            last_char_was_underscore = True

    sanitized = "".join(sanitized_list).strip("_")

    # Prevent names starting with multiple dots or consisting only of dots/underscores.
    if not sanitized or sanitized.lstrip("._") == "":
        return f"sanitized_file_{uuid.uuid4().hex[:8]}"

    # Limit filename length (e.g., 100 characters), preserving the extension.
    max_len = 100
    if len(sanitized) > max_len:
        name_part, ext_part = os.path.splitext(sanitized)
        # Ensure extension is not overly long itself; common extensions are short.
        ext_part = ext_part[:10]  # Limit extension length just in case.
        name_part = name_part[
            : max_len - len(ext_part) - 1
        ]  # -1 for the dot if ext exists
        sanitized = name_part + ext_part
        logger.warning(
            f"Original filename '{base_filename}' was truncated to '{sanitized}' due to length limits."
        )

    if not sanitized:  # Should not happen with previous checks, but as a failsafe.
        return f"final_fallback_name_{uuid.uuid4().hex[:8]}"

    return sanitized


# --- Audio Processing Utilities ---
def _to_mono_float32(audio_array: np.ndarray) -> np.ndarray:
    """
    Normalize audio input to a mono float32 array in approximately [-1, 1].
    """
    if audio_array.dtype != np.float32:
        if np.issubdtype(audio_array.dtype, np.integer):
            max_val = np.iinfo(audio_array.dtype).max
            audio_array = audio_array.astype(np.float32) / max_val
        else:
            audio_array = audio_array.astype(np.float32)
        logger.debug("Converted audio array to float32 for encoding.")

    if audio_array.ndim == 2 and audio_array.shape[1] == 1:
        audio_array = audio_array.squeeze(axis=1)
        logger.debug("Squeezed audio array from (samples, 1) to (samples,) for encoding.")
    elif audio_array.ndim > 1:
        logger.warning(
            "Multi-channel audio (shape: %s) provided to encode_audio. Using only the first channel.",
            audio_array.shape,
        )
        audio_array = audio_array[:, 0]

    return audio_array


def _to_pcm16(audio_array: np.ndarray) -> np.ndarray:
    audio_clipped = np.clip(audio_array, -1.0, 1.0)
    return (audio_clipped * 32767).astype(np.int16)


def encode_audio(
    audio_array: np.ndarray,
    sample_rate: int,
    output_format: str = "opus",
    target_sample_rate: Optional[int] = None,
) -> Optional[bytes]:
    """
    Encodes a NumPy audio array into the specified format (Opus or WAV) in memory.
    Can resample the audio to a target sample rate before encoding if specified.

    Args:
        audio_array: NumPy array containing audio data (expected as float32, range [-1, 1]).
        sample_rate: Sample rate of the input audio data.
        output_format: Desired output format ('opus', 'wav' or 'mp3').
        target_sample_rate: Optional target sample rate to resample to before encoding.

    Returns:
        Bytes object containing the encoded audio, or None if encoding fails.
    """
    if audio_array is None or audio_array.size == 0:
        logger.warning("encode_audio received empty or None audio array.")
        return None

    audio_array = _to_mono_float32(audio_array)

    # Resample if target_sample_rate is provided and different from current sample_rate
    if (
        target_sample_rate is not None
        and target_sample_rate != sample_rate
        and LIBROSA_AVAILABLE
    ):
        try:
            logger.info(
                f"Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz using Librosa."
            )
            audio_array = librosa.resample(
                y=audio_array, orig_sr=sample_rate, target_sr=target_sample_rate
            )
            sample_rate = (
                target_sample_rate  # Update sample_rate for subsequent encoding
            )
        except Exception as e_resample:
            logger.error(
                f"Error resampling audio to {target_sample_rate}Hz: {e_resample}. Proceeding with original sample rate {sample_rate}.",
                exc_info=True,
            )
    elif target_sample_rate is not None and target_sample_rate != sample_rate:
        logger.warning(
            f"Librosa not available. Cannot resample audio from {sample_rate}Hz to {target_sample_rate}Hz. "
            f"Proceeding with original sample rate for encoding."
        )

    start_time = time.time()
    output_buffer = io.BytesIO()

    try:
        audio_to_write = audio_array
        rate_to_write = sample_rate

        if output_format == "opus":
            OPUS_SUPPORTED_RATES = {8000, 12000, 16000, 24000, 48000}
            TARGET_OPUS_RATE = 48000  # Preferred Opus rate.

            if rate_to_write not in OPUS_SUPPORTED_RATES:
                if LIBROSA_AVAILABLE:
                    logger.warning(
                        f"Current sample rate {rate_to_write}Hz not directly supported by Opus. "
                        f"Resampling to {TARGET_OPUS_RATE}Hz using Librosa for Opus encoding."
                    )
                    audio_to_write = librosa.resample(
                        y=audio_array, orig_sr=rate_to_write, target_sr=TARGET_OPUS_RATE
                    )
                    rate_to_write = TARGET_OPUS_RATE
                else:
                    logger.error(
                        f"Librosa not available. Cannot resample audio from {rate_to_write}Hz for Opus encoding. "
                        f"Opus encoding may fail or produce poor quality."
                    )
                    # Proceed with current rate, soundfile might handle it or fail.
            sf.write(
                output_buffer,
                audio_to_write,
                rate_to_write,
                format="ogg",
                subtype="opus",
            )

        elif output_format == "wav":
            audio_to_write = _to_pcm16(audio_array)
            sf.write(
                output_buffer,
                audio_to_write,
                rate_to_write,
                format="wav",
                subtype="pcm_16",
            )

        elif output_format == "mp3":
            audio_int16 = _to_pcm16(audio_array)
            audio_segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1,
            )
            audio_segment.export(output_buffer, format="mp3")

        else:
            logger.error(
                f"Unsupported output format requested for encoding: {output_format}"
            )
            return None

        encoded_bytes = output_buffer.getvalue()
        end_time = time.time()
        logger.info(
            f"Encoded {len(encoded_bytes)} bytes to '{output_format}' at {rate_to_write}Hz in {end_time - start_time:.3f} seconds."
        )
        return encoded_bytes

    except ImportError as ie_sf:  # Specifically for soundfile import issues
        logger.critical(
            f"The 'soundfile' library or its dependency (libsndfile) is not installed or found. "
            f"Audio encoding/saving is not possible. Please install it. Error: {ie_sf}"
        )
        return None
    except Exception as e:
        logger.error(f"Error encoding audio to '{output_format}': {e}", exc_info=True)
        return None

# --- Performance Monitoring Utility ---
class PerformanceMonitor:
    """
    A simple helper class for recording and reporting elapsed time for different
    stages of an operation. Useful for debugging performance bottlenecks.
    """

    def __init__(
        self, enabled: bool = True, logger_instance: Optional[logging.Logger] = None
    ):
        self.enabled: bool = enabled
        self.logger = (
            logger_instance
            if logger_instance is not None
            else logging.getLogger(__name__)
        )
        self.start_time: float = 0.0
        self.events: List[Tuple[str, float]] = []
        if self.enabled:
            self.start_time = time.monotonic()
            self.events.append(("Monitoring Started", self.start_time))

    def record(self, event_name: str):
        if not self.enabled:
            return
        self.events.append((event_name, time.monotonic()))

    def report(self, log_level: int = logging.DEBUG) -> str:
        if not self.enabled or not self.events:
            return "Performance monitoring was disabled or no events recorded."

        report_lines = ["Performance Report:"]
        last_event_time = self.events[0][1]

        for i in range(1, len(self.events)):
            event_name, timestamp = self.events[i]
            prev_event_name, _ = self.events[i - 1]
            duration_since_last = timestamp - last_event_time
            duration_since_start = timestamp - self.start_time
            report_lines.append(
                f"  - Event: '{event_name}' (after '{prev_event_name}') "
                f"took {duration_since_last:.4f}s. Total elapsed: {duration_since_start:.4f}s"
            )
            last_event_time = timestamp

        total_duration = self.events[-1][1] - self.start_time
        report_lines.append(f"Total Monitored Duration: {total_duration:.4f}s")
        full_report_str = "\n".join(report_lines)

        if self.logger:
            self.logger.log(log_level, full_report_str)
        return full_report_str
