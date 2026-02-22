# File: models.py
# Pydantic models for API request and response validation.

from typing import Optional, Literal
from pydantic import BaseModel, Field


class GenerationParams(BaseModel):
    """Common parameters for TTS generation."""

    speed: Optional[float] = Field(
        None,
        ge=0.25,
        le=4.0,
        description="Speed factor for the generated audio. 1.0 is normal speed.",
    )
    language: Optional[str] = Field(
        None,
        description="Language of the text. (Primarily for UI, actual engine may infer)",
    )


class TextOptions(BaseModel):
    """Advanced text preprocessing and chunking behavior controls."""

    profile: Optional[Literal["balanced", "narration", "dialogue"]] = Field(
        None,
        description="Named text-processing profile. Applies only to this request on /tts.",
    )
    remove_punctuation: Optional[bool] = Field(
        None,
        description="If true, strips punctuation characters during preprocessing.",
    )
    normalize_pause_punctuation: Optional[bool] = Field(
        None,
        description="If true, normalizes punctuation runs and symbols into pause-friendly forms.",
    )
    pause_strength: Optional[Literal["light", "medium", "strong"]] = Field(
        None,
        description="Controls normalization aggressiveness for pause-like punctuation.",
    )
    dialogue_turn_splitting: Optional[bool] = Field(
        None,
        description="If true, split script-style dialogue turns into independent chunk segments.",
    )
    speaker_label_mode: Optional[Literal["drop", "speak"]] = Field(
        None,
        description="When dialogue turns are detected, either remove or keep speaker labels.",
    )
    max_punct_run: Optional[int] = Field(
        None,
        ge=1,
        le=6,
        description="Maximum run length allowed for repeated pause punctuation.",
    )


class CustomTTSRequest(BaseModel):
    """Request model for the custom /tts endpoint."""

    text: str = Field(..., min_length=1, description="Text to be synthesized.")

    voice: str = Field(
        ...,
        description="Voice identifier (e.g., 'Jasper'). Available voices: Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo",
    )

    output_format: Optional[Literal["wav", "opus", "mp3", "aac"]] = Field(
        "wav", description="Desired audio output format."
    )

    split_text: Optional[bool] = Field(
        True,
        description="Whether to automatically split long text into chunks for processing.",
    )
    chunk_size: Optional[int] = Field(
        120,
        ge=50,
        le=500,
        description="Approximate target character length for text chunks when splitting is enabled (50-500).",
    )

    # Embed generation parameters directly
    speed: Optional[float] = Field(
        None, description="Overrides default speed if provided."
    )
    language: Optional[str] = Field(
        None, description="Overrides default language if provided."
    )
    text_options: Optional[TextOptions] = Field(
        None,
        description="Optional request-level overrides for text preprocessing and dialogue handling.",
    )


class ErrorResponse(BaseModel):
    """Standard error response model for API errors."""

    detail: str = Field(..., description="A human-readable explanation of the error.")


class UpdateStatusResponse(BaseModel):
    """Response model for status updates, e.g., after saving settings."""

    message: str = Field(
        ..., description="A message describing the result of the operation."
    )
    restart_needed: Optional[bool] = Field(
        False,
        description="Indicates if a server restart is recommended or required for changes to take full effect.",
    )
