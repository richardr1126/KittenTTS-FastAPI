# File: server.py
# Main FastAPI application for the TTS Server.
# Handles API requests for text-to-speech generation, UI serving,
# configuration management, and file uploads.

import io
import logging
import time
import yaml  # For loading presets
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Literal

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    BackgroundTasks,
)
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    FileResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# --- Internal Project Imports ---
from config import (
    config_manager,
    get_host,
    get_port,
    get_ui_title,
    get_gen_default_speed,
    get_audio_sample_rate,
    get_full_config_for_template,
    get_audio_output_format,
    PROJECT_ROOT,
)

import engine  # TTS Engine interface
import nlp
from models import (  # Pydantic models
    CustomTTSRequest,
    ErrorResponse,
    UpdateStatusResponse,
)
import utils  # Utility functions

from pydantic import BaseModel, Field


class OpenAISpeechRequest(BaseModel):
    model: str
    input_: str = Field(..., alias="input")
    voice: str
    response_format: Literal["wav", "opus", "mp3", "aac"] = "wav"
    speed: float = 1.0
    seed: Optional[int] = None


_OPENAI_ROUTE_DEFAULT_SPLIT_TEXT = bool(
    CustomTTSRequest.model_fields["split_text"].default
)
_OPENAI_ROUTE_DEFAULT_CHUNK_SIZE = int(
    CustomTTSRequest.model_fields["chunk_size"].default
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ],
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Static Files and HTML Templates Configuration ---
ui_static_path = Path(__file__).parent / "ui"


def _log_access_urls(host: str, port: int):
    """Logs a readable startup summary with the UI and docs URLs."""
    display_host = "localhost" if host == "0.0.0.0" else host
    base_url = f"http://{display_host}:{port}"
    logger.info("")
    logger.info("========================================")
    logger.info("  Kitten TTS Server is ready")
    logger.info("  Visit UI:   %s/", base_url)
    logger.info("  API Docs:   %s/docs", base_url)
    if display_host != host:
        logger.info("  Listening:  http://%s:%s", host, port)
    logger.info("========================================")


def _preprocess_input_text(
    raw_text: str,
    *,
    context_label: str,
    perf_monitor: Optional[utils.PerformanceMonitor] = None,
) -> str:
    cleaned_text, preprocess_meta = engine.preprocess_text(raw_text)
    logger.info(
        "%s preprocessing complete: raw_len=%d, cleaned_len=%d, table_lines_removed=%d, reference_fragments_removed=%d, symbol_noise_collapsed=%d",
        context_label,
        len(raw_text),
        preprocess_meta.get("output_length", 0),
        preprocess_meta.get("table_lines_removed", 0),
        preprocess_meta.get("reference_fragments_removed", 0),
        preprocess_meta.get("symbol_noise_collapsed", 0),
    )
    if perf_monitor is not None:
        perf_monitor.record("Input text preprocessed")

    if not engine.is_speakable_text(cleaned_text):
        raise HTTPException(
            status_code=400,
            detail="Input text contained no speakable content after cleanup. Remove table/reference artifacts and retry.",
        )
    return cleaned_text


def _resolve_text_chunks(
    raw_text: str,
    *,
    split_text: bool,
    chunk_size: int,
    context_label: str,
    precleaned_text: Optional[str] = None,
    perf_monitor: Optional[utils.PerformanceMonitor] = None,
) -> List[str]:
    if split_text and len(raw_text) > (chunk_size * 1.5):
        logger.info(f"Splitting text into chunks of size ~{chunk_size}.")
        text_chunks = nlp.chunk_text_by_sentences(raw_text, chunk_size)
        if perf_monitor is not None:
            perf_monitor.record(f"Text split into {len(text_chunks)} chunks")
    else:
        text_chunks = [precleaned_text if precleaned_text is not None else raw_text]
        logger.info(
            "Processing text as a single chunk (splitting not enabled or text too short)."
        )

    text_chunks = [chunk for chunk in text_chunks if chunk and chunk.strip()]
    if not text_chunks:
        raise HTTPException(
            status_code=400,
            detail="Input text contained no usable speakable chunks after cleanup.",
        )

    if split_text:
        cleaned_chunks = []
        for idx, chunk in enumerate(text_chunks, start=1):
            cleaned_chunk, _ = engine.preprocess_text(chunk)
            if engine.is_speakable_text(cleaned_chunk):
                cleaned_chunks.append(cleaned_chunk)
            else:
                logger.info(
                    "%s dropped unspeakable chunk %d/%d after cleanup.",
                    context_label,
                    idx,
                    len(text_chunks),
                )
        if perf_monitor is not None:
            perf_monitor.record(
                f"Chunk-level cleanup kept {len(cleaned_chunks)}/{len(text_chunks)} chunks"
            )
        text_chunks = cleaned_chunks

    if not text_chunks:
        raise HTTPException(
            status_code=400,
            detail="Input text contained no usable speakable chunks after cleanup.",
        )

    return text_chunks


def _synthesize_chunks_and_merge(
    *,
    text_chunks: List[str],
    voice: str,
    speed: float,
    perf_monitor: Optional[utils.PerformanceMonitor] = None,
) -> tuple[np.ndarray, int]:
    all_audio_segments_np: List[np.ndarray] = []
    engine_output_sample_rate: Optional[int] = None

    for i, chunk in enumerate(text_chunks):
        logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}...")
        try:
            chunk_audio_np, chunk_sr_from_engine = engine.synthesize(
                text=chunk,
                voice=voice,
                speed=speed,
                clean_text=False,
            )
            if perf_monitor is not None:
                perf_monitor.record(f"Engine synthesized chunk {i+1}")

            if chunk_audio_np is None or chunk_sr_from_engine is None:
                error_detail = (
                    f"TTS engine failed to synthesize audio for chunk {i+1}. "
                    "The cleaned text may still be unspeakable."
                )
                logger.error(error_detail)
                raise HTTPException(status_code=500, detail=error_detail)

            if engine_output_sample_rate is None:
                engine_output_sample_rate = chunk_sr_from_engine
            elif engine_output_sample_rate != chunk_sr_from_engine:
                logger.warning(
                    f"Inconsistent sample rate from engine: chunk {i+1} ({chunk_sr_from_engine}Hz) "
                    f"differs from previous ({engine_output_sample_rate}Hz). Using first chunk's SR."
                )

            all_audio_segments_np.append(chunk_audio_np)

        except HTTPException:
            raise
        except Exception as e_chunk:
            error_detail = f"Error processing audio chunk {i+1}: {str(e_chunk)}"
            logger.error(error_detail, exc_info=True)
            raise HTTPException(status_code=500, detail=error_detail)

    if not all_audio_segments_np:
        logger.error("No audio segments were successfully generated.")
        raise HTTPException(
            status_code=500, detail="Audio generation resulted in no output."
        )

    if engine_output_sample_rate is None:
        logger.error("Engine output sample rate could not be determined.")
        raise HTTPException(
            status_code=500, detail="Failed to determine engine sample rate."
        )

    try:
        if len(all_audio_segments_np) > 1:
            silence_duration_ms = 200
            silence_samples = int(
                silence_duration_ms / 1000 * engine_output_sample_rate
            )
            silence_array = np.zeros(silence_samples, dtype=np.float32)
            crossfade_samples = int(0.01 * engine_output_sample_rate)

            merged_audio = []
            for i, chunk in enumerate(all_audio_segments_np):
                if i == 0:
                    merged_audio.append(chunk)
                else:
                    merged_audio.append(silence_array)
                    if (
                        len(merged_audio[-2]) >= crossfade_samples
                        and len(chunk) >= crossfade_samples
                    ):
                        fade_out = np.linspace(1, 0, crossfade_samples)
                        merged_audio[-2][-crossfade_samples:] *= fade_out

                        fade_in = np.linspace(0, 1, crossfade_samples)
                        chunk_copy = chunk.copy()
                        chunk_copy[:crossfade_samples] *= fade_in
                        merged_audio.append(chunk_copy)
                    else:
                        merged_audio.append(chunk)

            final_audio_np = np.concatenate(merged_audio)
            logger.debug(
                f"Added {silence_duration_ms}ms silence between {len(all_audio_segments_np)} chunks"
            )
        else:
            final_audio_np = all_audio_segments_np[0]

        if perf_monitor is not None:
            perf_monitor.record("All audio chunks processed and concatenated")

    except ValueError as e_concat:
        logger.error(f"Audio concatenation failed: {e_concat}", exc_info=True)
        for idx, seg in enumerate(all_audio_segments_np):
            logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
        raise HTTPException(
            status_code=500, detail=f"Audio concatenation error: {e_concat}"
        )

    return final_audio_np, engine_output_sample_rate


def _generate_audio_bytes(
    *,
    raw_text: str,
    voice: str,
    speed: float,
    output_format: str,
    target_sample_rate: int,
    split_text: bool,
    chunk_size: int,
    context_label: str,
    perf_monitor: Optional[utils.PerformanceMonitor] = None,
) -> bytes:
    cleaned_text = _preprocess_input_text(
        raw_text, context_label=context_label, perf_monitor=perf_monitor
    )
    text_chunks = _resolve_text_chunks(
        raw_text,
        split_text=split_text,
        chunk_size=chunk_size,
        context_label=context_label,
        precleaned_text=cleaned_text,
        perf_monitor=perf_monitor,
    )
    logger.info("%s generation will synthesize %d chunk(s).", context_label, len(text_chunks))

    final_audio_np, engine_output_sample_rate = _synthesize_chunks_and_merge(
        text_chunks=text_chunks,
        voice=voice,
        speed=speed,
        perf_monitor=perf_monitor,
    )

    encoded_audio_bytes = utils.encode_audio(
        audio_array=final_audio_np,
        sample_rate=engine_output_sample_rate,
        output_format=output_format,
        target_sample_rate=target_sample_rate,
    )
    if perf_monitor is not None:
        perf_monitor.record(
            f"Final audio encoded to {output_format} (target SR: {target_sample_rate}Hz from engine SR: {engine_output_sample_rate}Hz)"
        )

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(
            f"Failed to encode final audio to format: {output_format} or output is too small ({len(encoded_audio_bytes or b'')} bytes)."
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode audio to {output_format} or generated invalid audio.",
        )

    return encoded_audio_bytes


def _media_type_for_format(audio_format: str) -> str:
    if audio_format == "aac":
        return "audio/aac"
    return f"audio/{audio_format}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logger.info("TTS Server: Initializing application...")
    try:
        logger.info("Configuration loaded successfully.")

        paths_to_ensure = [
            ui_static_path,
            config_manager.get_path(
                "paths.model_cache", str(PROJECT_ROOT / "model_cache"), ensure_absolute=True
            ),
        ]
        for p in paths_to_ensure:
            p.mkdir(parents=True, exist_ok=True)

        if not engine.load_model():
            logger.critical(
                "CRITICAL: TTS Model failed to load on startup. Server might not function correctly."
            )
        else:
            logger.info("TTS Model loaded successfully via engine.")

        _log_access_urls(get_host(), get_port())

        logger.info("Application startup sequence complete.")
        yield
    except Exception as e_startup:
        logger.error(
            f"FATAL ERROR during application startup: {e_startup}", exc_info=True
        )
        yield
    finally:
        logger.info("TTS Server: Application shutdown sequence initiated...")
        logger.info("TTS Server: Application shutdown complete.")


# --- FastAPI Application Instance ---
app = FastAPI(
    title=get_ui_title(),
    description="Text-to-Speech server with advanced UI and API capabilities.",
    version="2.0.2",  # Version Bump
    lifespan=lifespan,
)

# --- Static Files mounting ---
if ui_static_path.is_dir():
    app.mount("/ui", StaticFiles(directory=ui_static_path), name="ui_static_assets")
else:
    logger.warning(
        f"UI static assets directory not found at '{ui_static_path}'. UI may not load correctly."
    )

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Initialized above lifespan ---

# This will serve files from 'ui_static_path/vendor' when requests come to '/vendor/*'
if (ui_static_path / "vendor").is_dir():
    app.mount(
        "/vendor", StaticFiles(directory=ui_static_path / "vendor"), name="vendor_files"
    )
else:
    logger.warning(
        f"Vendor directory not found at '{ui_static_path}' /vendor. Wavesurfer might not load."
    )


@app.get("/styles.css", include_in_schema=False)
async def get_main_styles():
    styles_file = ui_static_path / "styles.css"
    if styles_file.is_file():
        return FileResponse(styles_file)
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/script.js", include_in_schema=False)
async def get_main_script():
    script_file = ui_static_path / "script.js"
    if script_file.is_file():
        return FileResponse(script_file)
    raise HTTPException(status_code=404, detail="script.js not found")


templates = Jinja2Templates(directory=str(ui_static_path))

# --- API Endpoints ---


# --- Main UI Route ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request):
    """Serves the main web interface (index.html)."""
    logger.info("Request received for main UI page ('/').")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e_render:
        logger.error(f"Error rendering main UI page: {e_render}", exc_info=True)
        return HTMLResponse(
            "<html><body><h1>Internal Server Error</h1><p>Could not load the TTS interface. "
            "Please check server logs for more details.</p></body></html>",
            status_code=500,
        )


# --- API Endpoint for Initial UI Data ---
@app.get("/api/ui/initial-data", tags=["UI Helpers"])
async def get_ui_initial_data():
    """
    Provides all necessary initial data for the UI to render,
    including configuration, file lists, and presets.
    """
    logger.info("Request received for /api/ui/initial-data.")
    try:
        full_config = get_full_config_for_template()
        loaded_presets = []
        presets_file = ui_static_path / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content
                else:
                    logger.warning(
                        f"Invalid format in {presets_file}. Expected a list, got {type(yaml_content)}."
                    )
        else:
            logger.info(
                f"Presets file not found: {presets_file}. No presets will be loaded for initial data."
            )

        initial_gen_result_placeholder = {
            "outputUrl": None,
            "filename": None,
            "genTime": None,
            "submittedVoice": None,
        }

        return {
            "config": full_config,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to load initial data for UI."
        )


@app.post(
    "/restart_server", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def restart_server_endpoint():
    """Attempts to trigger a server restart."""
    logger.info("Request received for /restart_server.")
    message = (
        "Server restart initiated. If running locally without a process manager, "
        "you may need to restart manually. For managed environments (Docker, systemd), "
        "the manager should handle the restart."
    )
    logger.warning(message)
    return UpdateStatusResponse(message=message, restart_needed=True)


# --- TTS Generation Endpoint ---


@app.post(
    "/tts",
    tags=["TTS Generation"],
    summary="Generate speech with custom parameters",
    responses={
        200: {
            "content": {
                "audio/wav": {},
                "audio/opus": {},
                "audio/mp3": {},
                "audio/aac": {},
            },
            "description": "Successful audio generation.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or input.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during generation.",
        },
        503: {
            "model": ErrorResponse,
            "description": "TTS engine not available or model not loaded.",
        },
    },
)
async def custom_tts_endpoint(
    request: CustomTTSRequest, background_tasks: BackgroundTasks
):
    """
    Generates speech audio from text using specified parameters.
    Returns audio as a stream (WAV or Opus).
    """
    perf_monitor = utils.PerformanceMonitor(
        enabled=config_manager.get_bool("server.enable_performance_monitor", False)
    )
    perf_monitor.record("TTS request received")

    if not engine.MODEL_LOADED:
        logger.error("TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.info(
        f"Received /tts request: voice='{request.voice}', format='{request.output_format}'"
    )
    logger.debug(
        f"TTS params: speed={request.speed}, split={request.split_text}, chunk_size={request.chunk_size}"
    )
    logger.debug(f"Input text (first 100 chars): '{request.text[:100]}...'")
    perf_monitor.record("Parameters resolved")

    resolved_speed = (
        request.speed if request.speed is not None else get_gen_default_speed()
    )
    output_format_str = (
        request.output_format if request.output_format else get_audio_output_format()
    )
    split_text_to_use = (
        request.split_text if request.split_text is not None else True
    )
    chunk_size_to_use = request.chunk_size if request.chunk_size is not None else 120

    encoded_audio_bytes = _generate_audio_bytes(
        raw_text=request.text,
        voice=request.voice,
        speed=resolved_speed,
        output_format=output_format_str,
        target_sample_rate=get_audio_sample_rate(),
        split_text=split_text_to_use,
        chunk_size=chunk_size_to_use,
        context_label="/tts",
        perf_monitor=perf_monitor,
    )

    media_type = _media_type_for_format(output_format_str)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    suggested_filename_base = f"tts_output_{timestamp_str}"
    download_filename = utils.sanitize_filename(
        f"{suggested_filename_base}.{output_format_str}"
    )
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

    logger.info(
        f"Successfully generated audio: {download_filename}, {len(encoded_audio_bytes)} bytes, type {media_type}."
    )
    logger.debug(perf_monitor.report())

    return StreamingResponse(
        io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers
    )


@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(request: OpenAISpeechRequest):
    perf_monitor = utils.PerformanceMonitor(
        enabled=config_manager.get_bool("server.enable_performance_monitor", False)
    )
    perf_monitor.record("OpenAI speech request received")

    # Check if the TTS model is loaded
    if not engine.MODEL_LOADED:
        logger.error("OpenAI speech request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.info(
        "Received /v1/audio/speech request: voice='%s', format='%s'",
        request.voice,
        request.response_format,
    )
    logger.debug(
        "OpenAI speech params: speed=%s, split=%s, chunk_size=%s",
        request.speed,
        _OPENAI_ROUTE_DEFAULT_SPLIT_TEXT,
        _OPENAI_ROUTE_DEFAULT_CHUNK_SIZE,
    )
    logger.debug("Input text (first 100 chars): '%s...'", request.input_[:100])
    perf_monitor.record("Parameters resolved")

    try:
        encoded_audio = _generate_audio_bytes(
            raw_text=request.input_,
            voice=request.voice,
            speed=request.speed,
            output_format=request.response_format,
            target_sample_rate=get_audio_sample_rate(),
            split_text=_OPENAI_ROUTE_DEFAULT_SPLIT_TEXT,
            chunk_size=_OPENAI_ROUTE_DEFAULT_CHUNK_SIZE,
            context_label="/v1/audio/speech",
            perf_monitor=perf_monitor,
        )
        media_type = _media_type_for_format(request.response_format)
        logger.info(
            "Successfully generated OpenAI-compatible audio: %d bytes, type %s.",
            len(encoded_audio),
            media_type,
        )
        logger.debug(perf_monitor.report())
        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in openai_speech_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audio/voices", tags=["OpenAI Compatible"])
async def openai_voices_endpoint():
    """
    Returns a list of available voices. 
    Compatible with some OpenAI-like integrations (e.g. OpenReader).
    """
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )
    
    return {"voices": engine.KITTEN_TTS_VOICES}



# --- Main Execution ---
if __name__ == "__main__":
    server_host = get_host()
    server_port = get_port()

    logger.info(f"Starting TTS Server on http://{server_host}:{server_port}")
    logger.info("Startup in progress. URL summary will be logged when ready.")

    import uvicorn

    uvicorn.run(
        "server:app",
        host=server_host,
        port=server_port,
        log_level="info",
        workers=1,
        reload=False,
    )
