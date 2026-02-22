from pathlib import Path
import os
import sys
from contextlib import asynccontextmanager
from typing import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import engine
import server


def _is_this_file_explicitly_targeted() -> bool:
    this_file = Path(__file__).name
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        target = arg.split("::", 1)[0]
        if Path(target).name == this_file:
            return True
    return False


RUN_AUDIO_INTEGRATION = (
    os.getenv("INTEGRATION_TESTS", "0") == "1"
    or _is_this_file_explicitly_targeted()
)


@asynccontextmanager
async def _openai_async_client():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as http_client:
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="http://testserver/v1",
            http_client=http_client,
        )
        yield client, openai


@pytest.fixture
def api_client(monkeypatch) -> Generator[TestClient, None, None]:
    # Avoid real model loading for non-integration API behavior tests.
    monkeypatch.setattr(server.engine, "load_model", lambda: True)
    monkeypatch.setattr(server.engine, "MODEL_LOADED", True)
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def integration_client() -> Generator[TestClient, None, None]:
    assert engine.load_model(), "Model failed to load for integration test."
    with TestClient(server.app) as test_client:
        yield test_client


def test_tts_mixed_prose_and_table_uses_cleaned_chunks(api_client: TestClient, monkeypatch):
    captured_calls = []

    def fake_synthesize(
        text: str, voice: str, speed: float = 1.0, clean_text: bool = True
    ):
        captured_calls.append((text, clean_text))
        return np.ones(2400, dtype=np.float32), 24000

    monkeypatch.setattr(server.engine, "synthesize", fake_synthesize)
    monkeypatch.setattr(server.utils, "encode_audio", lambda **_: b"x" * 512)

    payload = {
        "text": (
            "Introduction to results.\n"
            "| Metric | Value |\n"
            "| --- | --- |\n"
            "| BLEU | 33.2 |\n"
            "Conclusion sentence."
        ),
        "voice": "Bella",
        "output_format": "wav",
        "split_text": True,
        "chunk_size": 60,
        "speed": 1.0,
    }

    response = api_client.post("/tts", json=payload)

    assert response.status_code == 200, response.text
    assert captured_calls
    assert all(clean_flag is False for _, clean_flag in captured_calls)
    assert all("|" not in chunk_text for chunk_text, _ in captured_calls)
    assert any(
        "introduction to results" in chunk_text for chunk_text, _ in captured_calls
    )
    assert any("conclusion sentence" in chunk_text for chunk_text, _ in captured_calls)


def test_tts_returns_400_when_cleaned_text_is_unspeakable(api_client: TestClient):
    payload = {
        "text": "| --- | --- |\n| 12 | 34 |\n[12][3,4]\n____ ||||",
        "voice": "Bella",
        "output_format": "wav",
        "split_text": True,
        "chunk_size": 80,
        "speed": 1.0,
    }

    response = api_client.post("/tts", json=payload)

    assert response.status_code == 400
    assert "no speakable content" in response.json()["detail"].lower()


def test_tts_reference_heavy_text_preserves_prose(api_client: TestClient, monkeypatch):
    captured_calls = []

    def fake_synthesize(
        text: str, voice: str, speed: float = 1.0, clean_text: bool = True
    ):
        captured_calls.append((text, clean_text))
        return np.ones(1200, dtype=np.float32), 24000

    monkeypatch.setattr(server.engine, "synthesize", fake_synthesize)
    monkeypatch.setattr(server.utils, "encode_audio", lambda **_: b"x" * 512)

    payload = {
        "text": (
            "The transformer model improves performance [12] and [3, 8]. "
            "Figure 1: Architecture overview. "
            "Table 2: Ablation summary."
        ),
        "voice": "Bella",
        "output_format": "wav",
        "split_text": False,
        "chunk_size": 120,
        "speed": 1.0,
    }

    response = api_client.post("/tts", json=payload)
    combined_text = " ".join(chunk for chunk, _ in captured_calls)

    assert response.status_code == 200, response.text
    assert captured_calls
    assert all(clean_flag is False for _, clean_flag in captured_calls)
    assert "transformer model improves performance" in combined_text
    assert "[" not in combined_text and "]" not in combined_text


def test_tts_split_text_preserves_sentence_boundaries_for_chunking(
    api_client: TestClient, monkeypatch
):
    captured_calls = []

    def fake_synthesize(
        text: str, voice: str, speed: float = 1.0, clean_text: bool = True
    ):
        captured_calls.append((text, clean_text))
        return np.ones(1200, dtype=np.float32), 24000

    monkeypatch.setattr(server.engine, "synthesize", fake_synthesize)
    monkeypatch.setattr(server.utils, "encode_audio", lambda **_: b"x" * 512)

    payload = {
        "text": (
            "First sentence is short. "
            "Second sentence is also short. "
            "Third sentence remains short. "
            "Fourth sentence stays short."
        ),
        "voice": "Bella",
        "output_format": "wav",
        "split_text": True,
        "chunk_size": 50,
        "speed": 1.0,
    }

    response = api_client.post("/tts", json=payload)

    assert response.status_code == 200, response.text
    assert len(captured_calls) >= 2
    assert all(clean_flag is False for _, clean_flag in captured_calls)


def test_tts_accepts_non_latin_text_when_speakable(api_client: TestClient, monkeypatch):
    captured_calls = []

    def fake_synthesize(
        text: str, voice: str, speed: float = 1.0, clean_text: bool = True
    ):
        captured_calls.append((text, clean_text))
        return np.ones(1200, dtype=np.float32), 24000

    monkeypatch.setattr(server.engine, "synthesize", fake_synthesize)
    monkeypatch.setattr(server.utils, "encode_audio", lambda **_: b"x" * 512)

    payload = {
        "text": "こんにちは世界。これはテストです。",
        "voice": "Bella",
        "output_format": "wav",
        "split_text": False,
        "chunk_size": 120,
        "speed": 1.0,
    }

    response = api_client.post("/tts", json=payload)

    assert response.status_code == 200, response.text
    assert captured_calls
    assert all(clean_flag is False for _, clean_flag in captured_calls)


@pytest.mark.asyncio
async def test_openai_route_accepts_kittentts_alias_and_uses_shared_default_chunking(
    monkeypatch,
):
    captured_kwargs = {}

    def fake_generate_audio_bytes(**kwargs):
        captured_kwargs.update(kwargs)
        return b"x" * 512

    monkeypatch.setattr(server.engine, "load_model", lambda: True)
    monkeypatch.setattr(server.engine, "MODEL_LOADED", True)
    monkeypatch.setattr(server, "_generate_audio_bytes", fake_generate_audio_bytes)

    async with _openai_async_client() as (client, _):
        speech = await client.audio.speech.create(
            model="KittenTTS",
            input="OpenAI compatible request text.",
            voice="Bella",
            response_format="wav",
            speed=1.0,
        )

    assert speech.response.status_code == 200
    assert len(speech.content) > 100
    assert captured_kwargs["split_text"] is server._OPENAI_ROUTE_DEFAULT_SPLIT_TEXT
    assert captured_kwargs["chunk_size"] == server._OPENAI_ROUTE_DEFAULT_CHUNK_SIZE


@pytest.mark.asyncio
async def test_openai_route_returns_400_for_unspeakable_cleaned_text(monkeypatch):
    monkeypatch.setattr(server.engine, "load_model", lambda: True)
    monkeypatch.setattr(server.engine, "MODEL_LOADED", True)

    async with _openai_async_client() as (client, openai):
        with pytest.raises(openai.BadRequestError) as exc_info:
            await client.audio.speech.create(
                model="tts-1",
                input="| --- | --- |\n| 10 | 20 |\n[12]\n____ ||||",
                voice="Bella",
                response_format="wav",
                speed=1.0,
            )

    assert exc_info.value.status_code == 400
    assert "no speakable content" in exc_info.value.body["detail"].lower()


@pytest.mark.asyncio
async def test_openai_route_returns_400_for_unsupported_model(monkeypatch):
    monkeypatch.setattr(server.engine, "load_model", lambda: True)
    monkeypatch.setattr(server.engine, "MODEL_LOADED", True)

    async with _openai_async_client() as (client, openai):
        with pytest.raises(openai.BadRequestError) as exc_info:
            await client.audio.speech.create(
                model="not-a-real-model",
                input="Hello world",
                voice="Bella",
                response_format="wav",
                speed=1.0,
            )

    assert exc_info.value.status_code == 400
    assert "unsupported model" in exc_info.value.body["detail"].lower()


def test_openai_models_endpoint_lists_tts_1(api_client: TestClient):
    response = api_client.get("/v1/models")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)

    tts_1 = next((model for model in payload["data"] if model["id"] == "tts-1"), None)
    assert tts_1 is not None
    assert tts_1["object"] == "model"
    assert isinstance(tts_1["created"], int)
    assert tts_1["owned_by"] == "kittentts-fastapi"


def test_openai_model_retrieve_endpoint(api_client: TestClient):
    response = api_client.get("/v1/models/tts-1")
    assert response.status_code == 200, response.text
    assert response.json()["id"] == "tts-1"

    missing_response = api_client.get("/v1/models/not-a-model")
    assert missing_response.status_code == 404


@pytest.mark.asyncio
async def test_openai_models_list_via_openai_python_client(monkeypatch):
    monkeypatch.setattr(server.engine, "load_model", lambda: True)
    monkeypatch.setattr(server.engine, "MODEL_LOADED", True)

    async with _openai_async_client() as (client, _):
        models = await client.models.list()

    assert any(model.id == "tts-1" for model in models.data)


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_AUDIO_INTEGRATION,
    reason=(
        "Set INTEGRATION_TESTS=1, or explicitly target this file, "
        "to run real audio generation integration tests."
    ),
)
@pytest.mark.parametrize("output_format", ["wav", "mp3", "opus", "aac"])
def test_real_audio_generation_via_tts_route(integration_client: TestClient, output_format: str):
    payload = {
        "text": (
            "This is a real synthesis test. "
            "| Col A | Col B |\n| --- | --- |\n| 1 | 2 |\n"
            "The sentence after the table should still be speakable."
        ),
        "voice": "Bella",
        "output_format": output_format,
        "speed": 1.0,
        "split_text": True,
        "chunk_size": 120,
    }

    response = integration_client.post("/tts", json=payload)

    expected_media_type = "audio/aac" if output_format == "aac" else f"audio/{output_format}"
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith(expected_media_type)
    if output_format == "wav":
        assert response.content[:4] == b"RIFF"
    if output_format == "opus":
        assert response.content[:4] == b"OggS"
    if output_format == "aac":
        assert response.content[0] == 0xFF
        assert (response.content[1] & 0xF0) == 0xF0
    assert len(response.content) > 100


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_AUDIO_INTEGRATION,
    reason=(
        "Set INTEGRATION_TESTS=1, or explicitly target this file, "
        "to run real audio generation integration tests."
    ),
)
@pytest.mark.parametrize("response_format", ["wav", "mp3", "opus", "aac"])
@pytest.mark.asyncio
async def test_real_audio_generation_via_openai_route(
    integration_client: TestClient, response_format: str
):
    async with _openai_async_client() as (client, _):
        speech = await client.audio.speech.create(
            model="tts-1",
            input=(
                "This is a real synthesis test. "
                "| Col A | Col B |\n| --- | --- |\n| 1 | 2 |\n"
                "The sentence after the table should still be speakable."
            ),
            voice="Bella",
            response_format=response_format,
            speed=1.0,
        )

    expected_media_type = (
        "audio/aac" if response_format == "aac" else f"audio/{response_format}"
    )
    assert speech.response.status_code == 200
    assert speech.response.headers["content-type"].startswith(expected_media_type)
    if response_format == "wav":
        assert speech.content[:4] == b"RIFF"
    if response_format == "opus":
        assert speech.content[:4] == b"OggS"
    if response_format == "aac":
        assert speech.content[0] == 0xFF
        assert (speech.content[1] & 0xF0) == 0xF0
    assert len(speech.content) > 100
