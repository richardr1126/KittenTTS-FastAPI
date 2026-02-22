from pathlib import Path
import os
import sys
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


def test_openai_route_uses_shared_default_chunking(api_client: TestClient, monkeypatch):
    captured_kwargs = {}

    def fake_generate_audio_bytes(**kwargs):
        captured_kwargs.update(kwargs)
        return b"x" * 512

    monkeypatch.setattr(server, "_generate_audio_bytes", fake_generate_audio_bytes)

    payload = {
        "model": "tts-1",
        "input": "OpenAI compatible request text.",
        "voice": "Bella",
        "response_format": "wav",
        "speed": 1.0,
    }

    response = api_client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 200, response.text
    assert captured_kwargs["split_text"] is server._OPENAI_ROUTE_DEFAULT_SPLIT_TEXT
    assert captured_kwargs["chunk_size"] == server._OPENAI_ROUTE_DEFAULT_CHUNK_SIZE


def test_openai_route_returns_400_for_unspeakable_cleaned_text(api_client: TestClient):
    payload = {
        "model": "tts-1",
        "input": "| --- | --- |\n| 10 | 20 |\n[12]\n____ ||||",
        "voice": "Bella",
        "response_format": "wav",
        "speed": 1.0,
    }

    response = api_client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 400
    assert "no speakable content" in response.json()["detail"].lower()


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
def test_real_audio_generation_via_openai_route(
    integration_client: TestClient, response_format: str
):
    payload = {
        "model": "tts-1",
        "input": (
            "This is a real synthesis test. "
            "| Col A | Col B |\n| --- | --- |\n| 1 | 2 |\n"
            "The sentence after the table should still be speakable."
        ),
        "voice": "Bella",
        "response_format": response_format,
        "speed": 1.0,
    }

    response = integration_client.post("/v1/audio/speech", json=payload)

    expected_media_type = (
        "audio/aac" if response_format == "aac" else f"audio/{response_format}"
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith(expected_media_type)
    if response_format == "wav":
        assert response.content[:4] == b"RIFF"
    if response_format == "opus":
        assert response.content[:4] == b"OggS"
    if response_format == "aac":
        assert response.content[0] == 0xFF
        assert (response.content[1] & 0xF0) == 0xF0
    assert len(response.content) > 100
