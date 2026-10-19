from pathlib import Path
import asyncio
import sys
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import server


class _FakeRequest:
    """Stands in for a Starlette request whose body has been read."""

    def __init__(self) -> None:
        self.disconnected = asyncio.Event()

    async def receive(self) -> dict:
        await self.disconnected.wait()
        return {"type": "http.disconnect"}


def _generation_kwargs() -> dict:
    return {
        "raw_text": "Hello there.",
        "voice": "Jasper",
        "speed": 1.0,
        "output_format": "wav",
        "target_sample_rate": 24000,
        "split_text": False,
        "chunk_size": 120,
        "context_label": "test",
    }


@pytest.fixture
def generation_calls(monkeypatch) -> list:
    calls = []

    def fake_generate(**kwargs) -> bytes:
        calls.append(kwargs)
        return b"x" * 512

    monkeypatch.setattr(server, "_generate_audio_bytes", fake_generate)
    monkeypatch.setattr(server, "_generation_queue_timeout_seconds", 0.0)
    return calls


@pytest.mark.asyncio
async def test_queued_request_whose_client_left_never_synthesizes(
    monkeypatch, generation_calls
):
    semaphore = asyncio.Semaphore(1)
    monkeypatch.setattr(server, "_generation_semaphore", semaphore)
    await semaphore.acquire()  # another request is generating
    request = _FakeRequest()

    queued = asyncio.ensure_future(
        server._generate_audio_bytes_with_limits(request, **_generation_kwargs())
    )
    await asyncio.sleep(0)
    request.disconnected.set()

    with pytest.raises(server.ClientDisconnectedError):
        await queued
    assert generation_calls == []
    # The abandoned waiter took no slot: freeing the busy one makes it available.
    semaphore.release()
    assert not semaphore.locked()


@pytest.mark.asyncio
async def test_slot_won_after_client_left_is_handed_back(monkeypatch, generation_calls):
    semaphore = asyncio.Semaphore(1)
    monkeypatch.setattr(server, "_generation_semaphore", semaphore)
    request = _FakeRequest()
    request.disconnected.set()

    with pytest.raises(server.ClientDisconnectedError):
        await server._generate_audio_bytes_with_limits(request, **_generation_kwargs())
    assert generation_calls == []
    assert not semaphore.locked()


@pytest.mark.asyncio
async def test_connected_request_generates_and_releases_its_slot(
    monkeypatch, generation_calls
):
    semaphore = asyncio.Semaphore(1)
    monkeypatch.setattr(server, "_generation_semaphore", semaphore)

    audio = await server._generate_audio_bytes_with_limits(
        _FakeRequest(), **_generation_kwargs()
    )

    assert audio == b"x" * 512
    assert len(generation_calls) == 1
    assert not semaphore.locked()


def test_remaining_chunks_stop_once_client_leaves(monkeypatch):
    cancel_event = threading.Event()
    synthesized = []

    def fake_synthesize(text: str, voice: str, speed: float = 1.0, clean_text: bool = True):
        synthesized.append(text)
        cancel_event.set()  # the client leaves while the first chunk runs
        return np.ones(2400, dtype=np.float32), 24000

    monkeypatch.setattr(server.engine, "synthesize", fake_synthesize)

    with pytest.raises(server.ClientDisconnectedError):
        server._synthesize_chunks_and_merge(
            text_chunks=["First chunk.", "Second chunk.", "Third chunk."],
            voice="Jasper",
            speed=1.0,
            cancel_event=cancel_event,
        )
    assert synthesized == ["First chunk."]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/tts", {"text": "Hello there.", "voice": "Jasper"}),
        ("/v1/audio/speech", {"model": "tts-1", "input": "Hello there.", "voice": "Jasper"}),
    ],
)
def test_routes_answer_a_departed_client_with_499(monkeypatch, path, body):
    monkeypatch.setattr(server.engine, "load_model", lambda: True)
    monkeypatch.setattr(server.engine, "MODEL_LOADED", True)

    async def departed(*_args, **_kwargs):
        raise server.ClientDisconnectedError()

    monkeypatch.setattr(server, "_generate_audio_bytes_with_limits", departed)
    with TestClient(server.app) as client:
        response = client.post(path, json=body)

    assert response.status_code == 499
