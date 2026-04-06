from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import EnvConfigManager


def test_text_profiles_json_env_merges_with_default_profiles(monkeypatch):
    monkeypatch.setenv("KITTEN_TTS_DEVICE", "cpu")
    monkeypatch.setenv(
        "KITTEN_TEXT_PROFILES_JSON",
        (
            '{"balanced":{"pause_strength":"strong","remove_urls":false},'
            '"custom":{"pause_strength":"light","dialogue_turn_splitting":true}}'
        ),
    )

    manager = EnvConfigManager()
    profiles = manager.get("text_processing.profiles", {})

    assert profiles["balanced"]["pause_strength"] == "strong"
    assert profiles["balanced"]["remove_urls"] is False
    assert profiles["balanced"]["filter_table_artifacts"] is True
    assert profiles["custom"]["pause_strength"] == "light"
    assert profiles["custom"]["dialogue_turn_splitting"] is True


def test_server_concurrency_env_values_are_coerced(monkeypatch):
    monkeypatch.setenv("KITTEN_TTS_DEVICE", "cpu")
    monkeypatch.setenv("KITTEN_SERVER_WORKERS", "3")
    monkeypatch.setenv("KITTEN_MAX_CONCURRENT_GENERATIONS", "2")
    monkeypatch.setenv("KITTEN_GENERATION_QUEUE_TIMEOUT_SECONDS", "15.5")

    manager = EnvConfigManager()

    assert manager.get_int("server.worker_processes") == 3
    assert manager.get_int("server.max_concurrent_generations") == 2
    assert manager.get_float("server.generation_queue_timeout_seconds") == 15.5
