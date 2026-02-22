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
