from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import engine


def _mock_text_processing_config(monkeypatch, text_processing_config):
    def _fake_get(key_path, default=None):
        if key_path == "text_processing":
            return text_processing_config
        return default

    monkeypatch.setattr(engine.config_manager, "get", _fake_get)


def test_resolve_text_options_uses_requested_profile_and_request_overrides(monkeypatch):
    _mock_text_processing_config(
        monkeypatch,
        {
            "active_profile": "balanced",
            "profiles": {
                "balanced": {
                    "filter_table_artifacts": True,
                    "filter_reference_artifacts": True,
                    "filter_symbol_noise": True,
                    "remove_urls": False,
                    "pause_strength": "medium",
                    "dialogue_turn_splitting": False,
                },
                "dialogue": {
                    "filter_table_artifacts": False,
                    "filter_reference_artifacts": True,
                    "filter_symbol_noise": False,
                    "remove_urls": False,
                    "pause_strength": "light",
                    "dialogue_turn_splitting": True,
                    "speaker_label_mode": "drop",
                },
            },
        },
    )

    options = engine.resolve_text_options(
        {"profile": "dialogue", "max_punct_run": 6, "speaker_label_mode": "speak"}
    )

    assert options["profile"] == "dialogue"
    assert options["dialogue_turn_splitting"] is True
    assert options["pause_strength"] == "light"
    assert options["speaker_label_mode"] == "speak"
    assert options["max_punct_run"] == 6
    assert options["filter_table_artifacts"] is False
    assert options["filter_reference_artifacts"] is True
    assert options["filter_symbol_noise"] is False
    assert options["remove_urls"] is False


def test_resolve_text_options_falls_back_to_active_profile_when_unknown(monkeypatch):
    _mock_text_processing_config(
        monkeypatch,
        {
            "active_profile": "narration",
            "profiles": {
                "narration": {
                    "filter_table_artifacts": True,
                    "filter_reference_artifacts": False,
                    "filter_symbol_noise": True,
                    "pause_strength": "strong",
                },
            },
        },
    )

    options = engine.resolve_text_options({"profile": "does-not-exist"})
    assert options["profile"] == "narration"
    assert options["pause_strength"] == "strong"
    assert options["filter_reference_artifacts"] is False


def test_build_runtime_preprocessor_uses_resolved_filter_settings():
    options = {
        "filter_table_artifacts": False,
        "filter_reference_artifacts": False,
        "filter_symbol_noise": True,
        "remove_urls": False,
        "remove_punctuation": False,
        "normalize_pause_punctuation": True,
        "pause_strength": "medium",
        "max_punct_run": 3,
    }

    preprocessor = engine._build_runtime_preprocessor(effective_text_options=options)

    assert preprocessor.config["filter_table_artifacts"] is False
    assert preprocessor.config["filter_reference_artifacts"] is False
    assert preprocessor.config["filter_symbol_noise"] is True
    assert preprocessor.config["remove_urls"] is False
