# File: config.py
# Manages application configuration using environment variables loaded from .env.

import logging
import os
import json
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency behavior
    ort = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

DEFAULT_MODEL_FILES_PATH = PROJECT_ROOT / "model_cache"

_BASE_TEXT_PROFILE_PROCESSING: Dict[str, Any] = {
    "lowercase": True,
    "replace_numbers": True,
    "replace_floats": True,
    "expand_contractions": True,
    "expand_model_names": True,
    "expand_ordinals": True,
    "expand_percentages": True,
    "expand_currency": True,
    "expand_time": True,
    "expand_ranges": True,
    "expand_units": True,
    "separate_alphanumeric_tokens": True,
    "expand_scale_suffixes": True,
    "expand_scientific_notation": True,
    "expand_fractions": True,
    "expand_decades": True,
    "expand_phone_numbers": True,
    "expand_ip_addresses": True,
    "normalize_leading_decimals": True,
    "expand_roman_numerals": False,
    "remove_urls": True,
    "remove_emails": True,
    "remove_html": True,
    "remove_hashtags": False,
    "remove_mentions": False,
    "remove_punctuation": False,
    "remove_stopwords": False,
    "normalize_unicode": True,
    "remove_accents": False,
    "remove_extra_whitespace": True,
    "filter_table_artifacts": True,
    "filter_reference_artifacts": True,
    "filter_symbol_noise": True,
    "normalize_pause_punctuation": True,
    "pause_strength": "medium",
    "max_punct_run": 3,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "server": {
        "host": "0.0.0.0",
        "port": 8005,
        "enable_performance_monitor": False,
    },
    "model": {
        "repo_id": "KittenML/kitten-tts-nano-0.8-fp32",
    },
    "tts_engine": {
        "device": "auto",
    },
    "paths": {
        "model_cache": str(DEFAULT_MODEL_FILES_PATH),
    },
    "generation_defaults": {
        "speed": 1.1,
        "language": "en",
    },
    "audio_output": {
        "format": "wav",
        "sample_rate": 24000,
    },
    "text_processing": {
        "active_profile": "balanced",
        "profiles": {
            "balanced": {
                **_BASE_TEXT_PROFILE_PROCESSING,
                "dialogue_turn_splitting": False,
                "speaker_label_mode": "drop",
            },
            "narration": {
                **_BASE_TEXT_PROFILE_PROCESSING,
                "pause_strength": "strong",
                "dialogue_turn_splitting": False,
                "speaker_label_mode": "drop",
                "max_punct_run": 2,
            },
            "dialogue": {
                **_BASE_TEXT_PROFILE_PROCESSING,
                "pause_strength": "light",
                "dialogue_turn_splitting": True,
                "speaker_label_mode": "drop",
            },
        },
    },
    "ui": {
        "title": "Kitten TTS Server",
        "show_language_select": True,
    },
}

ENV_KEY_MAP: Dict[str, str] = {
    "server.host": "KITTEN_SERVER_HOST",
    "server.port": "KITTEN_SERVER_PORT",
    "server.enable_performance_monitor": "KITTEN_SERVER_ENABLE_PERFORMANCE_MONITOR",
    "model.repo_id": "KITTEN_MODEL_REPO_ID",
    "tts_engine.device": "KITTEN_TTS_DEVICE",
    "paths.model_cache": "KITTEN_MODEL_CACHE",
    "generation_defaults.speed": "KITTEN_GEN_DEFAULT_SPEED",
    "generation_defaults.language": "KITTEN_GEN_DEFAULT_LANGUAGE",
    "audio_output.format": "KITTEN_AUDIO_FORMAT",
    "audio_output.sample_rate": "KITTEN_AUDIO_SAMPLE_RATE",
    "text_processing.active_profile": "KITTEN_TEXT_PROFILE",
    "ui.title": "KITTEN_UI_TITLE",
    "ui.show_language_select": "KITTEN_UI_SHOW_LANGUAGE_SELECT",
}


def _set_nested_value(d: Dict[str, Any], keys: list[str], value: Any):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _get_nested_value(d: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


def _parse_bool(raw_value: Any, default: bool = False) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(raw_value)


class EnvConfigManager:
    """Loads read-only runtime configuration from .env and process environment variables."""

    def __init__(self):
        self._lock = Lock()
        self.config: Dict[str, Any] = {}
        self.load_config()

    def _ensure_default_paths_exist(self):
        try:
            Path(DEFAULT_CONFIG["paths"]["model_cache"]).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Error creating default model cache directory: %s", exc, exc_info=True)

    def _parse_env_file(self) -> Dict[str, str]:
        if not ENV_FILE_PATH.exists():
            return {}

        parsed: Dict[str, str] = {}
        with open(ENV_FILE_PATH, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("export "):
                    line = line[len("export ") :].strip()

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                    value = value[1:-1]

                parsed[key] = value

        return parsed

    def _coerce_env_value(self, raw_value: str, default_value: Any) -> Any:
        if isinstance(default_value, bool):
            return _parse_bool(raw_value, default=default_value)
        if isinstance(default_value, int) and not isinstance(default_value, bool):
            try:
                return int(str(raw_value).strip())
            except (ValueError, TypeError):
                logger.warning("Invalid integer env value '%s'. Falling back to default '%s'.", raw_value, default_value)
                return default_value
        if isinstance(default_value, float):
            try:
                return float(str(raw_value).strip())
            except (ValueError, TypeError):
                logger.warning("Invalid float env value '%s'. Falling back to default '%s'.", raw_value, default_value)
                return default_value
        if isinstance(default_value, dict):
            try:
                parsed = json.loads(str(raw_value).strip())
                if isinstance(parsed, dict):
                    return parsed
                logger.warning(
                    "Expected JSON object for env value '%s'. Falling back to default.",
                    raw_value,
                )
                return default_value
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Invalid JSON env value '%s'. Falling back to default '%s'.",
                    raw_value,
                    default_value,
                )
                return default_value
        if isinstance(default_value, list):
            try:
                parsed = json.loads(str(raw_value).strip())
                if isinstance(parsed, list):
                    return parsed
                logger.warning(
                    "Expected JSON array for env value '%s'. Falling back to default.",
                    raw_value,
                )
                return default_value
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Invalid JSON env value '%s'. Falling back to default '%s'.",
                    raw_value,
                    default_value,
                )
                return default_value
        return str(raw_value)

    def _detect_best_device(self) -> str:
        if ort is None:
            logger.info("onnxruntime not available during device detection. Falling back to CPU.")
            return "cpu"

        try:
            available_providers = ort.get_available_providers()
            logger.debug("Available ONNX Runtime providers: %s", available_providers)

            if "CUDAExecutionProvider" in available_providers:
                logger.info("CUDAExecutionProvider found. Using CUDA mode.")
                return "cuda"

            logger.info("CUDAExecutionProvider not found. Using CPU.")
            return "cpu"
        except Exception as exc:
            logger.warning("Error during ONNX Runtime device detection: %s. Defaulting to CPU.", exc)
            return "cpu"

    def _resolve_paths_and_device(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        configured_device = str(_get_nested_value(config_data, ["tts_engine", "device"], "auto")).strip().lower()

        if configured_device == "auto":
            resolved_device = self._detect_best_device()
        elif configured_device in {"cuda", "gpu"}:
            resolved_device = "cuda"
        elif configured_device == "cpu":
            resolved_device = "cpu"
        else:
            logger.warning(
                "Invalid KITTEN_TTS_DEVICE '%s'. Using auto device detection instead.",
                configured_device,
            )
            resolved_device = self._detect_best_device()

        _set_nested_value(config_data, ["tts_engine", "device"], resolved_device)

        model_cache_raw = _get_nested_value(config_data, ["paths", "model_cache"])
        if isinstance(model_cache_raw, str):
            _set_nested_value(config_data, ["paths", "model_cache"], Path(model_cache_raw))

        logger.info("TTS processing device resolved to: %s", resolved_device)
        return config_data

    def _load_from_environment(self) -> Dict[str, Any]:
        base_config = deepcopy(DEFAULT_CONFIG)
        env_file_values = self._parse_env_file()

        for key_path, env_key in ENV_KEY_MAP.items():
            raw_value = os.environ.get(env_key, env_file_values.get(env_key))
            if raw_value is None:
                continue

            default_value = _get_nested_value(DEFAULT_CONFIG, key_path.split("."))
            coerced_value = self._coerce_env_value(raw_value, default_value)
            _set_nested_value(base_config, key_path.split("."), coerced_value)

        return self._resolve_paths_and_device(base_config)

    def load_config(self) -> Dict[str, Any]:
        with self._lock:
            self._ensure_default_paths_exist()
            self.config = self._load_from_environment()
            return deepcopy(self.config)

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        with self._lock:
            value = _get_nested_value(self.config, keys, default)
        return deepcopy(value) if isinstance(value, (dict, list)) else value

    def get_string(self, key_path: str, default: Optional[str] = None) -> str:
        value = self.get(key_path, default)
        if value is None:
            return default if default is not None else ""
        return str(value)

    def get_int(self, key_path: str, default: Optional[int] = None) -> int:
        value = self.get(key_path, default)
        if value is None:
            return default if default is not None else 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return default if isinstance(default, int) else 0

    def get_float(self, key_path: str, default: Optional[float] = None) -> float:
        value = self.get(key_path, default)
        if value is None:
            return default if default is not None else 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return default if isinstance(default, float) else 0.0

    def get_bool(self, key_path: str, default: Optional[bool] = None) -> bool:
        value = self.get(key_path, default)
        return _parse_bool(value, default if default is not None else False)

    def get_path(
        self,
        key_path: str,
        default_str_path: Optional[str] = None,
        ensure_absolute: bool = False,
    ) -> Path:
        value = self.get(key_path)

        if isinstance(value, Path):
            path_obj = value
        elif isinstance(value, str):
            path_obj = Path(value)
        elif default_str_path is not None:
            path_obj = Path(default_str_path)
        else:
            path_obj = Path(".")

        return path_obj.resolve() if ensure_absolute else path_obj

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self.config)

    def _prepare_config_for_saving(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        config_copy = deepcopy(config_dict)
        model_cache_path = _get_nested_value(config_copy, ["paths", "model_cache"])
        if isinstance(model_cache_path, Path):
            _set_nested_value(config_copy, ["paths", "model_cache"], str(model_cache_path))
        return config_copy


config_manager = EnvConfigManager()


def _get_default_from_structure(key_path: str) -> Any:
    return _get_nested_value(DEFAULT_CONFIG, key_path.split("."))


def get_host() -> str:
    return config_manager.get_string("server.host", _get_default_from_structure("server.host"))


def get_port() -> int:
    return config_manager.get_int("server.port", _get_default_from_structure("server.port"))


def get_audio_output_format() -> str:
    return config_manager.get_string(
        "audio_output.format", _get_default_from_structure("audio_output.format")
    )


def get_model_repo_id() -> str:
    return config_manager.get_string("model.repo_id", _get_default_from_structure("model.repo_id"))


def get_tts_device() -> str:
    return config_manager.get_string("tts_engine.device", _get_default_from_structure("tts_engine.device"))


def get_model_cache_path(ensure_absolute: bool = True) -> Path:
    return config_manager.get_path(
        "paths.model_cache",
        str(_get_default_from_structure("paths.model_cache")),
        ensure_absolute=ensure_absolute,
    )


def get_gen_default_speed() -> float:
    return config_manager.get_float(
        "generation_defaults.speed",
        _get_default_from_structure("generation_defaults.speed"),
    )


def get_gen_default_language() -> str:
    return config_manager.get_string(
        "generation_defaults.language",
        _get_default_from_structure("generation_defaults.language"),
    )


def get_audio_sample_rate() -> int:
    return config_manager.get_int(
        "audio_output.sample_rate",
        _get_default_from_structure("audio_output.sample_rate"),
    )


def get_ui_title() -> str:
    return config_manager.get_string("ui.title", _get_default_from_structure("ui.title"))


def get_full_config_for_template() -> Dict[str, Any]:
    return config_manager._prepare_config_for_saving(config_manager.get_all())


# --- End File: config.py ---
