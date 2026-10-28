"""Load and validate config.yaml.

Environment variables override secrets: LIMI_API_KEY for the OpenAI-compatible
backend, ANTHROPIC_API_KEY for the Anthropic backend.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints
from urllib.parse import urlparse

import yaml

from limi.logutil import mask_secret

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

log = logging.getLogger("limi.config")

STT_PROVIDERS = ("whisper_local", "google")
TTS_PROVIDERS = ("xvibe_local", "piper_local", "silero_local", "gtts")
LLM_PROVIDERS = ("local_gguf", "api", "anthropic")
WAKE_BACKENDS = ("auto", "openwakeword", "stt")


class ConfigError(Exception):
    """Raised when the config is unusable. `problems` has one line per issue."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass
class STTConfig:
    provider: str = "whisper_local"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    # Used when the top-level language is 'auto': detection is limited to these.
    auto_languages: list[str] = field(default_factory=lambda: ["ru", "en"])


@dataclass
class TTSConfig:
    provider: str = "xvibe_local"
    xvibe_voice: str = "male_deep"  # built-in voice name or path to a WAV sample
    piper_voice: str = "models/tts/ru_RU-irina-medium.onnx"
    silero_speaker: str = "xenia"


@dataclass
class LocalGGUFConfig:
    model_path: str = "models/llm/model.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = 0


@dataclass
class APIConfig:
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key: str = ""


@dataclass
class AnthropicConfig:
    model: str = "claude-sonnet-5-5"
    api_key: str = ""


@dataclass
class LLMConfig:
    provider: str = "local_gguf"
    local_gguf: LocalGGUFConfig = field(default_factory=LocalGGUFConfig)
    api: APIConfig = field(default_factory=APIConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    temperature: float = 0.7
    history_len: int = 20
    max_tokens: int = 400
    max_tool_rounds: int = 2
    system_prompt_path: str = "prompts/system.txt"


@dataclass
class WakeConfig:
    # auto: openWakeWord if a model is set and the package is installed, else STT matching
    backend: str = "auto"
    openwakeword_model: str = ""
    threshold: float = 0.5
    stt_match_ratio: float = 0.75  # fuzzy match strength for the STT backend


@dataclass
class AudioConfig:
    input_device: str = ""  # sounddevice index or name; empty = system default
    output_device: str = ""
    vad_min_rms: float = 250.0  # raise in noisy rooms, lower for quiet mics
    end_silence_s: float = 0.8


@dataclass
class BehaviorConfig:
    sleep_after_s: float = 30.0
    listen_timeout_s: float = 8.0
    max_utterance_s: float = 15.0
    confirm_timeout_s: float = 15.0
    interrupt_on_wake: bool = True


@dataclass
class DisplayConfig:
    width: int = 480
    height: int = 320
    fullscreen: bool = False


@dataclass
class MemoryConfig:
    db_path: str = "data/limi.db"


@dataclass
class Config:
    name: str = "Limi"
    language: str = "ru-RU"
    wake_words: list[str] = field(default_factory=lambda: ["limi", "лими"])
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    sounds_dir: str = "sounds"

    def secrets(self) -> list[str]:
        """Every non-empty API key, for log redaction."""
        return [k for k in (self.llm.api.api_key, self.llm.anthropic.api_key) if k]

    def describe(self) -> str:
        """One-line summary for the startup log (keys masked)."""
        llm = self.llm
        if llm.provider == "api":
            brain = f"api model={llm.api.model} key={mask_secret(llm.api.api_key)}"
        elif llm.provider == "anthropic":
            brain = f"anthropic model={llm.anthropic.model} key={mask_secret(llm.anthropic.api_key)}"
        else:
            brain = f"local_gguf {llm.local_gguf.model_path}"
        return f"stt={self.stt.provider} tts={self.tts.provider} llm={brain}"


def resolve_path(path: str | Path) -> Path:
    """Relative paths are relative to the project root, not the cwd."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


# --- building a Config from a plain dict -----------------------------------

_BAD = object()


def _coerce(tp: Any, value: Any, path: str, problems: list[str]) -> Any:
    origin = get_origin(tp)
    if origin is list:
        (item,) = get_args(tp)
        if isinstance(value, list) and all(isinstance(v, item) for v in value):
            return list(value)
        problems.append(f"{path}: expected a list of {item.__name__}, got {value!r}")
        return _BAD
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif tp is str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return str(value)
    problems.append(f"{path}: expected {tp.__name__}, got {type(value).__name__} ({value!r})")
    return _BAD


def _build(cls: type, data: Any, path: str, problems: list[str]) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        problems.append(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")
        data = {}
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    for key in data:
        if key not in known:
            log.warning("Unknown config key ignored: %s", f"{path}.{key}" if path else key)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        tp = hints[f.name]
        sub = f"{path}.{f.name}" if path else f.name
        if is_dataclass(tp):
            kwargs[f.name] = _build(tp, data.get(f.name), sub, problems)
        elif f.name in data:
            value = _coerce(tp, data[f.name], sub, problems)
            if value is not _BAD:
                kwargs[f.name] = value
    return cls(**kwargs)


# --- validation --------------------------------------------------------------


def _is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("192.168.") or host.endswith(".local")


def _choice(problems: list[str], key: str, value: str, allowed: tuple[str, ...]) -> bool:
    if value in allowed:
        return True
    problems.append(f"{key}: '{value}' is not valid, use one of: {', '.join(allowed)}")
    return False


def _positive(problems: list[str], key: str, value: float) -> None:
    if value <= 0:
        problems.append(f"{key}: must be greater than 0 (got {value})")


def validate(cfg: Config) -> list[str]:
    """Return a list of human-readable problems; empty means the config is usable."""
    p: list[str] = []
    if not cfg.language.strip():
        p.append("language: must not be empty (e.g. ru-RU, en-US or auto)")
    if not cfg.wake_words:
        p.append("wake_words: needs at least one word")
    _choice(p, "stt.provider", cfg.stt.provider, STT_PROVIDERS)
    _choice(p, "wake.backend", cfg.wake.backend, WAKE_BACKENDS)

    if cfg.tts.provider == "xvibe_local" and cfg.tts.xvibe_voice not in ("male_deep", "female_silky", "male_authoritative"):
        if not resolve_path(cfg.tts.xvibe_voice).is_file():
            p.append(f"tts.xvibe_voice: not a built-in voice and file not found: {cfg.tts.xvibe_voice}")
    if _choice(p, "tts.provider", cfg.tts.provider, TTS_PROVIDERS) and cfg.tts.provider == "piper_local":
        voice = resolve_path(cfg.tts.piper_voice)
        if not voice.is_file():
            p.append(
                f"tts.piper_voice: voice file not found: {voice}\n"
                "      Download a Piper .onnx voice (plus its .onnx.json) to that path, "
                "or set tts.provider: gtts for development."
            )

    llm = cfg.llm
    if _choice(p, "llm.provider", llm.provider, LLM_PROVIDERS):
        if llm.provider == "local_gguf":
            model = resolve_path(llm.local_gguf.model_path)
            if not model.is_file():
                p.append(
                    f"llm.local_gguf.model_path: model file not found: {model}\n"
                    "      Put a .gguf file there (Limi never downloads models), "
                    "or set llm.provider: api."
                )
            _positive(p, "llm.local_gguf.n_ctx", llm.local_gguf.n_ctx)
        elif llm.provider == "api":
            if not llm.api.base_url.startswith(("http://", "https://")):
                p.append(f"llm.api.base_url: must start with http:// or https:// (got '{llm.api.base_url}')")
            if not llm.api.model.strip():
                p.append("llm.api.model: must not be empty")
            if not llm.api.api_key and not _is_local_url(llm.api.base_url):
                p.append("llm.api.api_key: no API key. Set env LIMI_API_KEY (preferred) or llm.api.api_key.")
        else:
            if not llm.anthropic.model.strip():
                p.append("llm.anthropic.model: must not be empty")
            if not llm.anthropic.api_key:
                p.append("llm.anthropic.api_key: no API key. Set env ANTHROPIC_API_KEY (preferred) or the config key.")
    if llm.history_len < 0:
        p.append("llm.history_len: must not be negative")
    if llm.max_tool_rounds < 0:
        p.append("llm.max_tool_rounds: must not be negative")
    _positive(p, "llm.max_tokens", llm.max_tokens)

    _positive(p, "behavior.sleep_after_s", cfg.behavior.sleep_after_s)
    _positive(p, "behavior.listen_timeout_s", cfg.behavior.listen_timeout_s)
    _positive(p, "behavior.max_utterance_s", cfg.behavior.max_utterance_s)
    _positive(p, "behavior.confirm_timeout_s", cfg.behavior.confirm_timeout_s)
    _positive(p, "display.width", cfg.display.width)
    _positive(p, "display.height", cfg.display.height)
    _positive(p, "audio.vad_min_rms", cfg.audio.vad_min_rms)
    _positive(p, "audio.end_silence_s", cfg.audio.end_silence_s)

    if not resolve_path(cfg.sounds_dir).is_dir():
        log.warning("sounds_dir not found: %s (event sounds will be skipped)", resolve_path(cfg.sounds_dir))
    return p


# --- loading -----------------------------------------------------------------


def _apply_env(cfg: Config) -> None:
    """Environment variables win over config.yaml for secrets."""
    api_key = os.environ.get("LIMI_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        cfg.llm.api.api_key = api_key
    elif openrouter_key and "openrouter.ai" in cfg.llm.api.base_url:
        # OpenRouter's own variable, used only when base_url points at OpenRouter
        cfg.llm.api.api_key = openrouter_key
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        cfg.llm.anthropic.api_key = anthropic_key


def load_config(path: str | Path | None = None, validate_config: bool = True) -> Config:
    """Read config.yaml (or defaults if absent), apply env, validate.

    Raises ConfigError listing every problem found.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    problems: list[str] = []
    data: Any = {}
    if cfg_path.is_file():
        try:
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError([f"{cfg_path}: cannot read config: {exc}"]) from exc
    else:
        log.warning("%s not found, using built-in defaults (copy config.yaml.example to config.yaml)", cfg_path)

    cfg = _build(Config, data, "", problems)
    _apply_env(cfg)
    if validate_config and not problems:
        problems.extend(validate(cfg))
    if problems:
        raise ConfigError(problems)
    return cfg


def read_db_path(path: str | Path | None = None) -> Path:
    """Leniently find memory.db_path without validating the rest of the config."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    db = MemoryConfig().db_path
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        db = str((data.get("memory") or {}).get("db_path", db))
    except (OSError, yaml.YAMLError, AttributeError):
        pass
    return resolve_path(db)
