"""Loading and validation of the adapter configuration (TOML)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ALLOWED_QUANTIZATIONS = {"int8_weight_only", "fp8_weight_only", "w8a8_dynamic"}


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    project_path: Path
    model: str
    device: str
    quantization: str | None
    offload_to_cpu: bool
    task_timeout_seconds: int
    output_dir: Path
    logs_dir: Path


def load_settings(path: Path) -> Settings:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path.name}: {exc}") from exc

    base = path.resolve().parent
    server = _section(data, "server")
    ace = _section(data, "ace_step")
    tasks = _section(data, "tasks")
    output = _section(data, "output")
    logging_section = _section(data, "logging")

    port = _int(server, "port", 1, 65535)
    timeout = _int(tasks, "timeout_seconds", 1, 86400)

    quantization = _text(ace, "quantization")
    if quantization == "none":
        quantization_value = None
    elif quantization in ALLOWED_QUANTIZATIONS:
        quantization_value = quantization
    else:
        allowed = ", ".join(sorted(ALLOWED_QUANTIZATIONS | {"none"}))
        raise ConfigError(f"ace_step.quantization must be one of: {allowed}")

    offload = ace.get("offload_to_cpu", False)
    if not isinstance(offload, bool):
        raise ConfigError("ace_step.offload_to_cpu must be true or false")

    project_path = Path(_text(ace, "project_path"))
    if not project_path.is_absolute():
        project_path = base / project_path
    if not project_path.is_dir():
        raise ConfigError(f"ace_step.project_path does not exist: {project_path}")

    return Settings(
        host=_text(server, "host"),
        port=port,
        project_path=project_path,
        model=_text(ace, "model"),
        device=_text(ace, "device"),
        quantization=quantization_value,
        offload_to_cpu=offload,
        task_timeout_seconds=timeout,
        output_dir=_resolve(base, _text(output, "directory")),
        logs_dir=_resolve(base, _text(logging_section, "directory")),
    )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _section(data: dict, name: str) -> dict:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing section [{name}]")
    return value


def _text(section: dict, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty string")
    return value.strip()


def _int(section: dict, key: str, low: int, high: int) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ConfigError(f"'{key}' must be an integer between {low} and {high}")
    return value
