"""Validation of task requests sent to /release_task.

Field names follow the ACE-Step REST API (snake_case aliases are accepted).
Options that this adapter does not support yet are rejected explicitly,
instead of being silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

ALLOWED_FORMATS = {"wav", "mp3", "flac", "wav32", "opus", "aac"}
DEFAULT_DURATION_SECONDS = 120
MIN_DURATION_SECONDS = 10
MAX_DURATION_SECONDS = 600

# Options that need the 5Hz language model. The adapter runs without it.
UNSUPPORTED_FLAGS = {
    "thinking": ("thinking",),
    "use_format": ("use_format", "format"),
    "use_cot_caption": ("use_cot_caption", "cot_caption"),
    "use_cot_language": ("use_cot_language", "cot_language"),
    "sample_mode": ("sample_mode",),
}


class RequestError(ValueError):
    """The request is invalid or asks for an option this adapter does not support."""


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    lyrics: str
    duration_seconds: int
    vocal_language: str
    audio_format: str
    seed: int  # -1 means random, as in ACE-Step


def parse_release_request(body: dict) -> GenerationRequest:
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")

    for flag, names in UNSUPPORTED_FLAGS.items():
        for name in names:
            if body.get(name):
                raise RequestError(f"'{flag}' is not supported by this adapter yet")

    batch_size = body.get("batch_size", 1)
    if batch_size != 1:
        raise RequestError("'batch_size' must be 1 in this adapter")

    prompt = _text(body, "prompt", "caption")
    lyrics = _text(body, "lyrics")

    duration = _first(body, "audio_duration", "duration", default=DEFAULT_DURATION_SECONDS)
    if not isinstance(duration, int) or isinstance(duration, bool):
        raise RequestError("'audio_duration' must be an integer number of seconds")
    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        raise RequestError(
            f"'audio_duration' must be between {MIN_DURATION_SECONDS} and {MAX_DURATION_SECONDS}"
        )

    vocal_language = _first(body, "vocal_language", default="en")
    if not isinstance(vocal_language, str) or not vocal_language.strip():
        raise RequestError("'vocal_language' must be a non-empty string")

    audio_format = _first(body, "audio_format", default="mp3")
    if audio_format not in ALLOWED_FORMATS:
        raise RequestError(f"'audio_format' must be one of: {', '.join(sorted(ALLOWED_FORMATS))}")

    seed = -1
    if not body.get("use_random_seed", True):
        seed = body.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise RequestError("'seed' must be a non-negative integer when 'use_random_seed' is false")

    return GenerationRequest(
        prompt=prompt,
        lyrics=lyrics,
        duration_seconds=duration,
        vocal_language=vocal_language.strip(),
        audio_format=audio_format,
        seed=seed,
    )


def _text(body: dict, *names: str) -> str:
    value = _first(body, *names)
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"'{names[0]}' is required")
    return value.strip()


def _first(body: dict, *names: str, default=None):
    for name in names:
        if name in body and body[name] is not None:
            return body[name]
    return default
