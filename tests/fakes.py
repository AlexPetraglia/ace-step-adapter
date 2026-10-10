"""Test doubles. No test in this repository loads the ACE-Step model or uses the GPU."""

from __future__ import annotations

import threading
from pathlib import Path

from ace_step_adapter.engine import EngineError, EngineResult


class FakeEngine:
    """Mimics AceStepEngine. Writes a small file instead of generating audio."""

    def __init__(self, fail_with: str | None = None, block: threading.Event | None = None, seed: int = 4242):
        self.fail_with = fail_with
        self.block = block
        self.seed = seed
        self.loaded = False
        self.calls: list[tuple[object, Path]] = []

    def load(self) -> None:
        self.loaded = True

    def generate(self, request, save_dir: Path) -> EngineResult:
        self.calls.append((request, save_dir))
        if self.block is not None:
            self.block.wait(timeout=5)
        if self.fail_with:
            raise EngineError(self.fail_with)
        save_dir.mkdir(parents=True, exist_ok=True)
        audio = save_dir / "fake_audio.wav"
        audio.write_bytes(b"RIFF fake wav bytes")
        return EngineResult(files=[audio], seed=self.seed)
