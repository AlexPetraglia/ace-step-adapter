"""Wrapper around the ACE-Step handler.

The 'acestep' package is imported lazily, so the rest of the adapter (and its
tests) can run without the model or the GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .parsing import GenerationRequest

log = logging.getLogger("ace_step_adapter.engine")

# Turbo models use 8 steps and no classifier-free guidance.
TURBO_STEPS = 8
TURBO_GUIDANCE = 1.0


class EngineError(RuntimeError):
    """The model failed to load or the generation did not produce audio."""


@dataclass
class EngineResult:
    files: list[Path] = field(default_factory=list)
    seed: int | None = None


class AceStepEngine:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._dit = None
        self._llm = None

    @property
    def loaded(self) -> bool:
        return self._dit is not None

    def load(self) -> None:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        s = self._settings
        log.info("Loading model %s (quantization=%s, offload_to_cpu=%s)", s.model, s.quantization, s.offload_to_cpu)
        handler = AceStepHandler()
        status, ok = handler.initialize_service(
            project_root=str(s.project_path),
            config_path=s.model,
            device=s.device,
            offload_to_cpu=s.offload_to_cpu,
            quantization=s.quantization,
        )
        if not ok:
            raise EngineError(f"model initialization failed: {status}")
        self._dit = handler
        # The language model is never initialized: this adapter does not use it.
        self._llm = LLMHandler()
        log.info("Model loaded: %s", status)

    def generate(self, request: GenerationRequest, save_dir: Path) -> EngineResult:
        if not self.loaded:
            raise EngineError("model is not loaded")

        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        params = GenerationParams(
            task_type="text2music",
            caption=request.prompt,
            lyrics=request.lyrics,
            vocal_language=request.vocal_language,
            duration=request.duration_seconds,
            inference_steps=TURBO_STEPS,
            guidance_scale=TURBO_GUIDANCE,
            seed=request.seed,
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
        )
        config = GenerationConfig(
            batch_size=1,
            audio_format=request.audio_format,
            use_random_seed=request.seed == -1,
        )

        save_dir.mkdir(parents=True, exist_ok=True)
        result = generate_music(self._dit, self._llm, params=params, config=config, save_dir=str(save_dir))
        if not result.success:
            raise EngineError(result.error or result.status_message or "generation failed")

        files = [Path(audio["path"]) for audio in result.audios if audio.get("path")]
        if not files:
            raise EngineError("generation finished without an audio file")

        seed = result.audios[0].get("params", {}).get("seed")
        return EngineResult(files=files, seed=seed if isinstance(seed, int) else None)
