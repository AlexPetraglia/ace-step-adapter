"""HTTP routes that follow the ACE-Step REST contract.

Implemented: GET /health, POST /release_task, POST /query_result, GET /v1/audio.
Every other ACE-Step route is intentionally absent.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .config import Settings
from .engine import AceStepEngine
from .parsing import RequestError, parse_release_request
from .tasks import STATUS_DONE, STATUS_FAILED, TaskManager

log = logging.getLogger("ace_step_adapter.api")

MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".opus": "audio/ogg",
    ".aac": "audio/aac",
}


def create_app(settings: Settings, engine=None) -> FastAPI:
    engine = engine or AceStepEngine(settings)
    manager = TaskManager(engine, settings.output_dir, settings.task_timeout_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        engine.load()
        manager.start()
        yield

    app = FastAPI(title="ace-step-adapter", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.manager = manager

    @app.exception_handler(RequestError)
    async def _bad_request(_request: Request, exc: RequestError):
        return _envelope(None, code=400, error=str(exc))

    @app.get("/health")
    async def health():
        return _envelope(
            {
                "status": "ok",
                "service": "ace-step-adapter",
                "version": __version__,
                "models_initialized": engine.loaded,
                "loaded_model": settings.model,
            }
        )

    @app.post("/release_task")
    async def release_task(request: Request):
        request_body = await _json_body(request)
        parsed = parse_release_request(request_body)
        task = manager.submit(parsed)
        return _envelope(
            {
                "task_id": task.id,
                "status": "queued",
                "queue_position": manager.queue_position(task.id),
            }
        )

    @app.post("/query_result")
    async def query_result(request: Request):
        body = await _json_body(request)
        ids = body.get("task_id_list")
        if isinstance(ids, str):
            ids = _decode_id_list(ids)
        if not isinstance(ids, list) or not ids:
            raise RequestError("'task_id_list' must be a non-empty list")

        items = []
        for task_id in ids:
            task = manager.get(str(task_id))
            if task is None:
                items.append({"task_id": task_id, "status": STATUS_FAILED, "result": "", "error": "unknown task_id"})
                continue
            items.append(_status_item(task, settings.model))
        return _envelope(items)

    @app.get("/v1/audio")
    async def audio(path: str):
        target = _safe_output_file(settings.output_dir, path)
        if target is None:
            return _envelope(None, code=404, error="audio file not found")
        return FileResponse(target, media_type=MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream"))

    return app


def _status_item(task, model: str) -> dict:
    item = {"task_id": task.id, "status": task.status, "result": "", "error": task.error}
    if task.status == STATUS_DONE:
        results = []
        for file in task.files:
            results.append(
                {
                    "file": f"/v1/audio?path={quote(_relative_path(task.id, file))}",
                    "wave": "",
                    "status": STATUS_DONE,
                    "create_time": int(time.time()),
                    "env": "adapter",
                    "prompt": task.request.prompt,
                    "metas": {"duration": task.request.duration_seconds},
                    "generation_info": "",
                    "seed_value": str(task.seed) if task.seed is not None else "",
                    "lm_model": None,
                    "dit_model": model,
                }
            )
        item["result"] = json.dumps(results, ensure_ascii=False)
    return item


def _relative_path(task_id: str, file: Path) -> str:
    return f"{task_id}/{file.name}"


def _safe_output_file(output_dir: Path, requested: str) -> Path | None:
    root = output_dir.resolve()
    candidate = (root / requested).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate


def _decode_id_list(text: str) -> list:
    try:
        decoded = json.loads(text)
    except ValueError as exc:
        raise RequestError("'task_id_list' must be a JSON array") from exc
    return decoded


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError as exc:
        raise RequestError("request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    return body


def _envelope(data, code: int = 200, error: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={
            "data": data,
            "code": code,
            "error": error,
            "timestamp": int(time.time() * 1000),
            "extra": None,
        },
    )
