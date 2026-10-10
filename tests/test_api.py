import json
import time

import pytest
from fastapi.testclient import TestClient

from ace_step_adapter.api import create_app
from ace_step_adapter.config import Settings
from fakes import FakeEngine


@pytest.fixture
def settings(tmp_path):
    return Settings(
        host="127.0.0.1",
        port=8001,
        project_path=tmp_path,
        model="acestep-v15-turbo",
        device="auto",
        quantization="int8_weight_only",
        offload_to_cpu=True,
        task_timeout_seconds=30,
        output_dir=tmp_path / "output",
        logs_dir=tmp_path / "logs",
    )


@pytest.fixture
def client(settings):
    engine = FakeEngine()
    with TestClient(create_app(settings, engine=engine)) as test_client:
        yield test_client


VALID_BODY = {
    "prompt": "bossa nova, soft guitar",
    "lyrics": "[Verse 1]\nCoffee at the window",
    "audio_duration": 30,
    "vocal_language": "pt",
    "audio_format": "wav",
    "batch_size": 1,
    "thinking": False,
    "use_format": False,
    "use_cot_caption": False,
    "use_cot_language": False,
}


def wait_done(client, task_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        item = client.post("/query_result", json={"task_id_list": [task_id]}).json()["data"][0]
        if item["status"] != 0:
            return item
        time.sleep(0.02)
    raise AssertionError("task did not finish")


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["code"] == 200
    assert body["data"]["status"] == "ok"
    assert body["data"]["models_initialized"] is True


def test_release_then_query_then_download(client):
    release = client.post("/release_task", json=VALID_BODY)
    assert release.status_code == 200
    task_id = release.json()["data"]["task_id"]

    item = wait_done(client, task_id)
    assert item["status"] == 1
    result = json.loads(item["result"])[0]
    assert result["file"].startswith("/v1/audio?path=")
    assert result["seed_value"] == "4242"

    audio = client.get(result["file"])
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert audio.content == b"RIFF fake wav bytes"


def test_query_accepts_json_string_list(client):
    task_id = client.post("/release_task", json=VALID_BODY).json()["data"]["task_id"]
    wait_done(client, task_id)
    response = client.post("/query_result", json={"task_id_list": json.dumps([task_id])})
    assert response.json()["data"][0]["task_id"] == task_id


def test_unknown_task_is_reported_as_failed(client):
    item = client.post("/query_result", json={"task_id_list": ["missing"]}).json()["data"][0]
    assert item["status"] == 2
    assert item["error"] == "unknown task_id"


def test_unsupported_option_returns_400(client):
    response = client.post("/release_task", json={**VALID_BODY, "thinking": True})
    assert response.status_code == 400
    assert response.json()["error"].startswith("'thinking' is not supported")


def test_invalid_json_returns_400(client):
    response = client.post("/release_task", content=b"{not json", headers={"content-type": "application/json"})
    assert response.status_code == 400


def test_audio_download_cannot_escape_output_folder(client, settings):
    secret = settings.project_path / "secret.txt"
    secret.write_text("do not serve", encoding="utf-8")
    response = client.get("/v1/audio", params={"path": "../secret.txt"})
    assert response.status_code == 404


def test_audio_download_missing_file_returns_404(client):
    assert client.get("/v1/audio", params={"path": "nothing/here.wav"}).status_code == 404
