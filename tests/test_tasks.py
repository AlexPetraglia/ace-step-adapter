import threading
import time

from ace_step_adapter.parsing import GenerationRequest
from ace_step_adapter.tasks import STATUS_DONE, STATUS_FAILED, STATUS_RUNNING, TaskManager
from fakes import FakeEngine


def request(prompt="pop"):
    return GenerationRequest(prompt=prompt, lyrics="la la", duration_seconds=30, vocal_language="pt", audio_format="wav", seed=-1)


def wait_for(task, status, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if task.status == status:
            return
        time.sleep(0.01)
    raise AssertionError(f"task stayed in status {task.status}, expected {status}")


def test_successful_task_records_files_and_seed(tmp_path):
    manager = TaskManager(FakeEngine(), tmp_path, timeout_seconds=30)
    manager.start()
    task = manager.submit(request())

    wait_for(task, STATUS_DONE)
    assert task.files and task.files[0].exists()
    assert task.seed == 4242
    assert task.error is None


def test_engine_failure_is_reported_and_worker_keeps_running(tmp_path):
    engine = FakeEngine(fail_with="out of memory")
    manager = TaskManager(engine, tmp_path, timeout_seconds=30)
    manager.start()

    failed = manager.submit(request("first"))
    wait_for(failed, STATUS_FAILED)
    assert failed.error == "out of memory"

    engine.fail_with = None
    ok = manager.submit(request("second"))
    wait_for(ok, STATUS_DONE)


def test_tasks_run_one_at_a_time_in_submission_order(tmp_path):
    release = threading.Event()
    engine = FakeEngine(block=release)
    manager = TaskManager(engine, tmp_path, timeout_seconds=30)
    manager.start()

    first = manager.submit(request("first"))
    second = manager.submit(request("second"))
    deadline = time.monotonic() + 5
    while not engine.calls and time.monotonic() < deadline:
        time.sleep(0.01)  # wait until the worker has taken the first task
    assert manager.queue_position(second.id) == 1
    assert first.status == STATUS_RUNNING

    release.set()
    wait_for(second, STATUS_DONE)
    assert [call[0].prompt for call in engine.calls] == ["first", "second"]


def test_overdue_task_is_reported_as_failed(tmp_path):
    release = threading.Event()
    manager = TaskManager(FakeEngine(block=release), tmp_path, timeout_seconds=0)
    manager.start()
    task = manager.submit(request())

    time.sleep(0.05)
    overdue = manager.get(task.id)
    assert overdue.status == STATUS_FAILED
    assert overdue.timed_out
    assert "timeout" in overdue.error

    release.set()
    time.sleep(0.05)
    # A late result must not overwrite the failure the client already saw.
    assert manager.get(task.id).status == STATUS_FAILED


def test_unknown_task_is_not_returned(tmp_path):
    manager = TaskManager(FakeEngine(), tmp_path, timeout_seconds=30)
    assert manager.get("does-not-exist") is None
