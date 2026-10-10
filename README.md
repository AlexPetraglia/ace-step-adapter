# ace-step-adapter

A local HTTP service that wraps [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) and exposes **the same REST contract as the ACE-Step API** for the routes a music-generation client needs. It is built for clients such as the [gerador-musical](https://github.com/AlexPetraglia/gerador-musical) CLI.

> **Status:** in development. This README describes the goal, the architecture and the planned contract. Nothing here should be read as a validated feature yet.

---

## Why it exists

ACE-Step 1.5 is designed to run on GPUs with limited memory, but on an RTX 3050 with 6 GB the audio generation did not complete. Testing showed that:

- the diffusion stage (DiT) finishes in a few seconds;
- the audio decoding stage (VAE) stalls when the free VRAM drops close to zero;
- the main model stays resident on the GPU for the whole session (`Keeping main model on cuda (persistent)`).

With **INT8 weight-only quantization of the DiT** and **CPU offload between stages** (`offload_to_cpu=True`), a 20-second generation completed, with the VAE decode taking about 1.3 seconds and a valid audio file. Neither option is exposed by the ACE-Step API, so the adapter applies them directly on the ACE-Step Python handler.

The adapter exists to:

1. **Isolate the GPU** in its own process, with its own logs and timeouts, separate from the client.
2. **Keep the ACE-Step API contract**, so a client that already talks to ACE-Step can use the adapter without changes.
3. **Make moving environments easy.** If you later use the official ACE-Step server, on another machine or on a VPS, the client only needs a different URL.

---

## Architecture

```
┌──────────────┐   HTTP (ACE-Step contract)   ┌─────────────────────┐   Python   ┌──────────────────┐
│    Client    │ ───────────────────────────▶ │  ace-step-adapter   │ ─────────▶ │ ACE-Step handler │
│ (e.g. CLI)   │ ◀─────────────────────────── │  (FastAPI, GPU)     │ ◀───────── │ (DiT, VAE, ...)  │
└──────────────┘                              └─────────────────────┘            └──────────────────┘
```

ACE-Step stays installed and unmodified in its own folder. The adapter imports the `acestep` package as a library and controls model loading and generation.

---

## Implemented contract

The first version implements only the routes a generation client needs. The flow is asynchronous: submit a task, poll its status until it finishes, then download the generated file.

| Method | Route | Purpose |
|--------|-------|---------|
| `GET`  | `/health` | Checks that the service is up and the model is loaded. |
| `POST` | `/release_task` | Creates a generation task and returns its `task_id`. |
| `POST` | `/query_result` | Queries the status of one or more tasks (`task_id_list`). |
| `GET`  | `/v1/audio?path=...` | Downloads a generated file. Only files inside the service output folder are served. |

Response format: `{"data": ..., "code": 200, "error": null, "timestamp": ..., "extra": null}`. In the `/query_result` response, `result` is a JSON string, as in the ACE-Step API.

Task status values: `0` in progress, `1` completed, `2` failed.

Other ACE-Step API routes (training, lyric formatting, model management, etc.) are **out of scope** for the first version.

---

## Configuration (planned)

Runtime settings live in the adapter's own configuration file, not in server environment variables:

- path to the ACE-Step installation;
- DiT model (default: `acestep-v15-turbo`);
- DiT quantization (default: `int8_weight_only`);
- `offload_to_cpu` (default: `true`);
- output folder;
- listening host and port (default: `127.0.0.1`).

---

## Requirements

- Windows 10/11, or Linux with an NVIDIA GPU and up-to-date drivers.
- Python compatible with ACE-Step 1.5 (3.11 or 3.12).
- ACE-Step 1.5 installed with its models downloaded, in a separate folder from this repository.
- For INT8 quantization: `torchao`, already present in the ACE-Step environment.

---

## Known limitations

- Tested only on a laptop RTX 3050 with 6 GB. Other hardware has not been validated.
- One generation at a time. The GPU is not shared between tasks.
- The adapter is not a proxy: it runs ACE-Step directly. A stall inside ACE-Step can still block the GPU. The adapter applies timeouts to tasks, but it cannot interrupt a CUDA operation that is already running.
- Version coupling: the adapter uses ACE-Step's internal Python API, which may change between releases. The tested version will be listed in the installation guide.

---

## Installation

The adapter uses its own virtual environment. ACE-Step is installed from its own folder.

1. **Install ACE-Step 1.5** (in a folder separate from this repository) and download its models, following the ACE-Step documentation.

2. **Create the adapter environment** with Python 3.12 (3.11 also works):

   ```powershell
   cd ace-step-adapter
   uv venv --python 3.12 .venv
   uv pip install -p .venv\Scripts\python.exe -e ".[dev]"
   ```

3. **Install ACE-Step's pinned dependencies** into the same environment. The versions come from ACE-Step's own lock file, and the PyTorch CUDA index is required:

   ```powershell
   cd ..\ACE-Step-1.5
   uv export --frozen --no-hashes --no-emit-project --format requirements-txt -o ..\ace-step-adapter\ace-lock-requirements.txt
   cd ..\ace-step-adapter
   # Remove the relative nano-vllm line from ace-lock-requirements.txt, then:
   uv pip install -p .venv\Scripts\python.exe --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match -r ace-lock-requirements.txt
   uv pip install -p .venv\Scripts\python.exe --no-deps -e ..\ACE-Step-1.5\acestep\third_parts\nano-vllm
   uv pip install -p .venv\Scripts\python.exe --no-deps -e ..\ACE-Step-1.5
   ```

4. **Configure**: copy `config.example.toml` to `config.toml` and set `ace_step.project_path`.

5. **Run**:

   ```powershell
   .venv\Scripts\python.exe -m ace_step_adapter --config config.toml
   ```

   The service listens on `http://127.0.0.1:8001` by default. `GET /health` returns `models_initialized: true` once the model is loaded.

6. **Run the tests** (no GPU or model needed):

   ```powershell
   .venv\Scripts\python.exe -m pytest
   ```

The `ace-lock-requirements.txt` file is generated locally and is not versioned.

### Verified setup

- Windows 11, Python 3.12.8, NVIDIA GeForce RTX 3050 6 GB Laptop GPU, driver 581.95.
- PyTorch 2.7.1+cu128, ACE-Step 1.5.0.
- End-to-end test: 20-second WAV generation with `int8_weight_only` and CPU offload, completed in about 9 seconds of generation time.

---

## Relation to ACE-Step

This project is independent and is not affiliated with the ACE-Step project. It depends on ACE-Step 1.5, which is released under the MIT license. Model weights have their own licenses; check them before use.

---

## License

This project is licensed under the [MIT License](LICENSE).
