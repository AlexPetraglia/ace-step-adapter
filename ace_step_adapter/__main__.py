"""Command-line entry point: python -m ace_step_adapter --config config.toml"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ConfigError, load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ace-step-adapter", description="Serve the ACE-Step REST contract.")
    parser.add_argument("--config", required=True, type=Path, help="path to the TOML configuration file")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    _configure_logging(settings.logs_dir)

    import uvicorn

    from .api import create_app

    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
    return 0


def _configure_logging(logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ace_step_adapter")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logs_dir / "adapter.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logger.addHandler(console)


if __name__ == "__main__":
    sys.exit(main())
