"""Load the project's local configuration or the checked-in defaults."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    if path is not None:
        with Path(path).open(encoding="utf-8") as config_file:
            return json.load(config_file)

    try:
        with (PROJECT_ROOT / "config.json").open(encoding="utf-8") as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        with (PROJECT_ROOT / "config.example.json").open(encoding="utf-8") as config_file:
            return json.load(config_file)
