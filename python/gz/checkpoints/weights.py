from __future__ import annotations

from pathlib import Path
from typing import Any


def save_state_dict(path: str | Path, state_dict: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    save_file(state_dict, str(path))


def load_state_dict(path: str | Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(path))
