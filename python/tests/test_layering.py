from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "gz"

LAYERS = {
    "common": set(),
    "proto": {"common"},
    "codec": {"common", "proto"},
    "model": {"common", "codec"},
    "checkpoints": {"common", "codec"},
    "evaluator": {"common", "proto", "codec", "model", "checkpoints"},
    "trainer": {"common", "proto", "codec", "model", "checkpoints"},
}

TORCH_ALLOWED = {
    PACKAGE_ROOT / "model" / "exphormer.py",
    PACKAGE_ROOT / "checkpoints" / "weights.py",
    PACKAGE_ROOT / "evaluator" / "backends.py",
    PACKAGE_ROOT / "trainer" / "data.py",
    PACKAGE_ROOT / "trainer" / "diagnostics.py",
    PACKAGE_ROOT / "trainer" / "driver.py",
    PACKAGE_ROOT / "trainer" / "loop.py",
    PACKAGE_ROOT / "trainer" / "optim.py",
    PACKAGE_ROOT / "trainer" / "publish.py",
}


def test_import_layering_and_torch_ban() -> None:
    for path in PACKAGE_ROOT.rglob("*.py"):
        module = path.relative_to(PACKAGE_ROOT).parts[0]
        tree = ast.parse(path.read_text())
        for imported in _imports(tree):
            if imported == "torch":
                assert path in TORCH_ALLOWED, f"{path} imports torch"
                continue
            if imported == "gz":
                continue
            if imported.startswith("gz."):
                target = imported.split(".")[1]
                if target == module:
                    continue
                assert target in LAYERS[module], f"{module} may not import {target}: {path}"
                assert target != "evaluator", f"nothing imports evaluator: {path}"


def _imports(tree: ast.AST) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            out.append(node.module)
    return out
