"""Small bundled inputs and atomic regenerated trajectory frames."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import netCDF4

from .runtime import file_hash, root
from .artifacts import atomic_npz


def load_input(config: dict, precision="fp64") -> dict:
    path = root() / config["input"]
    manifest = json.loads((root() / "glaciers/runs/data/manifest.json").read_text())
    if file_hash(path) != manifest[config["case"]]["sha256"]:
        raise ValueError(f"Input checksum mismatch: {path}")
    dtype = np.float64 if precision == "fp64" else np.float32
    with netCDF4.Dataset(path) as dataset:
        result = {name: np.asarray(var[:], dtype=dtype) for name, var in dataset.variables.items()}
    dx = float(np.median(np.diff(result["x"])))
    dy = float(np.median(np.diff(result["y"])))
    if not np.isclose(dx, dy) or dx <= 0:
        raise ValueError("This study assumes square, increasing raster coordinates")
    if config.get("crop"):
        # Deterministic interior crop around maximum thickness, tagged validation-only.
        size = config["crop"]["size"]
        y, x = np.unravel_index(np.argmax(result["thk"]), result["thk"].shape)
        ny, nx = result["thk"].shape
        y, x = min(max(y - size // 2, 0), ny - size), min(max(x - size // 2, 0), nx - size)
        for name, value in list(result.items()):
            if value.ndim == 2:
                result[name] = value[y : y + size, x : x + size]
        result["x"], result["y"] = result["x"][x : x + size], result["y"][y : y + size]
    result.update(
        dx=np.asarray(dx, dtype=dtype), time=np.asarray(config["time"]["start"], dtype=np.float64)
    )
    if config["case"] == "jakobshavn":
        result["water_level"] = np.full_like(
            result["thk"], config["physics"].get("water_level", 0.0)
        )
        result["Href"] = np.zeros_like(result["thk"])
        ratio = config["physics"]["ice_density"] / config["physics"]["water_density"]
        result["usurf"] = (
            np.maximum(result["topg"], result["water_level"] - ratio * result["thk"])
            + result["thk"]
        )
    return result


def save_state(path: Path, state: dict, uv=None, velocity=None, *, compressed=True, **extra):
    arrays = {k: np.asarray(v) for k, v in state.items()}
    if uv is not None:
        arrays["uv"] = np.asarray(uv)
    if velocity is not None:
        arrays["velocity"] = np.asarray(velocity)
    atomic_npz(path, compressed=compressed, **arrays, **extra)


def load_state(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def frame_key(time: float) -> str:
    return f"t{time:.8f}"
