"""Declared architecture specifications and scientific configuration checks."""
from __future__ import annotations

import copy
import hashlib
import json
import yaml

from .runtime import root


def fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate(config):
    if not isinstance(config["solver"].get("velocity_warm_start", False), bool):
        raise ValueError("solver.velocity_warm_start must be a boolean")
    if config.get("metrics", {}).get("selection_criterion", "both") not in ("median", "both"):
        raise ValueError("metrics.selection_criterion must be median or both")
    for steps in config["solver"].get("independent_refinement_steps", []):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("solver.independent_refinement_steps must contain positive integers")
    stopping = config["solver"].get("stopping", "strict")
    if stopping not in ("strict", "marine_velocity"):
        raise ValueError("solver.stopping must be strict or marine_velocity")
    if stopping == "marine_velocity":
        for key, default in (
            ("velocity_tolerance", 1.0),
            ("independent_rms", 1.0),
            ("consecutive_updates", 3),
            ("max_newton_iterations", 200),
        ):
            if config["solver"].get(key, default) <= 0:
                raise ValueError(f"solver.{key} must be positive")
        for key in ("consecutive_updates", "max_newton_iterations"):
            value = config["solver"].get(key, 3 if key == "consecutive_updates" else 200)
            if int(value) != value:
                raise ValueError(f"solver.{key} must be an integer")
    if config.get("runtime", {}).get("checkpoint_interval_seconds", 60.0) < 0:
        raise ValueError("runtime.checkpoint_interval_seconds must be nonnegative")
    if config["physics"]["sliding_regularization"] != 1.0:
        raise ValueError(
            "This study requires sliding regularization = 1 m/year; use a separate study for other physics"
        )
    if config["reference"]["fp32_policy"] not in ("strict", "retain_candidate"):
        raise ValueError("reference.fp32_policy must be strict or retain_candidate")
    for key in (
        "cg_tol",
        "cg_max_iter",
        "block_iterations",
        "max_blocks",
        "refinement_rms",
        "relative_gradient",
    ):
        if config["solver"][key] <= 0:
            raise ValueError(f"solver.{key} must be positive")


def transitions(spec: str | None, count: int) -> list[int]:
    if not spec or spec == "all":
        return list(range(count))
    selected = set()
    for piece in spec.split(","):
        if ":" in piece:
            low, high = piece.split(":")
            selected.update(range(int(low or 0), int(high or count)))
        else:
            selected.add(int(piece))
    if not selected or min(selected) < 0 or max(selected) >= count:
        raise ValueError(f"Transition selection outside 0..{count-1}")
    return sorted(selected)


def architecture(name: str) -> dict:
    specs = yaml.safe_load((root() / "_shared/configs/architectures/models.yaml").read_text())
    if name not in specs:
        raise ValueError(f"Unknown architecture {name}; choose from {list(specs)}")
    result = copy.deepcopy(specs[name])
    return result
