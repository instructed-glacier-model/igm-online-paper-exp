"""Anchor training, regenerated target states, and independent correction branches."""
from __future__ import annotations
import numpy as np
import tensorflow as tf

from .artifacts import Run, atomic_json
from .data import load_state, save_state, frame_key
from .evolution import Evolution
from .experiments import base_directory, require_reference, scientific
from .physics import Problem
from .solver import DirectSolver, verified_solve, ReferenceFailure


def transition_directory(args, config, index):
    return base_directory(args, config) / "transitions" / f"{index:03d}"


def reference_uncertainty(info):
    """Empirical reference resolution, not a certified solution-error bound.

    The marine single-block displacement is from zero to the whole solution;
    it is not a refinement error. Use the independently measured discrepancy.
    Strict block solves additionally provide an actual refinement displacement.
    """
    independent = info.get("independent", {})
    if info.get("stopping_rule") == "marine_velocity":
        if info.get("status") != "converged" or independent.get("status") != "converged":
            raise ReferenceFailure(
                "Marine transition references require successful independent starts"
            )
        value = independent["velocity_rms"]
    else:
        value = max(info.get("velocity_refinement_rms", 0.0), independent.get("velocity_rms", 0.0))
    if not np.isfinite(value) or value < 0:
        raise ReferenceFailure("Invalid empirical reference uncertainty")
    return float(value)


def generate_transition(args, config, context, index):
    ref = require_reference(args, config)
    time = config["transfer"]["anchor_start"] + index * config["transfer"]["anchor_interval"]
    frame = ref / "frames" / f"{frame_key(time)}.npz"
    if not frame.exists():
        raise FileNotFoundError(f"Anchor {time} unavailable in this profile: {frame}")
    directory = transition_directory(args, config, index)
    from .runtime import file_hash

    sig = scientific(
        config, context, workflow="transition", index=index, anchor_sha256=file_hash(frame)
    )
    # Generated transitions are shared by transfer and controlled input ablation. Reuse only an exactly matching completed dependency.
    with Run(directory, sig, context, resume=True, retry_failed=args.retry_failed) as run:
        if run.skip:
            return directory
        problem = Problem(config, "fp64")
        anchor = load_state(frame)
        solver = DirectSolver(problem, anchor)
        uv, info_a = verified_solve(solver, anchor, independent=True)
        atomic_json(directory / "anchor_reference.json", info_a)
        save_state(
            directory / "anchor.npz",
            {k: v for k, v in anchor.items() if k not in ("uv", "velocity")},
            uv,
            problem.depth_average(tf.constant(uv), anchor["thk"]).numpy(),
        )
        if info_a["status"] != "converged":
            raise ReferenceFailure("Anchor reference unresolved")
        target = dict(anchor)
        target["uv"] = uv
        target["velocity"] = problem.depth_average(tf.constant(uv), target["thk"]).numpy()
        end = float(time + config["transfer"]["fine_dt"])
        evolution = Evolution(problem)
        records = []
        while float(target["time"]) < end - 1e-9:
            target = evolution.advance(target, target["velocity"], end - float(target["time"]))
            uv, check = verified_solve(solver, target, independent=False)
            records.append(check)
            target["uv"] = uv
            target["velocity"] = problem.depth_average(tf.constant(uv), target["thk"]).numpy()
            save_state(directory / "target_progress.npz", target)
            atomic_json(directory / "target_progress.json", records)
            run.checkpoint(time=float(target["time"]))
            if check["status"] != "converged":
                raise ReferenceFailure("Target evolution reference unresolved")
        uv, info_b = verified_solve(solver, target, independent=True)
        atomic_json(directory / "target_reference.json", info_b)
        save_state(
            directory / "target.npz",
            {k: v for k, v in target.items() if k not in ("uv", "velocity")},
            uv,
            problem.depth_average(tf.constant(uv), target["thk"]).numpy(),
        )
        if info_b["status"] != "converged":
            raise ReferenceFailure("Target reference unresolved")
        components = {
            "anchor": reference_uncertainty(info_a),
            "target": reference_uncertainty(info_b),
        }
        uncertainty = sum(components.values())
        atomic_json(
            directory / "result.json",
            {
                "workflow": "transition",
                "index": index,
                "anchor_time": time,
                "target_time": end,
                "reference_uncertainty": uncertainty,
                "reference_uncertainty_components": components,
                "reference_uncertainty_method": "Sum of independent-start vector RMS discrepancies for marine halts; strict block solves also include final refinement changes. Empirical resolution estimate, not a certified error bound.",
                "status": "converged",
                "case": config["case"],
            },
        )
    return directory
