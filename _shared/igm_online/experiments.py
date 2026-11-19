"""Reference and online trajectories sharing one evolution and accuracy protocol."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf

from .artifacts import Run, atomic_json
from .config import architecture
from .data import load_input, load_state, save_state, frame_key
from .evolution import Evolution
from .metrics import compare, historical_ice_mask, trajectory_summary, vector_rms

from .physics import Problem
from .runtime import file_hash, root
from .solver import DirectSolver, ReferenceFailure, verified_solve, velocity_start


def base_directory(args, config):
    return Path(args.output).resolve() / args.namespace / config["case"] / config["profile"]


def scientific(config, context, **values):
    return {
        "config": config,
        "source_commit": context["dependency"]["commit"],
        "source_sha256": context["dependency"]["package_sha256"],
        "code_sha256": context["code_sha256"],
        "packages": context["packages"],
        "input_sha256": file_hash(root() / config["input"]),
        **values,
    }


def reference_directory(args, config, precision="fp64"):
    return (
        Path(args.reference_source).resolve()
        if getattr(args, "reference_source", None) and precision == "fp64"
        else base_directory(args, config) / "reference" / precision
    )


def require_reference(args, config, precision="fp64"):
    directory = reference_directory(args, config, precision)
    status = directory / "status.json"
    if not status.exists() or json.loads(status.read_text())["status"] != "complete":
        raise ReferenceFailure(f"Validated {precision} reference required: {directory}")
    result = json.loads((directory / "result.json").read_text())
    if not result["validated"]:
        raise ReferenceFailure("Reference trajectory is unresolved")
    saved = json.loads((directory / "manifest.json").read_text())["scientific"]
    keys = ("case", "profile", "physics", "time", "input", "crop")
    if {k: saved["config"].get(k) for k in keys} != {k: config.get(k) for k in keys}:
        raise ValueError("Reference configuration differs from this experiment")
    from .runtime import activate

    dependency = activate()
    if saved.get("source_sha256") != dependency["package_sha256"]:
        raise ValueError(
            "Reference source differs from the installed IGM release; regenerate rather than relabel old data"
        )
    if saved.get("input_sha256") != file_hash(root() / config["input"]):
        raise ValueError("Reference input hash differs")
    return directory


def horizon(config):
    t = config["time"]
    return t["start"] + t["duration"] if "duration" in t else t["end"]


def save_times(config):
    t = config["time"]
    end = horizon(config)
    values = list(np.arange(t["start"], end, t["save"], dtype=np.float64))
    if not values or not np.isclose(values[-1], end, rtol=0, atol=1e-9):
        values.append(end)
    return [float(v) for v in values]


def simulate(
    args, config, context, *, precision="fp32", name="CNN", spec=None, reference=False, output=None
):
    directory = (
        Path(output)
        if output
        else reference_directory(args, config, precision)
        if reference
        else base_directory(args, config) / "online" / name / f"seed{args.seed}"
    )
    spec = spec or architecture(name)
    if not reference and spec["family"] != "identity":
        if precision != "fp32":
            raise ValueError(
                "Native neural trajectories currently require FP32; validate native FP64 before enabling it"
            )
        from .native import run as native_run

        return native_run(args, config, context, name, spec, directory)
    sig = scientific(
        config,
        context,
        workflow="reference" if reference else "online",
        precision=precision,
        architecture=None if reference else name,
        architecture_spec=None if reference else spec,
        seed=args.seed,
    )
    defer_scoring = not reference and getattr(args, "defer_scoring", False)
    if defer_scoring:
        sig["scoring_deferred"] = True
    canonical = None if reference or defer_scoring else require_reference(args, config)
    if canonical:
        sig["reference_signature"] = json.loads((canonical / "manifest.json").read_text())[
            "signature"
        ]
    with Run(directory, sig, context, args.resume, args.retry_failed) as run:
        if run.skip:
            return json.loads((directory / "result.json").read_text())
        begin = time.perf_counter()
        problem = Problem(config, precision)
        state = load_input(config, precision)
        historical_mask = state["thk"] > config["metrics"]["historical_ice_threshold"]
        checkpoint = directory / "checkpoint.npz"
        progress_path = directory / "progress.json"
        progress = {
            "steps": 0,
            "frames": [],
            "solves": [],
            "steady_seconds": 0.0,
            "initialization_seconds": 0.0,
            "prior_attempt_seconds": 0.0,
            "checkpoint_seconds": 0.0,
            "checkpoint_count": 0,
            "frame_io_seconds": 0.0,
        }
        if args.resume and checkpoint.exists():
            state = load_state(checkpoint)
            progress = json.loads(progress_path.read_text())
        evolution = Evolution(problem)
        if not (args.resume and checkpoint.exists()):
            state = evolution.initialize_state(state)
        if "smb_last_time" in state:
            evolution.last_smb_time = float(state.pop("smb_last_time"))
            evolution.last_smb = state.pop("smb_last")
        direct = DirectSolver(problem, state)
        retain_candidate = (
            reference
            and precision == "fp32"
            and config["reference"]["fp32_policy"] == "retain_candidate"
        )
        candidate_auditor = None
        times = save_times(config)
        frame_times = {round(float(f["time"]), 8) for f in progress["frames"]}
        needs_solve = not bool(state.pop("_solved", np.asarray("uv" in state)))
        if not progress["solves"]:
            progress["initialization_seconds"] = time.perf_counter() - begin
        runtime = config.get("runtime", {})
        checkpoint_interval = runtime.get("checkpoint_interval_seconds", 60.0)
        compress_checkpoints = runtime.get("compress_checkpoints", False)
        last_checkpoint = time.perf_counter()
        # Evolution and checkpoint writers intentionally discard in-memory uv.
        # Keep the preceding completed physical solution explicitly; on resume
        # the trajectory checkpoint supplies exactly that solution and its time.
        previous_uv = state.get("uv")
        previous_uv_time = float(state["time"]) if previous_uv is not None else None

        def checkpoint_state(uv, velocity):
            nonlocal last_checkpoint
            io_begin = time.perf_counter()
            extra = {}
            if evolution.last_smb is not None:
                extra.update(
                    smb_last=evolution.last_smb, smb_last_time=np.asarray(evolution.last_smb_time)
                )
            state.pop("uv", None)
            state.pop("velocity", None)
            save_state(
                checkpoint,
                state,
                uv,
                velocity,
                compressed=compress_checkpoints,
                _solved=np.asarray(True),
                **extra,
            )
            progress["checkpoint_seconds"] = (
                progress.get("checkpoint_seconds", 0.0) + time.perf_counter() - io_begin
            )
            progress["checkpoint_count"] = progress.get("checkpoint_count", 0) + 1
            atomic_json(progress_path, progress)
            last_checkpoint = time.perf_counter()
            run.checkpoint(time=float(state["time"]), steps=progress["steps"])

        while True:
            current = float(state["time"])
            x = problem.tensor(state)
            initial = not progress["solves"]
            if needs_solve:
                tick = time.perf_counter()
                start = velocity_start(state, previous_uv, config["solver"])
                independence = (
                    reference
                    and config["solver"]["independent_starts"]
                    and any(
                        abs(current - v) < 1e-7
                        for v in (times[0], times[len(times) // 2], times[-1])
                    )
                )

                def save_partial(partial, history):
                    save_state(
                        directory / "solve_checkpoint.npz",
                        {k: v for k, v in state.items() if k not in ("uv", "velocity")},
                        partial,
                    )
                    atomic_json(
                        directory / "solve_checkpoint.json",
                        {
                            "time": current,
                            "history": history,
                            "resume_policy": (
                                "restart this incomplete snapshot from the preceding completed trajectory checkpoint"
                                if config["solver"].get("velocity_warm_start", False)
                                else "restart this incomplete snapshot cold; completed states are resumed"
                            ),
                        },
                    )
                    run.checkpoint(time=current, newton_blocks=len(history))

                uv, diagnostic = verified_solve(
                    direct, state, independent=independence, callback=save_partial, start=start
                )
                diagnostic["velocity_initialization"] = (
                    "previous_solution" if start is not None else "zero"
                )
                diagnostic["initialization_source_time"] = (
                    previous_uv_time if start is not None else None
                )
                atomic_json(
                    directory / "diagnostics" / f"step{progress['steps']:06d}.json", diagnostic
                )
                if reference and diagnostic["status"] != "converged" and not retain_candidate:
                    save_state(directory / "unresolved_state.npz", state, uv)
                    raise ReferenceFailure(
                        f"Unresolved {precision} solve at {current}: {diagnostic['status']}"
                    )
                solve = {
                    "time": current,
                    "seconds": diagnostic["seconds"],
                    "status": diagnostic["status"],
                    "newton_iterations": diagnostic["newton_iterations"],
                    "cg_iterations": diagnostic["cg_iterations"],
                    "velocity_initialization": diagnostic["velocity_initialization"],
                    "initialization_source_time": diagnostic["initialization_source_time"],
                }
                velocity = problem.depth_average(
                    tf.constant(uv, problem.dtype), state["thk"]
                ).numpy()
                if not np.all(np.isfinite(velocity)):
                    raise FloatingPointError("Nonfinite velocity")
                progress["solves"].append(solve)
                if initial:
                    progress["initialization_seconds"] += time.perf_counter() - tick
                else:
                    progress["steady_seconds"] += time.perf_counter() - tick
                state["uv"], state["velocity"] = uv, velocity
            else:
                uv, velocity = state["uv"], state["velocity"]
            previous_uv, previous_uv_time = uv, current
            frame_time = next((t for t in times if abs(current - t) < 1e-7), None)
            saved_frame = frame_time is not None and round(frame_time, 8) not in frame_times
            if saved_frame:
                io_begin = time.perf_counter()
                save_state(
                    directory / "frames" / f"{frame_key(frame_time)}.npz",
                    {k: v for k, v in state.items() if k not in ("uv", "velocity")},
                    uv,
                    velocity,
                )
                progress["frame_io_seconds"] = (
                    progress.get("frame_io_seconds", 0.0) + time.perf_counter() - io_begin
                )
                row = {
                    "time": frame_time,
                    "state_time": current,
                    "status": "awaiting_reference" if defer_scoring else "reference",
                }
                if retain_candidate and config["reference"]["fp32_saved_frame_checks"]:
                    audit_problem = direct.validation_problem
                    if candidate_auditor is None:
                        candidate_auditor = DirectSolver(audit_problem, state)
                    audit_uv, audit = verified_solve(candidate_auditor, state, independent=True)
                    atomic_json(directory / "fp64_checks" / f"{frame_key(frame_time)}.json", audit)
                    audit_velocity = audit_problem.depth_average(
                        tf.constant(audit_uv), state["thk"]
                    ).numpy()
                    rms = vector_rms(
                        velocity - audit_velocity, state["thk"] > config["physics"]["ice_threshold"]
                    )
                    row["fp64_same_geometry_check"] = {
                        "reference_status": audit["status"],
                        "vector_rms": rms,
                        "accuracy_pass": audit["status"] == "converged"
                        and rms < config["solver"]["refinement_rms"],
                        "audit_seconds": audit["seconds"],
                        "decision": "deferred; native residual status retained",
                    }
                if canonical:
                    target = load_state(canonical / "frames" / f"{frame_key(frame_time)}.npz")
                    row.update(
                        compare(
                            velocity,
                            target["velocity"],
                            state.get("thk_true", state["thk"]),
                            target.get("thk_true", target["thk"]),
                            speed_offset=config["metrics"]["speed_offset"],
                            threshold=config["metrics"]["threshold_percent"],
                            historical_mask=historical_ice_mask(config, target, historical_mask),
                            volume_thickness=state["thk"] + state.get("Href", 0.0),
                            reference_volume_thickness=target["thk"] + target.get("Href", 0.0),
                        )
                    )
                progress["frames"].append(row)
                frame_times.add(round(frame_time, 8))
            if (
                saved_frame
                or run.stopped
                or time.perf_counter() - last_checkpoint >= checkpoint_interval
                or current >= times[-1] - 1e-8
            ):
                checkpoint_state(uv, velocity)
            if current >= times[-1] - 1e-8:
                break
            following = min(t for t in times if t > current + 1e-8)
            tick = time.perf_counter()
            state = evolution.advance(state, velocity, following - current)
            progress["steady_seconds"] += time.perf_counter() - tick
            progress["steps"] += 1
            # Resume replays from the last completed solve, including its geometry,
            # optimizer and SMB state. Do not serialize all arrays at every CFL step.
            needs_solve = True
        elapsed = time.perf_counter() - begin
        previous_wall = sum(
            json.loads(p.read_text()).get("elapsed_seconds", 0.0)
            for p in (directory / "attempts").glob("*/status.json")
            if p.parent != run.attempt
        )
        native_convergence_pass = reference and all(
            s.get("status") == "converged" for s in progress["solves"]
        )
        validated = native_convergence_pass and not retain_candidate
        result = {
            "workflow": "reference" if reference else "online",
            "validated": validated,
            "reference_quality": "validated"
            if validated
            else "unresolved_candidate"
            if reference
            else None,
            "reference_decision": "deferred" if retain_candidate and not validated else None,
            "native_convergence_pass": native_convergence_pass,
            "unresolved_solves": sum(s.get("status") != "converged" for s in progress["solves"])
            if reference
            else None,
            "case": config["case"],
            "profile": config["profile"],
            "validation_only": config.get("validation_only", False),
            "architecture": "Direct-CGNewton" if reference else name,
            "precision": precision,
            "seed": args.seed,
            "parameters": int(np.prod(uv.shape)),
            "total_seconds": context["startup_seconds"] + elapsed + previous_wall,
            "startup_seconds": context["startup_seconds"],
            "computation_seconds": progress["initialization_seconds"] + progress["steady_seconds"],
            "initialization_seconds": progress["initialization_seconds"],
            "steady_seconds": progress["steady_seconds"],
            "attempt_wall_seconds": elapsed,
            "previous_attempt_wall_seconds": previous_wall,
            "checkpoint_seconds": progress.get("checkpoint_seconds", 0.0),
            "checkpoint_count": progress.get("checkpoint_count", 0),
            "frame_io_seconds": progress.get("frame_io_seconds", 0.0),
            "resumed": previous_wall > 0,
            "steps": progress["steps"],
            "device": context["device"],
            "packages": context["packages"],
            "newton_iterations": sum(s.get("newton_iterations", 0) for s in progress["solves"]),
            "cg_iterations": sum(s.get("cg_iterations", 0) for s in progress["solves"]),
            "trajectory": trajectory_summary(progress["frames"])
            if not reference and not defer_scoring
            else None,
            "scoring_status": "pending" if defer_scoring else "complete",
            "frames": progress["frames"],
            "source_commit": context["dependency"]["commit"],
        }
        if not reference and config.get("classical"):
            result["classical_budget"] = config["classical"]
        if reference:
            result["reference_acceptance"] = {
                "stopping_rule": config["solver"].get("stopping", "strict"),
                "velocity_tolerance_m_per_year": config["solver"].get("velocity_tolerance"),
                "consecutive_updates": config["solver"].get("consecutive_updates"),
                "independent_agreement_m_per_year": config["solver"].get(
                    "independent_rms", config["solver"]["refinement_rms"]
                ),
                "independent_refinement_steps_per_branch": config["solver"].get(
                    "independent_refinement_steps", []
                ),
                "residual_is_diagnostic": config["solver"].get("stopping") == "marine_velocity",
            }
        atomic_json(directory / "result.json", result)
        return result
