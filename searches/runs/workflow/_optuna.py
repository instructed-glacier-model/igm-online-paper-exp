"""Sequential Optuna coordinator; each trial runs in a fresh worker process."""
from __future__ import annotations
import copy
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import optuna

from igm_online.artifacts import Run, Paused, atomic_json
from igm_online.config import architecture
from igm_online.runtime import root, file_hash, code_hash


def suggest(trial, name, config):
    spec = architecture(name)
    online = copy.deepcopy(config["online"])
    import yaml

    spaces = yaml.safe_load((root() / "searches/runs/experiment/distributions.yaml").read_text())

    def sample(group):
        values = {}
        for item in spaces[group]:
            item = dict(item)
            key = item.pop("name")
            kind = item.pop("kind")
            values[key] = getattr(trial, "suggest_" + kind)(key, **item)
        return values

    online.update(sample("online"))
    if spec.get("backend") == "fno" or spec["family"] == "fno":
        spec["params"].update(sample("fno"))
        spec["params"]["modes2"] = spec["params"]["modes1"]
    else:
        spec["params"].update(sample("dense"))
        if spec["family"] != "mlp":
            spec["params"].update(sample("cnn"))
    return spec, online


def child_command(args, command):
    values = [
        sys.executable,
        str(root() / "searches/runs/workflow/_trial.py"),
        "--case",
        args.case,
        "--profile",
        args.profile,
        "--namespace",
        args.namespace,
        "--output",
        str(Path(args.output).resolve()),
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
    ]
    if args.config:
        values += ["--config", str(Path(args.config).resolve())]
    if args.resume:
        values += ["--resume"]
    if args.retry_failed:
        values += ["--retry-failed"]
    if getattr(args, "reference_source", None):
        values += ["--reference-source", str(Path(args.reference_source).resolve())]
    return values


def run_child(argv, logfile, timeout):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root() / "_shared")
    begin = time.perf_counter()
    with Path(logfile).open("a") as stream:
        child = subprocess.Popen(
            argv,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=root(),
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return {
                "status": "timeout",
                "returncode": child.returncode,
                "process_seconds": time.perf_counter() - begin,
            }
    return {
        "status": "complete" if code == 0 else "failed",
        "returncode": code,
        "process_seconds": time.perf_counter() - begin,
    }


def sweep(args, config):
    """Trial-addressed sampler seeds make ask/tell replay deterministic on resume."""
    names = (
        args.architectures.split(",") if args.architectures else config["sweep"]["architectures"]
    )
    if any(n.startswith("Direct-") for n in names):
        raise ValueError("Classical sweeps are disabled; benchmark declared direct recipes instead")
    count = (
        args.trials
        if args.trials is not None
        else config["sweep"]["uniform_trials" if args.sampler == "uniform" else "pareto_trials"]
    )
    deferred = getattr(args, "defer_scoring", False)
    if deferred and args.sampler != "uniform":
        raise ValueError(
            "Deferred scoring is supported for uniform sampling; NSGA-II needs objective feedback"
        )
    from igm_online.runtime import activate
    import importlib.metadata

    dep_manifest = activate()
    from igm_online.experiments import require_reference

    ref = require_reference(args, config)
    ref_signature = json.loads((ref / "manifest.json").read_text())["signature"]
    versions = {
        n: importlib.metadata.version(n) for n in ["tensorflow", "keras", "numpy", "optuna"]
    }
    result = []
    for name in names:
        directory = (
            Path(args.output).resolve()
            / args.namespace
            / args.case
            / args.profile
            / "sweep"
            / args.sampler
            / "inclusive"
            / name
        )
        scientific = {
            "workflow": "sweep",
            "config": config,
            "architecture": name,
            "sampler": args.sampler,
            "seed": args.seed,
            "trials": count,
            "input_sha256": file_hash(root() / config["input"]),
            "code_sha256": code_hash(),
            "reference_signature": ref_signature,
            "source_sha256": dep_manifest["package_sha256"],
            "packages": versions,
        }
        if deferred:
            scientific["scoring_deferred"] = True
        with Run(
            directory, scientific, {"coordinator": True}, args.resume, args.retry_failed
        ) as run:
            if run.skip:
                result.append(json.loads((directory / "result.json").read_text()))
                continue
            storage = "sqlite:///" + str(directory / "study.sqlite3")
            pending_path = directory / "pending.json"
            for number in range(count):
                sampler_seed = args.seed + 100003 * number
                sampler = (
                    optuna.samplers.RandomSampler(seed=sampler_seed)
                    if args.sampler == "uniform"
                    else optuna.samplers.NSGAIISampler(
                        seed=sampler_seed, population_size=config["sweep"]["population"]
                    )
                )
                study = optuna.create_study(
                    storage=storage,
                    study_name="online",
                    directions=["minimize", "minimize"],
                    sampler=sampler,
                    load_if_exists=True,
                )
                prior = [t for t in study.trials if t.number == number]
                completed_simulation = directory / "trials" / f"{number:04d}" / "result.json"
                if deferred and completed_simulation.exists():
                    if (
                        pending_path.exists()
                        and json.loads(pending_path.read_text())["number"] == number
                    ):
                        pending_path.unlink()
                    continue
                if prior and prior[0].state.is_finished():
                    if (
                        pending_path.exists()
                        and json.loads(pending_path.read_text())["number"] == number
                    ):
                        pending_path.unlink()
                    continue
                if prior and not pending_path.exists():
                    # A crash between ask() and the durable pending record has
                    # no executable trial specification. Retain it as failed.
                    study.tell(prior[0].number, state=optuna.trial.TrialState.FAIL)
                    atomic_json(
                        directory / "trials" / f"{number:04d}" / "sampling_failure.json",
                        {"status": "interrupted_during_sampling", "trial": number},
                    )
                    continue
                if pending_path.exists():
                    pending = json.loads(pending_path.read_text())
                    if pending["number"] != number:
                        raise RuntimeError("Pending trial disagrees with Optuna storage")
                    trial = optuna.Trial(study, prior[0]._trial_id)
                    spec, online = pending["spec"], pending["online"]
                else:
                    trial = study.ask()
                    if trial.number != number:
                        raise RuntimeError("Unexpected trial numbering")
                    spec, online = suggest(trial, name, config)
                    pending = {"number": number, "spec": spec, "online": online}
                    atomic_json(pending_path, pending)
                folder = directory / "trials" / f"{number:04d}"
                folder.mkdir(parents=True, exist_ok=True)
                job = {
                    **pending,
                    "architecture": name,
                    "config": config,
                    "seed": args.seed + 104729 * number,
                }
                atomic_json(folder / "job.json", job)
                argv = child_command(args, "_trial") + [
                    "--job",
                    str(folder / "job.json"),
                    "--trial-output",
                    str(folder / "run"),
                ]
                if deferred:
                    argv += ["--defer-scoring"]
                # Replaying a pending worker uses its checkpoint; its original seed is in job.json.
                if (folder / "run/manifest.json").exists() and "--resume" not in argv:
                    argv += ["--resume"]
                executions = []
                while True:
                    execution = run_child(
                        argv, folder / "worker.log", config["sweep"]["trial_timeout_seconds"]
                    )
                    executions.append(execution)
                    atomic_json(folder / "worker_attempts.json", executions)
                    if execution["returncode"] != 3:
                        break
                    print(f"GPU {args.gpu} occupied; trial {number} remains queued", flush=True)
                    run.checkpoint(queued_trial=number)
                    time.sleep(15)
                atomic_json(folder / "execution.json", execution)
                worker_status = folder / "run/status.json"
                paused = (
                    worker_status.exists()
                    and json.loads(worker_status.read_text())["status"] == "paused"
                )
                if paused or run.stopped:
                    raise Paused(
                        "Sweep paused with its pending trial and worker checkpoint retained"
                    )
                if execution["status"] == "complete":
                    row = json.loads((folder / "run/result.json").read_text())
                    # Native resume totals include computation from prior attempts.
                    # A final short subprocess must not hide that earlier work.
                    inclusive_time = max(execution["process_seconds"], row["total_seconds"])
                    objective_time = inclusive_time
                    if not deferred:
                        error = row["trajectory"]["mean_speed_rmse"]
                        if error is None:
                            study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        else:
                            study.tell(trial, [error, objective_time])
                    row.update(
                        spec=spec,
                        online=online,
                        trial=number,
                        objective_seconds=objective_time,
                        process_seconds=inclusive_time,
                        last_attempt_process_seconds=execution["process_seconds"],
                    )
                    atomic_json(folder / "result.json", row)
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                pending_path.unlink()
                run.checkpoint(trial=number + 1, total=count)
            rows = [
                json.loads(p.read_text())
                for p in sorted((directory / "trials").glob("*/result.json"))
            ]
            summary = {
                "workflow": "sweep",
                "architecture": name,
                "sampler": args.sampler,
                "scoring_status": "pending" if deferred else "complete",
                "requested_trials": count,
                "completed_trials": len(rows),
                "failed_trials": count - len(rows),
                "trials": rows,
            }
            atomic_json(directory / "result.json", summary)
            result.append(summary)
    return result
