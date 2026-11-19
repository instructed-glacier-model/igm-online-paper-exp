"""Atomic, configuration-addressed experiments with immutable attempt histories."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from .config import fingerprint


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(clean_json(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_npz(path: Path, *, compressed=True, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        writer = np.savez_compressed if compressed else np.savez
        writer(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Paused(RuntimeError):
    pass


class Run:
    def __init__(
        self, directory: Path, scientific: dict, provenance: dict, resume=False, retry_failed=False
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.stopped = False
        self.started = time.perf_counter()
        self.skip = False
        signature = fingerprint(scientific)
        manifest = self.directory / "manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if old["signature"] != signature:
                self.lock.close()
                raise ValueError(f"Resume configuration/source mismatch at {self.directory}")
            if not resume:
                self.lock.close()
                raise FileExistsError(
                    f"Run exists; use --resume or a new namespace: {self.directory}"
                )
            previous = self.directory / "status.json"
            status = (
                json.loads(previous.read_text())["status"] if previous.exists() else "interrupted"
            )
            if status == "complete":
                self.skip = True
            if status == "failed" and not retry_failed:
                self.lock.close()
                raise RuntimeError("Failed attempt retained; use --resume --retry-failed")
        else:
            atomic_json(manifest, {"signature": signature, "scientific": scientific})
        self.attempt = self.directory / "attempts" / str(time.time_ns())
        if not self.skip:
            self.attempt.mkdir(parents=True)
            atomic_json(self.attempt / "provenance.json", provenance)
        self._handlers = {}

    def __enter__(self):
        if not self.skip:
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._stop)
            self.status("running")
        return self

    def _stop(self, *_):
        self.stopped = True

    def status(self, status, **extra):
        record = {
            "status": status,
            "utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "elapsed_seconds": time.perf_counter() - self.started,
            "attempt": self.attempt.name,
            **extra,
        }
        atomic_json(self.attempt / "status.json", record)
        atomic_json(self.directory / "status.json", record)

    def checkpoint(self, **extra):
        self.status("running", **extra)
        if self.stopped:
            raise Paused("Signal received; state checkpointed")

    def __exit__(self, kind, value, tb):
        if not self.skip:
            if kind:
                status = "paused" if issubclass(kind, Paused) else "failed"
                atomic_json(
                    self.attempt / "failure.json",
                    {
                        "type": kind.__name__,
                        "message": str(value),
                        "traceback": "".join(traceback.format_exception(kind, value, tb)),
                    },
                )
                self.status(status, error=str(value))
            else:
                self.status("complete")
            for sig, handler in self._handlers.items():
                signal.signal(sig, handler)
        self.lock.close()


def checkpoint_weights(path: Path, model, *, compressed=True, **extra):
    weights = model.get_weights()
    atomic_npz(
        path,
        compressed=compressed,
        **{f"weight_{i:04d}": w for i, w in enumerate(weights)},
        n_weights=np.asarray(len(weights)),
        **extra,
    )


def restore_weights(path: Path, model) -> dict:
    with np.load(path, allow_pickle=False) as data:
        model.set_weights([data[f"weight_{i:04d}"] for i in range(int(data["n_weights"]))])
        return {
            k: data[k].copy()
            for k in data.files
            if not k.startswith("weight_") and k != "n_weights"
        }
