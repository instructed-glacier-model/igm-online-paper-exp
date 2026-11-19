"""Resolve and verify the dependency before importing any IGM module."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

from . import IGM_VERSION


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def output_root() -> Path:
    return Path(os.environ.get("IGM_ONLINE_OUTPUT", root() / "results")).resolve()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_hash(path: Path) -> str:
    if not Path(path).is_dir():
        raise FileNotFoundError(f"Missing source directory: {path}")
    h = hashlib.sha256()
    for item in sorted(Path(path).rglob("*")):
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
            h.update(str(item.relative_to(path)).encode())
            h.update(bytes.fromhex(file_hash(item)))
    return h.hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for label, path in (
        ("shared", root() / "_shared"),
        ("user", root() / "searches/runs/user"),
        ("search_workflow", root() / "searches/runs/workflow"),
        ("snapshot_workflow", root() / "transfer/runs/workflow"),
        ("ablation_workflow", root() / "input_ablation/runs/workflow"),
        ("reference_workflow", root() / "glaciers/runs/workflow"),
        ("seed_workflow", root() / "seed_reruns/runs/workflow"),
    ):
        if path.exists():
            h.update(label.encode())
            h.update(bytes.fromhex(tree_hash(path)))
    return h.hexdigest()


def activate() -> dict:
    """Verify the installed release, including its wheel RECORD hashes.

    Never change sys.path or borrow an editable sibling checkout. This also
    detects modified installed IGM files rather than trusting a version label.
    """
    import base64

    dist = importlib.metadata.distribution("igm-model")
    if dist.version != IGM_VERSION:
        raise RuntimeError(f"Install igm-model=={IGM_VERSION}; found {dist.version}")
    direct = dist.read_text("direct_url.json")
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise RuntimeError("An editable IGM installation is not a verified PyPI release")
    import igm

    origin = Path(igm.__file__).resolve()
    expected = Path(dist.locate_file("igm/__init__.py")).resolve()
    if origin != expected:
        raise RuntimeError(f"Installed release is shadowed by {origin}")
    checked = 0
    for entry in dist.files or []:
        if str(entry).startswith("igm/") and entry.hash:
            digest = hashlib.new(
                entry.hash.mode, Path(dist.locate_file(entry)).read_bytes()
            ).digest()
            if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != entry.hash.value:
                raise RuntimeError(f"Modified IGM release file: {entry}")
            checked += 1
    if not checked:
        raise RuntimeError("IGM wheel RECORD is unavailable; install from PyPI")
    return dict(
        version=IGM_VERSION,
        commit=None,
        import_origin=str(origin),
        package_sha256=tree_hash(origin.parent),
        checked_release_files=checked,
    )


def configure_device(gpu: str = "cpu", seed: int = 0) -> dict:
    """Called before TensorFlow import; physical indices are resolved to UUIDs."""
    import fcntl

    record = {"requested": gpu, "hostname": socket.gethostname()}
    if gpu == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,pci.bus_id,driver_version,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        fields = [x.strip() for x in raw.split(",")]
        index, uuid = fields[:2]
        lock = open(f"/tmp/igm-online-{socket.gethostname()}-{uuid}.lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError(f"GPU {index} is occupied by another worker") from None
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
        )
        if uuid in apps or float(fields[3]) > 300 or float(fields[5]) > 5:
            lock.close()
            raise RuntimeError(f"GPU {index} is occupied; queue this job and retry later")
        # Hold lock for process lifetime. A second occupancy check follows the lock.
        configure_device._lock = lock
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        record.update(
            index=int(index),
            uuid=uuid,
            name=fields[2],
            pci_bus_id=fields[6],
            driver=fields[7],
            power_limit_W=float(fields[8]),
            temperature_C=float(fields[9]),
            occupancy_query=raw,
        )
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/igm-online-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/igm-online-cache")
    import tensorflow as tf

    tf.config.experimental.enable_tensor_float_32_execution(False)
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(seed)
    record["tensor_float_32_execution"] = False
    record["visible_devices"] = [d.name for d in tf.config.get_visible_devices()]
    if gpu != "cpu" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Requested GPU is not visible to TensorFlow")
    return record


def provenance(device: dict, dependency: dict) -> dict:
    packages = {}
    for name in (
        "tensorflow",
        "keras",
        "numpy",
        "scipy",
        "netCDF4",
        "optuna",
        "omegaconf",
        "matplotlib",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "device": device,
        "dependency": dependency,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "conda_environment": Path(sys.prefix).name,
        "shell_conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": packages,
        "code_sha256": code_hash(),
        "configs_sha256": tree_hash(root() / "_shared/configs"),
    }
