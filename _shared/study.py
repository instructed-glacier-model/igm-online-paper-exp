"""Small helpers shared by the numbered study scripts (not a command interface)."""
from pathlib import Path
from types import SimpleNamespace
import argparse, copy, json, sys, time
import yaml

ROOT = Path(__file__).resolve().parent.parent


def options(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "results")
    p.add_argument("--namespace", default="rerun")
    p.add_argument("--gpu", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference-source", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--case", choices=["aletsch", "valais", "jakobshavn"])
    p.add_argument("--architectures", help="Optional comma-separated subset")
    p.add_argument("--transitions", help="Optional zero-based indices/ranges")
    return p


def prepare(ns, config):
    from igm_online.runtime import configure_device, activate, provenance

    start = time.perf_counter()
    device = configure_device(ns.gpu, ns.seed)
    dep = activate()
    ctx = provenance(device, dep)
    ctx["startup_seconds"] = time.perf_counter() - start
    return ctx


def arguments(ns, config):
    a = copy.copy(ns)
    a.case = config["case"]
    a.profile = config["profile"]
    a.output = a.output.resolve()
    a.defer_scoring = False
    return a


def resolved(path):
    c = yaml.safe_load(Path(path).read_text())
    # Complete frozen configurations take precedence over later defaults.
    if "physics" in c and "solver" in c:
        c["solver"].setdefault("stopping", "strict")
        c.pop("pipeline", None)
        c.pop("ntk", None)
        from igm_online.config import validate

        validate(c)
    return c
