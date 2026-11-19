#!/usr/bin/env python3
"""Compare independently evolved FP32/FP64 trajectories at every saved frame."""
from pathlib import Path
import argparse, json, sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_shared"))
from exports import csvout, js
from igm_online.metrics import compare, vector_rms

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--fp32", type=Path, required=True)
p.add_argument("--fp64", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
r64 = json.loads((a.fp64 / "result.json").read_text())
if not r64["validated"]:
    raise ValueError("Canonical FP64 reference is unresolved")
f64 = {p.name: p for p in (a.fp64 / "frames").glob("*.npz")}
f32 = {p.name: p for p in (a.fp32 / "frames").glob("*.npz")}
if f32.keys() != f64.keys():
    raise ValueError("Saved frames differ; do not omit failed or missing frames")
rows = []
for name in sorted(f64):
    with np.load(f32[name]) as x, np.load(f64[name]) as y:
        d = compare(
            x["velocity"], y["velocity"], x.get("thk_true", x["thk"]), y.get("thk_true", y["thk"])
        )
        d["time"] = float(y["time"])
        d["vector_rms_m_per_year"] = vector_rms(x["velocity"] - y["velocity"], y["thk"] > 0.1)
        rows.append(d)
if any(r["status"] != "ok" for r in rows):
    raise ValueError("Invalid frame; comparison retained as failure")
csvout(a.output / "frame_metrics.csv", rows)
summary = {}
for k in [
    "speed_median_percent",
    "vector_l2_percent",
    "speed_rmse",
    "vector_rms_m_per_year",
    "thickness_rmse",
]:
    summary["worst_" + k] = max(r[k] for r in rows)
    summary["mean_" + k] = float(np.mean([r[k] for r in rows]))
summary.update(
    frames=len(rows),
    median_only_qualifies=all(r["median_only_qualifies"] for r in rows),
    fp32_strict_reference_accepted=json.loads((a.fp32 / "result.json").read_text())["validated"],
)
js(a.output / "comparison.json", summary)

from igm_online.metrics import trajectory_summary

summary_row = trajectory_summary(rows)
summary_row.update(
    case=r64["case"],
    fp32_signature=json.loads((a.fp32 / "manifest.json").read_text())["signature"],
    fp64_signature=json.loads((a.fp64 / "manifest.json").read_text())["signature"],
)
csvout(a.output / "comparison_summary.csv", [summary_row])
