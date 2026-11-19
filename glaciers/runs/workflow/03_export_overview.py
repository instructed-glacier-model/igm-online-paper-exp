#!/usr/bin/env python3
"""Select the largest thk > 1 m ice footprint from a completed FP64 reference."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
n = p.parse_args()
r = json.loads((n.reference / "result.json").read_text())
if not r.get("validated") or r["precision"] != "fp64":
    raise ValueError("A validated FP64 trajectory is required")
frames = []
for f in sorted((n.reference / "frames").glob("*.npz")):
    with np.load(f) as z:
        frames.append((int((z["thk"] > 1).sum()), -float(z["time"]), f))
_, _, f = max(frames)
with np.load(f) as z:
    ice = z["thk"] > 1
    u, v = z["velocity"]
    speed = np.hypot(u, v)
    wl = z["water_level"] if "water_level" in z else 0
    n.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        n.output,
        x_km=z["x"] / 1000,
        y_km=z["y"] / 1000,
        dx_m=z["dx"],
        dy_m=z["dx"],
        nx=ice.shape[1],
        ny=ice.shape[0],
        ice_mask=ice,
        ocean_mask=(z["topg"] < wl) & ~ice,
        vel_ice=np.where(ice, np.clip(speed, 1, 1000), np.nan),
        u=u,
        v=v,
        slope_proxy=False,
        topg=z["topg"],
        lsurf=z["lsurf"] if "lsurf" in z else z["usurf"] - z["thk"],
    )
n.output.with_suffix(".json").write_text(
    json.dumps(
        dict(
            source_frame_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            time=-max(frames)[1],
            reference_manifest=json.loads((n.reference / "manifest.json").read_text()),
        ),
        indent=2,
    )
    + "\n"
)
