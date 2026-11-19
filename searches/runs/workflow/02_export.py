#!/usr/bin/env python3
"""Export numerical tables/figure data from bundled records or a fresh output directory."""
from pathlib import Path
import argparse, sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "_shared"))
from exports import searches

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, default=ROOT / "searches/figure/data")
p.add_argument("--output", type=Path, default=ROOT / "results/exports/searches")
p.add_argument(
    "--classical",
    type=Path,
    help="Explicitly associated three-case classical benchmark CSV; omit to leave speedups undefined",
)
n = p.parse_args()
if (
    n.output.resolve().is_relative_to(ROOT / "searches/figure/data")
    or n.output.resolve() == n.source.resolve()
):
    p.error("Use a separate export directory; bundled paper data are immutable")
searches(n.source.resolve(), n.output.resolve(), n.classical)
print(n.output.resolve())
