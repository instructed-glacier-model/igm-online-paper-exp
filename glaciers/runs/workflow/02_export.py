#!/usr/bin/env python3
"""Export numerical tables/figure data from bundled records or a fresh output directory."""
from pathlib import Path
import argparse, sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "_shared"))
from exports import references

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, default=ROOT / "glaciers/figure/data")
p.add_argument("--output", type=Path, default=ROOT / "results/exports/glaciers")
p.add_argument("--comparisons", type=Path, help="Fresh three-case FP32-versus-FP64 comparison CSV")
p.add_argument(
    "--neural-timings", type=Path, help="Associated shortlisted neural steady timings CSV"
)
n = p.parse_args()
if (
    n.output.resolve().is_relative_to(ROOT / "glaciers/figure/data")
    or n.output.resolve() == n.source.resolve()
):
    p.error("Use a separate export directory; bundled paper data are immutable")
references(n.source.resolve(), n.output.resolve(), n.comparisons, n.neural_timings)
print(n.output.resolve())
