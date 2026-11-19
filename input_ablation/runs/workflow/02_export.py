#!/usr/bin/env python3
"""Export numerical tables/figure data from bundled records or a fresh output directory."""
from pathlib import Path
import argparse, sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "_shared"))
from exports import ablation

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, default=ROOT / "input_ablation/figure/data")
p.add_argument("--output", type=Path, default=ROOT / "results/exports/input_ablation")
n = p.parse_args()
if (
    n.output.resolve().is_relative_to(ROOT / "input_ablation/figure/data")
    or n.output.resolve() == n.source.resolve()
):
    p.error("Use a separate export directory; bundled paper data are immutable")
ablation(n.source.resolve(), n.output.resolve())
print(n.output.resolve())
