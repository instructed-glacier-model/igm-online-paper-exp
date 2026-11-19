#!/usr/bin/env python3
"""Summarize the 36 additional Aletsch correction-learning-rate cases."""
from pathlib import Path
import argparse, sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "_shared"))
from exports import rate_summary

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, default=ROOT / "input_ablation/figure/data")
p.add_argument("--output", type=Path, default=ROOT / "results/exports/input_ablation")
n = p.parse_args()
if n.source.resolve() == n.output.resolve():
    p.error("Use a separate output directory")
rate_summary(n.source, n.output)
