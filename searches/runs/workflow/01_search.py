#!/usr/bin/env python3
"""Fresh random/NSGA-II search; SQLite and trajectories go to --output only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_shared"))
from study import options, resolved, arguments

p = options(__doc__)
p.add_argument("--sampler", choices=["uniform", "pareto"], required=True)
p.add_argument("--trials", type=int, help="Override the paper budget for a smoke test")
n = p.parse_args()
c = resolved(n.config)
a = arguments(n, c)
from _optuna import sweep

sweep(a, c)
