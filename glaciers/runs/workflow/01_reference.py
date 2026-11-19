#!/usr/bin/env python3
"""Run one recorded FP64 reference or warm FP32 comparator recipe."""
from pathlib import Path
import sys, json

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_shared"))
from study import options, resolved, arguments, prepare

p = options(__doc__)
p.add_argument("--precision", choices=["fp32", "fp64"], required=True)
n = p.parse_args()
c = resolved(n.config)
a = arguments(n, c)
ctx = prepare(a, c)
from igm_online.experiments import simulate

result = simulate(a, c, ctx, precision=n.precision, reference=True)
print(json.dumps(result.get("reference_decision", {}), indent=2))
