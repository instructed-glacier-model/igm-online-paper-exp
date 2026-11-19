#!/usr/bin/env python3
"""Run the fixed published cohort; use filters for an individual case/model/transition."""
from pathlib import Path
import copy, sys, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_shared"))
from study import options, arguments, prepare

p = options(__doc__)
n = p.parse_args()
co = yaml.safe_load(n.config.read_text())
context = None
from igm_online.config import transitions

for row in co["cases"]:
    if n.case and row["case"] != n.case:
        continue
    if n.architectures and row["architecture"] not in n.architectures.split(","):
        continue
    c = copy.deepcopy(co["configurations"][row["case"]])
    if n.transitions and row["index"] not in transitions(n.transitions, c["transfer"]["count"]):
        continue
    n.seed = row["seed"]
    a = arguments(n, c)
    if context is None:
        context = prepare(a, c)
    from snapshots import run_case

    rates = [5e-5]
    run_case(a, c, context, row, co["models"][row["model"]], 10, rates, "transfer")
