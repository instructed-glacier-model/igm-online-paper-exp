#!/usr/bin/env python3
"""Re-evaluate the published selected architectures with seeds 0, 1 and 2."""
from pathlib import Path
import copy, sys, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_shared"))
from study import options, arguments

n = options(__doc__).parse_args()
co = yaml.safe_load(n.config.read_text())
from igm_online.native import run

for row in co["cases"]:
    if n.case and row["case"] != n.case:
        continue
    if n.architectures and row["architecture"] not in n.architectures.split(","):
        continue
    c = copy.deepcopy(co["configurations"][row["case"]])
    model = co["models"][row["model"]]
    c["online"] = model["online"]
    n.seed = row["seed"]
    a = arguments(n, c)
    d = a.output / a.namespace / c["case"] / "seed_reruns" / row["architecture"] / f"seed{a.seed}"
    run(a, c, None, row["architecture"], model["spec"], d)
