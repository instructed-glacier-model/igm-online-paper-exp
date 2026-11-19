#!/usr/bin/env python3
"""Replay recorded parameters into a fresh output directory, without sampling."""
from pathlib import Path
import copy, csv, sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "_shared"))
from study import options, arguments, resolved
from igm_online.config import architecture

p = options(__doc__)
p.add_argument("--sampler", choices=["uniform", "pareto"], required=True)
p.add_argument("--trials", default="0", help="Comma-separated attempt numbers")
n = p.parse_args()
c = resolved(n.config)
from igm_online.native import run

rows = list(csv.DictReader((ROOT / "searches/figure/data/trials.csv").open()))
for row in rows:
    if (
        row["case"] != c["case"]
        or row["sampler"] != n.sampler
        or row["trial"] not in n.trials.split(",")
    ):
        continue
    if n.architectures and row["architecture"] not in n.architectures.split(","):
        continue
    cc = copy.deepcopy(c)
    spec = architecture(row["architecture"])
    for key in spec["params"]:
        if row.get(key) not in (None, ""):
            import yaml

            spec["params"][key] = yaml.safe_load(row[key])
    for key in ["nbit", "nbit_init", "retrain_freq", "lr", "lr_init"]:
        if row.get(key) in (None, ""):
            raise ValueError("Recorded recipe is unavailable: " + str(row))
        cc["online"][key] = (
            int(float(row[key]))
            if key in ["nbit", "nbit_init", "retrain_freq"]
            else float(row[key])
        )
    n.seed = int(row["seed"])
    a = arguments(n, cc)
    d = (
        a.output
        / a.namespace
        / c["case"]
        / "replay"
        / n.sampler
        / row["architecture"]
        / row["trial"]
    )
    run(a, cc, None, row["architecture"], spec, d)
