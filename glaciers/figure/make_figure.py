#!/usr/bin/env python3
"""Redraw the manuscript overview entirely from the bundled map arrays."""
from pathlib import Path
import numpy as np
from glacier_overview import make_figure

HERE = Path(__file__).resolve().parent
maps = {}
for case in ["aletsch", "valais", "jakobshavn"]:
    with np.load(HERE / "data" / f"map_{case}.npz") as z:
        maps["jakob" if case == "jakobshavn" else case] = {
            k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files
        }
for key, value in maps.items():
    value["key"] = key
make_figure(str(HERE), maps)

import sys

sys.path.insert(0, str(HERE.parents[1] / "_shared"))
from tables import render

render(HERE)
