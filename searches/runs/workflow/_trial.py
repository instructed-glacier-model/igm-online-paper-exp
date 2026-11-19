"""Private subprocess worker used only by 01_search.py."""
from pathlib import Path
import argparse, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_shared"))
from study import options, arguments

p = options(__doc__)
p.add_argument("--profile")
p.add_argument("--job", type=Path, required=True)
p.add_argument("--trial-output", type=Path, required=True)
n = p.parse_args()
j = json.loads(n.job.read_text())
c = j["config"]
c["online"] = j["online"]
n.seed = j["seed"]
a = arguments(n, c)
from igm_online.native import run

try:
    run(a, c, None, j["architecture"], j["spec"], n.trial_output)
except RuntimeError as e:
    if "occupied" in str(e).lower():
        print(e)
        sys.exit(3)
    raise
