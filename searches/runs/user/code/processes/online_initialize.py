"""Experiment bookkeeping before the ordinary IGM process chain."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "_shared"))
from igm_online.native import active


def initialize(cfg, state):
    active(cfg).initialize(cfg, state)


def update(cfg, state):
    active().begin_step(state)


def finalize(cfg, state):
    pass
