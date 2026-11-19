"""Apply the same declared geometry cleanup as the direct-reference driver."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "_shared"))
from igm_online.geometry import clean_geometry


def initialize(cfg, state):
    update(cfg, state)


def update(cfg, state):
    p = cfg.study.config.physics
    clean_geometry(
        cfg,
        state,
        float(p.get("minimum_ice_thickness", 0.01)),
        bool(p.get("remove_rigid_body_modes", True)) and "floating" in p.energy_components,
    )


def finalize(cfg, state):
    pass
