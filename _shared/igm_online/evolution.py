"""Release transport and calving, with precision-preserving time."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf


class Evolution:
    def __init__(self, problem):
        import igm

        self.problem = problem
        self.config = problem.config
        cfg_dir = Path(igm.__file__).parent / "conf/processes"
        base = {
            "thk": OmegaConf.load(cfg_dir / "thk.yaml")["thk"],
            "calving_rate": OmegaConf.load(cfg_dir / "calving_rate.yaml")["calving_rate"],
        }
        self.cfg = OmegaConf.merge(problem.cfg, {"processes": base})
        t = self.config["time"]
        self.cfg.processes.thk.slope_type = t["slope_type"]
        self.cfg.processes.thk.ratio_density = (
            self.config["physics"]["ice_density"] / self.config["physics"]["water_density"]
        )
        self.cfg.processes.thk.remove_rigid_body_modes = (
            self.config["physics"].get("remove_rigid_body_modes", True)
            and "floating" in self.config["physics"]["energy_components"]
        )
        if t.get("marine"):
            self.cfg.processes.thk = OmegaConf.merge(self.cfg.processes.thk, t["marine"])
            self.cfg.processes.calving_rate.law = "water_depth"
            self.cfg.processes.calving_rate.water_depth.k = t["calving"]["k"]
            self.cfg.processes.calving_rate.max_rate = t["calving"]["c_max"]
        self.last_smb = None
        self.last_smb_time = -np.inf

    def object(self, state, velocity, dt):
        dtype = self.problem.dtype
        fields = {
            k: tf.convert_to_tensor(v, dtype)
            for k, v in state.items()
            if k not in ("uv", "velocity") and np.asarray(v).dtype.kind in "fib"
        }
        if "front_initial_extent" in fields:
            fields["front_initial_extent"] = tf.cast(fields["front_initial_extent"], tf.bool)
        fields["water_level"] = fields.get(
            "water_level", tf.ones_like(fields["thk"]) * tf.constant(-1e10, dtype)
        )
        obj = SimpleNamespace(**fields)
        obj.t = tf.convert_to_tensor(float(state["time"]), dtype)
        obj.it = 0
        obj.dt = tf.convert_to_tensor(dt, dtype)
        obj.ubar, obj.vbar = tf.convert_to_tensor(velocity, dtype)
        obj.dx = tf.convert_to_tensor(float(state["dx"]), dtype)
        return obj

    def smb(self, state):
        t = float(state["time"])
        if (
            self.last_smb is None
            or t - self.last_smb_time >= self.config["time"]["smb_update_frequency"] - 1e-10
        ):
            rows = np.asarray(self.config["time"]["smb"], np.float64)
            abl, acc, ela, cap = [np.interp(t, rows[:, 0], rows[:, i]) for i in range(1, 5)]
            surface = np.asarray(state["usurf"])
            result = np.clip((surface - ela) * np.where(surface < ela, abl, acc), -100.0, cap)
            if "icemask" in state:
                result = np.where((result < 0) | (state["icemask"] > 0.5), result, -10.0)
            self.last_smb = result.astype(np.asarray(state["thk"]).dtype)
            self.last_smb_time = t
        return self.last_smb

    def initialize_state(self, state):
        from igm.processes.thk import thk
        from .geometry import clean_geometry

        obj = self.object(state, np.zeros((2, *state["thk"].shape)), 0.0)
        thk.initialize(self.cfg, obj)
        p = self.config["physics"]
        clean_geometry(
            self.cfg,
            obj,
            p.get("minimum_ice_thickness", 0.01),
            p.get("remove_rigid_body_modes", True) and "floating" in p["energy_components"],
        )
        result = dict(state)
        for key in (
            "thk",
            "usurf",
            "lsurf",
            "Href",
            "ice_area_fraction",
            "front_initial_extent",
            "geometry_removed_volume_m3",
        ):
            if hasattr(obj, key):
                result[key] = np.asarray(getattr(obj, key))
        return result

    def advance(self, state, velocity, maximum_dt, exact=False):
        from igm.processes.time.time import compute_dt_from_cfl
        from igm.processes.thk import thk
        from igm.processes.calving_rate import calving_rate

        obj = self.object(state, velocity, maximum_dt)
        thk.initialize(self.cfg, obj)
        marine = self.config["case"] == "jakobshavn"
        if marine:
            calving_rate.initialize(self.cfg, obj)
            calving_rate.update(self.cfg, obj)
        dt = maximum_dt
        if not exact:
            kwargs = {"ablation_speed": obj.calving_rate + obj.frontal_melt_rate} if marine else {}
            dt = min(
                maximum_dt,
                float(
                    compute_dt_from_cfl(
                        obj.ubar,
                        obj.vbar,
                        self.config["time"]["cfl"],
                        obj.dx,
                        min(maximum_dt, self.config["time"]["step_max"]),
                        **kwargs,
                    )
                ),
            )
        if not np.isfinite(dt) or dt <= 1e-12:
            raise FloatingPointError(f"Unusable transient time step {dt}")
        obj.dt = tf.convert_to_tensor(dt, self.problem.dtype)
        obj.smb = tf.convert_to_tensor(self.smb(state), self.problem.dtype)
        thk.update(self.cfg, obj)
        from .geometry import clean_geometry

        p = self.config["physics"]
        clean_geometry(
            self.cfg,
            obj,
            p.get("minimum_ice_thickness", 0.01),
            p.get("remove_rigid_body_modes", True) and "floating" in p["energy_components"],
        )
        result = dict(state)
        for key in (
            "thk",
            "usurf",
            "lsurf",
            "Href",
            "ice_area_fraction",
            "calved_thk",
            "front_initial_extent",
            "geometry_removed_volume_m3",
        ):
            if hasattr(obj, key):
                result[key] = np.asarray(getattr(obj, key))
        result["time"] = np.asarray(float(state["time"]) + dt, dtype=np.float64)
        result.pop("uv", None)
        result.pop("velocity", None)
        return result
