"""Small adapter to the release energy and discretization, preserving arithmetic."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf


class Problem:
    def __init__(self, config: dict, precision="fp64"):
        from igm.processes.iceflow.horizontal import HorizontalDiscrs
        from igm.processes.iceflow.vertical import VerticalDiscrs
        from igm.processes.iceflow.unified.utils import get_cost_fn

        p = config["physics"]
        self.config = config
        self.precision = precision
        self.dtype = tf.float64 if precision == "fp64" else tf.float32
        self.inputs = ["thk", "usurf", "arrhenius", "tau_ref", "dX"]
        if "floating" in p["energy_components"]:
            self.inputs += ["water_level"]
        explicit = OmegaConf.create(
            {
                "processes": {
                    "iceflow": {
                        "method": "unified",
                        "numerics": {
                            "precision": "double" if precision == "fp64" else "single",
                            "Nz": p["Nz"],
                            "basis_horizontal": p["horizontal"],
                            "basis_vertical": p["vertical"],
                            "vert_spacing": 4.0,
                        },
                        "physics": {
                            "energy_components": p["energy_components"],
                            "gravity_cst": p["gravity"],
                            "ice_density": p["ice_density"],
                            "water_density": p["water_density"],
                            "viscosity": {
                                "arrhenius": p["arrhenius"],
                                "enhancement_factor": 1.0,
                                "exponent": p["glen_exponent"],
                                "regularization": p["strain_regularization"],
                            },
                            "sliding": {
                                "law": "weertman",
                                "use_mask_gr": p.get("use_mask_gr", False),
                                "use_subgrid_gl": False,
                                "slidingco": p["slidingco"],
                                "tau_ref": p["slidingco"]
                                * p["u_ref"] ** (1 / p["sliding_exponent"]),
                                "exponent": p["sliding_exponent"],
                                "regularization": p["sliding_regularization"],
                                "u_ref": p["u_ref"],
                            },
                            "thr_ice_thk": p["ice_threshold"],
                            "min_sr": p["min_sr"],
                            "max_sr": p["max_sr"],
                            "force_negative_gravitational_energy": False,
                            "cf_eswn": p["cf_eswn"],
                        },
                        "unified": {"inputs": self.inputs},
                    }
                }
            }
        )
        import igm
        from pathlib import Path

        defaults = OmegaConf.load(Path(igm.__file__).parent / "conf/processes/iceflow.yaml")
        self.cfg = OmegaConf.merge({"processes": defaults}, explicit)
        self.horizontal = HorizontalDiscrs[p["horizontal"]](self.cfg)
        self.vertical = VerticalDiscrs[p["vertical"]](self.cfg)
        state = SimpleNamespace(
            iceflow=SimpleNamespace(discr_h=self.horizontal, discr_v=self.vertical)
        )
        self.cost_fn = get_cost_fn(self.cfg, state)
        self.energy = tf.function(self._energy, reduce_retracing=True)
        self.gradient = tf.function(self._gradient, reduce_retracing=True)

    def _energy(self, uv, inputs):
        return self.cost_fn(uv[0], uv[1], inputs)

    def _gradient(self, uv, inputs):
        with tf.GradientTape() as tape:
            tape.watch(uv)
            value = self._energy(uv, inputs)
        return tape.gradient(value, uv)

    def tensor(self, state):
        from .materials import resolve_material_fields

        p = self.config["physics"]
        shape = np.asarray(state["thk"]).shape
        fields = dict(state)
        fields.update(resolve_material_fields(fields, p, shape))
        fields["dX"] = np.full(shape, float(state["dx"]))
        return tf.convert_to_tensor(
            np.stack([fields[n] for n in self.inputs], axis=-1)[None], self.dtype
        )

    def zeros(self, state):
        return tf.zeros((2, 1, 2, *np.asarray(state["thk"]).shape), self.dtype)

    def depth_average(self, uv, thickness=None):
        # Release evaluator unconditionally casts to FP32. Apply the same V_bar
        # contraction directly so FP64 references retain their actual precision.
        value = tf.einsum("cbzyx,z->cbyx", uv, tf.cast(self.vertical.V_bar, uv.dtype))[:, 0]
        if thickness is not None:
            value = tf.where(tf.convert_to_tensor(thickness)[None] > 0, value, tf.zeros_like(value))
        return value


def active_dofs(state, nz=2):
    """Nodes adjacent to positive-thickness Q1 cells, including ice-free corners."""
    ice = np.asarray(state["thk"]) > 0.1
    cells = ice[:-1, :-1] | ice[:-1, 1:] | ice[1:, :-1] | ice[1:, 1:]
    active = np.zeros_like(ice)
    active[:-1, :-1] |= cells
    active[:-1, 1:] |= cells
    active[1:, :-1] |= cells
    active[1:, 1:] |= cells
    return np.broadcast_to(active, (2, 1, nz, *active.shape)).copy()
