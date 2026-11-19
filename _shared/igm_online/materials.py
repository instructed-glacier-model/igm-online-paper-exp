"""Resolve input material fields identically for direct and native workflows."""
from __future__ import annotations

import numpy as np


def resolve_material_fields(state, physics, shape):
    """Prefer supplied tau_ref, otherwise convert supplied legacy slidingco.

    Case constants are fallbacks for fields absent from the input. Keep the
    arithmetic used by the direct energy adapter, including zero coefficients.
    """
    fields = dict(state)
    fields.setdefault("arrhenius", np.full(shape, physics["arrhenius"]))
    if "tau_ref" not in fields:
        legacy = fields.get("slidingco", np.full(shape, physics["slidingco"]))
        fields["tau_ref"] = np.asarray(legacy) * physics["u_ref"] ** (
            1 / physics["sliding_exponent"]
        )
    fields.setdefault("water_level", np.full(shape, physics.get("water_level", 0.0)))
    return {name: fields[name] for name in ("arrhenius", "tau_ref", "water_level")}


def assert_same_material_fields(actual, reference, config):
    """Reject accuracy comparisons between different physical input fields."""
    a = resolve_material_fields(actual, config["physics"], np.asarray(actual["thk"]).shape)
    b = resolve_material_fields(reference, config["physics"], np.asarray(reference["thk"]).shape)
    names = (
        ("arrhenius", "tau_ref", "water_level")
        if "floating" in config["physics"]["energy_components"]
        else ("arrhenius", "tau_ref")
    )
    for name in names:
        if np.asarray(a[name]).shape != np.asarray(b[name]).shape or not np.allclose(
            a[name], b[name], rtol=3e-6, atol=1e-10
        ):
            raise ValueError(
                f"Reference and model have different {name} fields; accuracy comparison is invalid"
            )
