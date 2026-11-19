"""Depth-averaged vector diagnostics; no failed cells are silently filtered."""
from __future__ import annotations
import numpy as np


def historical_ice_mask(config, reference_state, initial_mask):
    """Original Jakobshavn RMSE follows the reference's moving calving front."""
    if config["case"] == "jakobshavn":
        return np.asarray(reference_state.get("thk_true", reference_state["thk"])) > 0.5
    return initial_mask


def vector_rms(vector, mask):
    vector = np.asarray(vector, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(vector[:, mask] ** 2, axis=0), dtype=np.float64)))


def compare(
    velocity,
    reference,
    thickness,
    reference_thickness,
    *,
    speed_offset=10.0,
    threshold=5.0,
    ice_threshold=0.1,
    historical_mask=None,
    volume_thickness=None,
    reference_volume_thickness=None,
) -> dict:
    velocity, reference = np.asarray(velocity, np.float64), np.asarray(reference, np.float64)
    raw_speed = np.linalg.norm(velocity, axis=0)
    raw_reference_speed = np.linalg.norm(reference, axis=0)
    ice = np.asarray(thickness) > ice_threshold
    ref_ice = np.asarray(reference_thickness) > ice_threshold
    union = ice | ref_ice
    velocity = np.where(ice[None], velocity, 0.0)
    reference = np.where(ref_ice[None], reference, 0.0)
    bad = union & (~np.all(np.isfinite(velocity), axis=0) | ~np.all(np.isfinite(reference), axis=0))
    result = {
        "status": "ok",
        "ice_cells": int(union.sum()),
        "nonfinite_cells": int(bad.sum()),
        "missing_ice_cells": int((ref_ice & ~ice).sum()),
        "model_only_ice_cells": int((ice & ~ref_ice).sum()),
        "thickness_rmse": float(
            np.sqrt(np.mean((np.asarray(thickness, np.float64) - reference_thickness) ** 2))
        ),
        "volume_relative_change_percent": None,
        "speed_median_percent": None,
        "vector_l2_percent": None,
        "speed_p95_percent": None,
        "speed_rmse": None,
        "qualifies": False,
        "median_only_qualifies": False,
    }
    mass = np.asarray(thickness if volume_thickness is None else volume_thickness, np.float64)
    reference_mass = np.asarray(
        reference_thickness if reference_volume_thickness is None else reference_volume_thickness,
        np.float64,
    )
    result["mass_thickness_rmse"] = float(np.sqrt(np.mean((mass - reference_mass) ** 2)))
    volume = float(np.sum(reference_mass, dtype=np.float64))
    if volume > 0:
        result["volume_relative_change_percent"] = (
            100 * (float(np.sum(mass, dtype=np.float64)) - volume) / volume
        )
    if (
        not union.any()
        or bad.any()
        or not np.all(np.isfinite(thickness))
        or not np.all(np.isfinite(reference_thickness))
        or not np.all(np.isfinite(mass))
        or not np.all(np.isfinite(reference_mass))
    ):
        result["status"] = "nonfinite" if bad.any() else "invalid_or_empty_geometry"
        return result
    speed, speed_ref = np.linalg.norm(velocity, axis=0), np.linalg.norm(reference, axis=0)
    relative = 100.0 * np.abs(speed - speed_ref)[union] / (speed_ref[union] + speed_offset)
    result["speed_median_percent"] = float(np.median(relative))
    result["speed_p95_percent"] = float(np.percentile(relative, 95))
    norm = float(np.linalg.norm(reference[:, union]))
    if norm > 0:
        result["vector_l2_percent"] = (
            100.0 * float(np.linalg.norm((velocity - reference)[:, union])) / norm
        )
    mask = np.asarray(historical_mask, bool) if historical_mask is not None else union
    if mask.any():
        # The historical objective uses the unmodified depth-averaged speeds.
        # Union-mask zero filling belongs only to the newer geometry metrics.
        difference = (
            (raw_speed - raw_reference_speed)
            if historical_mask is not None
            else (speed - speed_ref)
        )
        if not np.isfinite(difference[mask]).all():
            result["status"] = "nonfinite_historical_velocity"
            return result
        result["speed_rmse"] = float(np.sqrt(np.mean(difference[mask] ** 2)))
    result["median_only_qualifies"] = result["speed_median_percent"] < threshold
    result["qualifies"] = (
        result["median_only_qualifies"]
        and result["vector_l2_percent"] is not None
        and result["vector_l2_percent"] < threshold
    )
    return result


def trajectory_summary(frames: list[dict]) -> dict:
    valid = bool(frames) and all(r.get("status") == "ok" for r in frames)

    def aggregate(key, fn):
        values = [r.get(key) for r in frames]
        return (
            float(fn(values))
            if values and all(v is not None and np.isfinite(v) for v in values)
            else None
        )

    return {
        "frames": len(frames),
        "valid": valid,
        "qualifies": valid and all(r["qualifies"] for r in frames),
        "median_only_qualifies": valid and all(r["median_only_qualifies"] for r in frames),
        "worst_speed_median_percent": aggregate("speed_median_percent", max),
        "worst_vector_l2_percent": aggregate("vector_l2_percent", max),
        "mean_speed_rmse": aggregate("speed_rmse", np.mean),
    }


def decomposition(
    anchor, transferred, corrected, ref_anchor, ref_target, mask, reference_uncertainty=0.0
) -> dict:
    values = [
        np.asarray(v, np.float64) for v in (anchor, transferred, corrected, ref_anchor, ref_target)
    ]
    anchor, transferred, corrected, ref_anchor, ref_target = values
    delta = ref_target - ref_anchor
    response = transferred - anchor
    initial_error = anchor - ref_anchor
    correction = corrected - transferred
    error = corrected - ref_target
    reconstructed = initial_error + response - delta + correction
    mask = np.asarray(mask, bool)
    norm_delta = vector_rms(delta, mask)
    a, b = response[:, mask].ravel(), delta[:, mask].ravel()
    dot = float(np.dot(a, b))
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    resolved = np.isfinite(norm_delta) and norm_delta > max(0.0, reference_uncertainty)
    return {
        "reconstruction_max_abs": float(np.max(np.abs(error[:, mask] - reconstructed[:, mask])))
        if mask.any()
        else None,
        "reference_change_rms": norm_delta,
        "input_response_rms": vector_rms(response, mask),
        "initial_error_rms": vector_rms(initial_error, mask),
        "final_error_rms": vector_rms(error, mask),
        "anticipation_ratio": dot / (norm_b**2) if resolved and norm_b > 0 else None,
        "anticipation_cosine": dot / (norm_a * norm_b)
        if resolved and norm_a > 0 and norm_b > 0
        else None,
        "ratio_status": "resolved" if resolved else "below_reference_uncertainty",
        "reference_uncertainty": reference_uncertainty,
    }
