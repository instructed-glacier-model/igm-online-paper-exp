"""Shared explicit marine geometry hygiene for direct and native neural paths."""
import tensorflow as tf


def clean_geometry(cfg, state, minimum_thickness=0.01, remove_rigid=True):
    from igm.processes.thk.rigid_body import remove_rigid_body_modes
    from igm.processes.thk.surfaces import get_density_ratio, update_surfaces

    thin = (state.thk > 0.0) & (state.thk < minimum_thickness)
    removed = tf.where(thin, state.thk, tf.zeros_like(state.thk))
    state.thk = tf.where(thin, tf.zeros_like(state.thk), state.thk)
    components = getattr(state, "thk_components", None)
    front = (
        components.front is not None
        if components
        else cfg.processes.thk.front.method not in (None, "none", "")
    )
    native_cleaned = components is not None and components.remove_rigid_body_modes
    if remove_rigid and (not native_cleaned or bool(tf.reduce_any(thin))):
        removed += remove_rigid_body_modes(state, get_density_ratio(cfg), front=front)
    if front and hasattr(state, "calved_thk"):
        state.calved_thk = state.calved_thk + removed
    volume = tf.reduce_sum(tf.cast(removed, tf.float64)) * tf.cast(state.dx, tf.float64) ** 2
    previous = getattr(state, "geometry_removed_volume_m3", tf.constant(0.0, tf.float64))
    state.geometry_removed_volume_m3 = tf.cast(previous, tf.float64) + volume
    update_surfaces(cfg, state)
    return removed
