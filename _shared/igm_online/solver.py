"""Reference convergence controller around the unmodified upstream Newton-CG."""
from __future__ import annotations
import time
import numpy as np
import tensorflow as tf

from .metrics import vector_rms
from .physics import active_dofs


class ReferenceFailure(RuntimeError):
    pass


def make_optimizer(problem, state, recipe=None):
    from igm.processes.iceflow.unified.mappings.identity import MappingIdentity
    from igm.processes.iceflow.unified.optimizers.cg_newton import OptimizerCGNewton
    from igm.processes.iceflow.unified.operators.factory import build_energy_operator

    cfg = dict(recipe or problem.config["solver"])
    zero = problem.zeros(state)
    mapping = MappingIdentity([], zero[0], zero[1], precision=problem.dtype.name)
    operator = build_energy_operator(
        cfg["operator"],
        "autodiff",
        "molho",
        problem.dtype.name,
        problem.cost_fn,
        mapping,
        probe_batch=cfg["probe_batch"] if cfg["operator"] == "banded" else 0,
    )

    class MeasuredCG(OptimizerCGNewton):
        """Observe native calls and work counters; do not alter any iterate."""

        def _cg_solve(self, *args, **kwargs):
            value = super()._cg_solve(*args, **kwargs)
            self.work.append(
                {"cg_iterations": int(value[1]), "cg_relative_residual": float(value[2])}
            )
            return value

    optimizer = MeasuredCG(
        problem.cost_fn,
        mapping,
        precision=problem.dtype.name,
        print_cost=False,
        line_search_method=cfg["line_search"],
        line_search_compile=False,
        iter_max=cfg["block_iterations"],
        damping=cfg["damping"],
        cg_max_iter=cfg["cg_max_iter"],
        cg_tol=cfg["cg_tol"],
        warm_start=cfg.get("warm_start", False),
        damping_adaptive=cfg["adaptive_damping"],
        damping_min=cfg.get("damping_min", 1e-12),
        operator=operator,
        preconditioner=cfg["preconditioner"],
    )
    if cfg.get("stopping", "strict") == "marine_velocity":
        # Use the pinned upstream individual-update criterion, without changing
        # the optimizer, energy or Hessian implementation.
        from igm.processes.iceflow.unified.halt import Halt
        from igm.processes.iceflow.unified.halt.criteria.abs_change import CriterionAbsChange
        from igm.processes.iceflow.unified.halt.metrics.u import MetricU

        criterion = CriterionAbsChange(
            MetricU(),
            dtype=problem.dtype.name,
            tol=cfg.get("velocity_tolerance", 1.0),
            reduction="rmse",
            consecutive=cfg.get("consecutive_updates", 3),
        )
        optimizer.halt = Halt(crit_success=[criterion], freq=1, dtype=problem.dtype.name)
    optimizer.work = []
    return optimizer


class DirectSolver:
    def __init__(self, problem, initial_state, recipe=None):
        self.problem = problem
        self.recipe = dict(recipe or problem.config["solver"])
        self.optimizer = make_optimizer(problem, initial_state, self.recipe)
        # Residual validation is external to the native solver. FP32 reductions
        # can have a cancellation floor on full rasters; test the same field in
        # the common FP64 physical functional instead of relaxing the threshold.
        from .physics import Problem

        self.validation_problem = (
            Problem(problem.config, "fp64") if problem.precision == "fp32" else problem
        )
        self.calls = 0

    def traces(self):
        optimizer = self.optimizer

        def count(value):
            return (
                value.experimental_get_tracing_count()
                if hasattr(value, "experimental_get_tracing_count")
                else None
            )

        return {
            "pcg": count(super(type(optimizer), optimizer)._cg_solve),
            "gradient": count(optimizer._get_grad_cg_newton),
            "mapping": count(optimizer.map.get_UV),
        }

    def solve(self, state, *, start=None, callback=None, fixed_iterations=None):
        p, opt, cfg = self.problem, self.optimizer, self.recipe
        marine = fixed_iterations is None and cfg.get("stopping", "strict") == "marine_velocity"
        inputs = p.tensor(state)
        validation = self.validation_problem
        validation_inputs = validation.tensor(state)
        initial = p.zeros(state) if start is None else tf.convert_to_tensor(start, p.dtype)
        opt.map.set_theta([initial[0], initial[1]])
        if opt._p_prev is not None:
            opt._p_prev.assign(tf.zeros_like(opt._p_prev))
        mask = np.asarray(state["thk"]) > p.config["physics"]["ice_threshold"]
        active = active_dofs(state)
        norm = lambda g: float(tf.linalg.norm(tf.boolean_mask(g, active)))
        initial_gradient = norm(
            validation.gradient(tf.cast(initial, tf.float64), validation_inputs)
        )
        before = self.traces()
        started = time.perf_counter()
        work_start = len(opt.work)
        history = []
        previous = initial
        converged = False
        budget = 1 if fixed_iterations is not None or marine else cfg["max_blocks"]
        for block in range(budget):
            if time.perf_counter() - started > cfg["timeout_seconds"]:
                break
            iterations = (
                cfg.get("max_newton_iterations", 200) if marine else cfg["block_iterations"]
            )
            opt.iter_max.assign(fixed_iterations if fixed_iterations is not None else iterations)
            t0 = time.perf_counter()
            saved_halt = opt.halt
            if fixed_iterations is not None:
                opt.halt = None
            try:
                costs = opt.minimize(inputs)
            finally:
                opt.halt = saved_halt
            uv = tf.stack(opt.map.get_theta())
            value = float(p.energy(uv, inputs))
            native_gnorm = norm(p.gradient(uv, inputs))
            gnorm = norm(validation.gradient(tf.cast(uv, tf.float64), validation_inputs))
            refinement = vector_rms(p.depth_average(uv - previous).numpy(), mask)
            relative = gnorm / max(initial_gradient, 1e-300)
            finite = bool(np.isfinite(value) and np.isfinite(gnorm) and np.isfinite(refinement))
            history.append(
                {
                    "block": block,
                    "energy": value,
                    "gradient_norm": gnorm,
                    "relative_gradient": relative,
                    "velocity_refinement_rms": refinement,
                    "native_gradient_norm": native_gnorm,
                    "validation_gradient_precision": "fp64",
                    "seconds": time.perf_counter() - t0,
                    "newton_iterations": int(tf.size(costs)),
                    "finite": finite,
                }
            )
            if marine:
                from igm.processes.iceflow.unified.halt import HaltStatus

                change = float(opt.halt_state.criterion_values[0])
                history[-1]["velocity_change_rms"] = change
                history[-1]["halt_status"] = int(opt.halt_state.status)
                finite = (
                    finite and bool(tf.reduce_all(tf.math.is_finite(uv))) and np.isfinite(change)
                )
                history[-1]["finite"] = finite
                converged = finite and int(opt.halt_state.status) == HaltStatus.SUCCESS.value
            if callback:
                callback(uv.numpy(), history)
            if not finite:
                break
            if marine:
                # Success requires three individual changes below tolerance.
                # A small gradient is diagnostic, not an additional veto.
                break
            # The first block cannot constitute a refinement of a converged solve.
            if (
                block > 0
                and refinement < cfg["refinement_rms"]
                and relative < cfg["relative_gradient"]
            ):
                converged = True
                break
            stagnant = cfg.get("stagnant_blocks", 3)
            if len(history) >= stagnant and all(
                h["velocity_refinement_rms"] < cfg.get("stagnation_rms", 1e-6)
                for h in history[-stagnant:]
            ):
                # An explicit failure exit, never a convergence criterion.
                break
            previous = tf.identity(uv)
        elapsed = time.perf_counter() - started
        self.calls += 1
        work = opt.work[work_start:]
        report = {
            "status": "converged"
            if converged
            else "fixed_budget"
            if fixed_iterations is not None
            else "unresolved",
            "precision": p.precision,
            "cold_start": start is None,
            "seconds": elapsed,
            "newton_iterations": len(work),
            "cg_iterations": sum(w["cg_iterations"] for w in work),
            "zero_cg_steps": sum(w["cg_iterations"] == 0 for w in work),
            "cg_work": work,
            "history": history,
            "traces_before": before,
            "traces_after": self.traces(),
            "first_block_including_compilation_seconds": history[0]["seconds"] if history else None,
            "subsequent_blocks_seconds": sum(h["seconds"] for h in history[1:]),
            "compilation_seconds": None,
            "compilation_note": "First-call inclusive timing; exclusive compilation requires a profiler",
            "initial_gradient_norm": initial_gradient,
            "recipe": cfg,
        }
        if history:
            report.update(
                velocity_refinement_rms=history[-1]["velocity_refinement_rms"],
                relative_gradient=history[-1]["relative_gradient"],
            )
        if marine:
            report.update(
                stopping_rule="marine_velocity",
                residual_is_diagnostic=True,
                convergence_definition="Whole-grid vector-DOF RMS change <= velocity_tolerance for consecutive individual Newton updates",
                velocity_tolerance=cfg.get("velocity_tolerance", 1.0),
                consecutive_updates=cfg.get("consecutive_updates", 3),
                velocity_change_rms=history[-1].get("velocity_change_rms") if history else None,
            )
        return uv.numpy(), report


def velocity_start(state, previous, recipe):
    """Reuse physical DOFs, independently of the native inner-CG warm start.

    Retain nodes adjacent to active Q1 cells, including ice-free corners. Truly
    inactive DOFs are zeroed; new active nodes inherit zero from the prior solve.
    """
    if not recipe.get("velocity_warm_start", False) or previous is None:
        return None
    previous = np.asarray(previous)
    active = active_dofs(state)
    if previous.shape != active.shape or not np.all(np.isfinite(previous)):
        raise ValueError("Previous velocity has incompatible shape or nonfinite DOFs")
    return np.where(active, previous, 0.0)


def verified_solve(solver, state, independent=False, callback=None, start=None):
    uv, info = solver.solve(state, start=start, callback=callback)
    if independent and info.get("history") and info["history"][-1]["finite"]:
        # Independent starting field, with the same energy and arithmetic.
        rng = np.random.default_rng(1729)
        start = rng.normal(scale=0.1, size=uv.shape).astype(uv.dtype)
        other, check = solver.solve(state, start=start, callback=callback)
        mask = np.asarray(state["thk"]) > 0.1
        discrepancy = vector_rms(
            solver.problem.depth_average(tf.constant(uv - other)).numpy(), mask
        )
        info["independent"] = {
            "status": check["status"],
            "velocity_rms": discrepancy,
            "solve": check,
        }
        primary_iterations = info["newton_iterations"]
        other_iterations = check["newton_iterations"]
        info["seconds"] += check["seconds"]
        info["newton_iterations"] += check["newton_iterations"]
        info["cg_iterations"] += check["cg_iterations"]
        threshold = solver.recipe.get("independent_rms", solver.recipe["refinement_rms"])
        info["independent"]["agreement_tolerance_m_per_year"] = threshold
        disagreement = (
            discrepancy > threshold
            if solver.recipe.get("stopping") == "marine_velocity"
            else discrepancy >= threshold
        )
        # A native small-update halt need not make two independent starts agree.
        # Refine both already-converged branches, with a declared bounded budget;
        # keep the original acceptance threshold and every attempted refinement.
        if info["status"] == check["status"] == "converged" and disagreement:
            attempts = info["independent"]["refinement"] = []
            cap = solver.recipe.get("max_newton_iterations", 200)
            for requested in solver.recipe.get("independent_refinement_steps", []):
                steps = min(requested, cap - primary_iterations, cap - other_iterations)
                if steps <= 0:
                    break
                a, primary = solver.solve(
                    state, start=uv, fixed_iterations=steps, callback=callback
                )
                b, secondary = solver.solve(
                    state, start=other, fixed_iterations=steps, callback=callback
                )
                for result in (primary, secondary):
                    for key in ("seconds", "newton_iterations", "cg_iterations"):
                        info[key] += result[key]
                primary_iterations += primary["newton_iterations"]
                other_iterations += secondary["newton_iterations"]
                candidate = vector_rms(
                    solver.problem.depth_average(tf.constant(a - b)).numpy(), mask
                )
                finite = all(
                    result["history"][-1]["finite"] for result in (primary, secondary)
                ) and np.isfinite(candidate)
                old_energies = (
                    (info["history"][-1]["energy"], check["history"][-1]["energy"])
                    if not attempts
                    else (
                        attempts[-1]["primary"]["history"][-1]["energy"],
                        attempts[-1]["secondary"]["history"][-1]["energy"],
                    )
                )
                decreasing = all(
                    result["history"][-1]["energy"]
                    <= old + 64 * np.finfo(a.dtype).eps * max(1.0, abs(old))
                    for result, old in zip((primary, secondary), old_energies)
                )
                attempts.append(
                    {
                        "steps_per_branch": steps,
                        "velocity_rms": candidate,
                        "finite": finite,
                        "energy_nonincreasing": decreasing,
                        "primary": primary,
                        "secondary": secondary,
                    }
                )
                if not finite or not decreasing:
                    break
                uv, other, discrepancy = a, b, candidate
                info["independent"]["velocity_rms"] = discrepancy
                disagreement = (
                    discrepancy > threshold
                    if solver.recipe.get("stopping") == "marine_velocity"
                    else discrepancy >= threshold
                )
                if not disagreement:
                    break
        if info["status"] == "converged" and (check["status"] != "converged" or disagreement):
            info["status"] = "independent_check_failed"
    return uv, info
