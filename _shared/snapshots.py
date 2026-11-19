"""Published fixed-state experiments: anchor Adam, then independent fresh Adam.

No GD, NTK or spectral experiments are included. Checkpoints contain weights,
normalization and optimizer variables; branches restore the immutable anchor.
"""
import copy, json, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from igm_online.artifacts import Run, atomic_json, atomic_npz, checkpoint_weights, restore_weights
from igm_online.data import load_state
from igm_online.experiments import base_directory, scientific
from igm_online.transitions import generate_transition
from igm_online.models import build, forward, count, optimizer, training_step
from igm_online.physics import Problem
from igm_online.metrics import compare, decomposition, vector_rms
from igm_online.runtime import file_hash


def run_case(args, config, context, row, model_info, steps, rates, study):
    source = generate_transition(args, config, context, row["index"])
    anchor, target = load_state(source / "anchor.npz"), load_state(source / "target.npz")
    reference = json.loads((source / "result.json").read_text())
    problem = Problem(config, "fp32")
    pa, pb = problem.tensor(anchor), problem.tensor(target)
    model = build(model_info["spec"], pa, problem.inputs, args.seed + 1009 * row["index"])
    d = (
        base_directory(args, config)
        / study
        / row["architecture"]
        / f"seed{args.seed}"
        / f"{row['index']:03d}"
    )
    anchor_dir = d / "anchor"
    sig = scientific(
        config,
        context,
        workflow="anchor",
        seed=args.seed,
        index=row["index"],
        spec=model_info["spec"],
        anchor_sha256=file_hash(source / "anchor.npz"),
        target_sha256=file_hash(source / "target.npz"),
    )
    with Run(anchor_dir, sig, context, args.resume, args.retry_failed) as run:
        saved = anchor_dir / "checkpoint.npz"
        opt = optimizer(config, config["transfer"]["anchor_lr"])
        opt.build(model.trainable_variables)
        done = 0
        if saved.exists() and args.resume:
            extra = restore_weights(saved, model)
            done = int(extra["steps"])
            for i, v in enumerate(opt.variables):
                v.assign(extra[f"optimizer_{i:04d}"])
        if not run.skip:

            @tf.function
            def train():
                return training_step(problem, model, pa, opt=opt)

            for k in range(done, config["transfer"]["anchor_steps"]):
                loss = float(train())
                if not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite anchor loss")
                if (k + 1) % 1000 == 0 or k + 1 == config["transfer"]["anchor_steps"]:
                    checkpoint_weights(
                        saved,
                        model,
                        steps=np.asarray(k + 1),
                        **{f"optimizer_{i:04d}": v.numpy() for i, v in enumerate(opt.variables)},
                    )
                    run.checkpoint(anchor_steps=k + 1)
            checkpoint_weights(anchor_dir / "weights.npz", model)
            atomic_json(
                anchor_dir / "result.json",
                dict(
                    status="complete",
                    parameters=count(model),
                    steps=config["transfer"]["anchor_steps"],
                ),
            )
        restore_weights(anchor_dir / "weights.npz", model)
    # Preserve weights, including the anchor normalization, across all branches.
    weights = model.get_weights()
    ua = forward(model, pa).numpy()
    ut = forward(model, pb).numpy()
    va = problem.depth_average(tf.constant(ua)).numpy()
    vt = problem.depth_average(tf.constant(ut)).numpy()
    mask = target.get("thk_true", target["thk"]) > 0.1
    p64 = Problem(config, "fp64")
    xa64, xb64 = p64.tensor(anchor), p64.tensor(target)
    ea = float(p64.energy(tf.constant(anchor["uv"], tf.float64), xa64))
    eb = float(p64.energy(tf.constant(target["uv"], tf.float64), xb64))

    def energy(uv, inputs):
        return float(p64.energy(tf.constant(uv, tf.float64), inputs))

    es = energy(ua, xa64)
    et = energy(ut, xb64)
    for lr in rates:
        model.set_weights(weights)
        branch = d / f"adam_{lr:.0e}"
        signature = scientific(
            config,
            context,
            workflow=study,
            seed=args.seed,
            index=row["index"],
            architecture=row["architecture"],
            spec=model_info["spec"],
            anchor_weights_sha256=file_hash(anchor_dir / "weights.npz"),
            correction_steps=steps,
            correction_lr=float(lr),
            fresh_adam=True,
        )
        with Run(branch, signature, context, args.resume, args.retry_failed) as run:
            if run.skip:
                continue
            opt = optimizer(config, float(lr))
            opt.build(model.trainable_variables)
            curves = []
            done = 0
            cp = branch / "checkpoint.npz"
            if args.resume and cp.exists():
                extra = restore_weights(cp, model)
                done = int(extra["steps"])
                for i, v in enumerate(opt.variables):
                    v.assign(extra[f"optimizer_{i:04d}"])
                curves = (
                    json.loads(str(extra["curves_json"]))
                    if "curves_json" in extra
                    else json.loads((branch / "progress.json").read_text())
                )

            def measure(k):
                uv = forward(model, pb)
                v = problem.depth_average(uv).numpy()
                return dict(
                    step=k,
                    vector_rms=vector_rms(v - target["velocity"], mask),
                    **compare(v, target["velocity"], target["thk"], target["thk"]),
                )

            if not curves:
                curves = [measure(0)]

            @tf.function
            def update():
                return training_step(problem, model, pb, opt=opt)

            for k in range(done, steps):
                update()
                curves.append(measure(k + 1))
                checkpoint_weights(
                    cp,
                    model,
                    steps=np.asarray(k + 1),
                    curves_json=np.asarray(json.dumps(curves)),
                    **{f"optimizer_{i:04d}": v.numpy() for i, v in enumerate(opt.variables)},
                )
                atomic_json(branch / "progress.json", curves)
                run.checkpoint(correction_steps=k + 1)
                if curves[-1]["status"] != "ok":
                    raise FloatingPointError("Nonfinite correction retained")
            u = forward(model, pb).numpy()
            v = problem.depth_average(tf.constant(u)).numpy()
            eo = energy(u, xb64)
            dec = decomposition(
                va,
                vt,
                v,
                anchor["velocity"],
                target["velocity"],
                mask,
                reference["reference_uncertainty"],
            )
            held = vector_rms(va - target["velocity"], mask)
            trans = vector_rms(vt - target["velocity"], mask)
            floor = 64 * np.finfo(float).eps * max(1.0, abs(eb))
            resolved = et - eb > floor
            result = dict(
                case=config["case"],
                architecture=row["architecture"],
                seed=args.seed,
                index=row["index"],
                status="complete",
                parameters=count(model),
                correction_lr=float(lr),
                completed_steps=steps,
                curves=curves,
                decomposition=dec,
                held_error_rms=held,
                transferred_error_rms=trans,
                correction_factor=curves[-1]["vector_rms"] / trans if trans else None,
                transfer_ratio=trans / held if held else None,
                anchor_error_rms=dec["initial_error_rms"],
                input_change_benefit_percent=100 * (1 - trans / held) if held else None,
                G_s=(es - ea) / abs(ea),
                G_t=(et - eb) / abs(eb),
                G_o_adam=(eo - eb) / abs(eb),
                gamma_adam=(eo - eb) / (et - eb) if resolved else None,
                ratio_status="positive_above_roundoff"
                if resolved
                else "undefined_nonpositive_or_roundoff",
                ratio_energy_floor=floor,
                reference_uncertainty_included=False,
                adam_energy_increased=eo > et,
                normalization="Anchor mean and variance frozen; included in checkpoints",
                selection=model_info.get("selection"),
            )
            atomic_npz(
                branch / "velocities.npz",
                uv_anchor=ua,
                uv_transferred=ut,
                uv_adam=u,
                velocity_anchor=va,
                velocity_transferred=vt,
                velocity_adam=v,
            )
            atomic_json(branch / "result.json", result)
