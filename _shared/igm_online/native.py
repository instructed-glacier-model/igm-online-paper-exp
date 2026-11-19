"""Native igm_run trajectories with observation-only user modules.

IGM owns SMB, time stepping, thickness transport, optimization and evaluation.
The hooks below own provenance, restart state, output frames and accuracy metrics.
"""
from __future__ import annotations

import atexit
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import os
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import yaml

from .artifacts import Run, atomic_json, checkpoint_weights, restore_weights
from .data import load_input, load_state, save_state, frame_key
from .metrics import compare, historical_ice_mask, trajectory_summary
from .materials import assert_same_material_fields
from .runtime import root

_ACTIVE = None


def initialize_material_fields(state, config):
    """Convert legacy inputs before native unified initialization fills defaults."""
    import tensorflow as tf
    from .materials import resolve_material_fields

    supplied = {
        name: np.asarray(getattr(state, name))
        for name in ("arrhenius", "tau_ref", "slidingco", "water_level")
        if hasattr(state, name)
    }
    fields = resolve_material_fields(supplied, config["physics"], tuple(state.thk.shape))
    for name, value in fields.items():
        if (
            name == "water_level"
            and name not in supplied
            and "floating" not in config["physics"]["energy_components"]
        ):
            continue  # Preserve upstream's no-ocean default for grounded cases.
        setattr(state, name, tf.convert_to_tensor(value, dtype=state.thk.dtype))


def configuration(args, config, name, spec, directory):
    """Return ordinary IGM YAML; no numerical implementation lives here."""
    p, t, online = config["physics"], config["time"], config["online"]
    end = t["start"] + t["duration"] if "duration" in t else t["end"]
    params = copy.deepcopy(spec["params"])
    family = spec["family"]
    if family == "dahunet":
        params.update(backend=spec["backend"], features=spec["features"])
    if spec.get("scramble"):
        family = "online_scrambled"
    if family == "controlled_backbone":
        family = "online_backbone"
        params["features"] = spec["features"]
    dtype = "single"
    fields = ["thk", "usurf", "arrhenius", "tau_ref", "dX"]
    if "floating" in p["energy_components"]:
        fields.append("water_level")
    physics = dict(
        energy_components=p["energy_components"],
        gravity_cst=p["gravity"],
        ice_density=p["ice_density"],
        water_density=p["water_density"],
        thr_ice_thk=p["ice_threshold"],
        min_sr=p["min_sr"],
        max_sr=p["max_sr"],
        cf_eswn=p["cf_eswn"],
        viscosity=dict(
            arrhenius=p["arrhenius"],
            enhancement_factor=1.0,
            exponent=p["glen_exponent"],
            regularization=p["strain_regularization"],
        ),
        sliding=dict(
            law="weertman",
            use_mask_gr=p.get("use_mask_gr", False),
            use_subgrid_gl=False,
            slidingco=p["slidingco"],
            tau_ref=p["slidingco"] * p["u_ref"] ** (1 / p["sliding_exponent"]),
            exponent=p["sliding_exponent"],
            regularization=p["sliding_regularization"],
            u_ref=p["u_ref"],
        ),
    )
    crop = {}
    if config.get("crop"):
        initial = load_input(config, "fp32")
        pad = float(initial["dx"]) * 0.25
        crop = dict(
            xmin=float(initial["x"][0]) - pad,
            xmax=float(initial["x"][-1]) + pad,
            ymin=float(initial["y"][0]) - pad,
            ymax=float(initial["y"][-1]) + pad,
        )
    arguments = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("job", "trial_output")
    }
    arguments.setdefault("gpu", "cpu")
    arguments.setdefault("case", config["case"])
    arguments.setdefault("profile", config["profile"])
    result = dict(
        defaults=[
            {"override /inputs": ["local"]},
            {
                "override /processes": [
                    "online_initialize",
                    "smb",
                    "time",
                    "thk",
                    "online_geometry",
                    "iceflow",
                    "online_save",
                ]
            },
            {"override /outputs": []},
        ],
        core=dict(
            check_compat_params=False,
            check_module_needs=False,
            print_info=False,
            print_params=False,
            hardware=dict(visible_gpus=[0]),
        ),
        hydra=dict(
            run=dict(dir=str(directory / "engine" / "${now:%Y%m%d_%H%M%S_%f}")),
            job=dict(chdir=True),
        ),
        inputs=dict(
            local=dict(filename=str(root() / config["input"]), coarsening=dict(ratio=1), crop=crop)
        ),
        processes=dict(
            smb=dict(
                method="simple",
                simple=dict(
                    update_freq=t["smb_update_frequency"],
                    array=[["time", "gradabl", "gradacc", "ela", "accmax"], *t["smb"]],
                ),
            ),
            time=dict(
                start=t["start"],
                end=end,
                save=t["save"],
                cfl=t["cfl"],
                cfl_percentile=t["cfl_percentile"],
                step_max=t["step_max"],
            ),
            thk=dict(
                slope_type=t["slope_type"],
                remove_rigid_body_modes=p.get("remove_rigid_body_modes", True)
                and "floating" in p["energy_components"],
                ratio_density=p["ice_density"] / p["water_density"],
            ),
            iceflow=dict(
                method="unified",
                physics=physics,
                numerics=dict(
                    precision=dtype,
                    Nz=p["Nz"],
                    basis_horizontal=p["horizontal"],
                    basis_vertical=p["vertical"],
                    vert_spacing=4.0,
                ),
                unified=dict(
                    mapping="network",
                    optimizer="adam",
                    inputs=fields,
                    bcs=p["imposed_bcs"],
                    nbit_init=online["nbit_init"],
                    nbit=online["nbit"],
                    retrain_freq=online["retrain_freq"],
                    nbit_warmup=-1,
                    init_stop=dict(enabled=False),
                    adaptive_time=dict(method="none"),
                    adaptive_patching=dict(enabled=False),
                    network=dict(
                        architecture=family,
                        pretrained=False,
                        fp16_inference=False,
                        seed=args.seed,
                        params="${study.network_params}",
                    ),
                    adam=dict(
                        lr=online["lr"],
                        lr_init=online["lr_init"],
                        optimizer_clipnorm=online["clipnorm"],
                    ),
                    halt=dict(success=[], failure=[]),
                    display=dict(print_cost=False),
                    data_preparation=dict(framesizemax=100000000, patches_per_batch=1),
                ),
            ),
        ),
        study=dict(
            workflow="online",
            args=arguments,
            config=config,
            architecture=name,
            architecture_spec=spec,
            directory=str(directory),
            network_params=params,
        ),
    )
    if config["case"] == "jakobshavn":
        result["defaults"][1]["override /processes"].insert(2, "calving_rate")
        result["inputs"]["local"]["water_level"] = dict(
            include=True, value=p.get("water_level", 0.0)
        )
        result["processes"]["thk"].update(copy.deepcopy(t.get("marine", {})))
        result["processes"]["calving_rate"] = dict(
            law="water_depth", water_depth=dict(k=t["calving"]["k"]), max_rate=t["calving"]["c_max"]
        )
    return result


class Observer:
    def __init__(self, args, config, context, name, spec, directory):
        from .experiments import scientific, require_reference, save_times

        self.args, self.config, self.context = args, config, context
        self.name, self.spec, self.directory = name, spec, Path(directory)
        self.deferred = getattr(args, "defer_scoring", False)
        self.canonical = None if self.deferred else require_reference(args, config)
        self.signature = scientific(
            config,
            context,
            workflow="online",
            precision="fp32",
            architecture=name,
            architecture_spec=spec,
            seed=args.seed,
            execution_backend="igm_run",
        )
        self.signature["native_hooks_sha256"] = context.get("native_hooks_sha256")
        if self.deferred:
            self.signature["scoring_deferred"] = True
        if self.canonical:
            self.signature["reference_signature"] = json.loads(
                (self.canonical / "manifest.json").read_text()
            )["signature"]
        self.run = Run(self.directory, self.signature, context, args.resume, args.retry_failed)
        self.started = time.perf_counter()
        self.times = save_times(config)
        self.progress = dict(
            steps=0,
            frames=[],
            solves=[],
            steady_seconds=0.0,
            initialization_seconds=0.0,
            checkpoint_seconds=0.0,
            checkpoint_count=0,
            frame_io_seconds=0.0,
        )
        self.checkpoint = None
        if args.resume and (self.directory / "checkpoint.npz").exists():
            self.checkpoint = load_state(self.directory / "checkpoint.npz")
            self.progress = json.loads((self.directory / "progress.json").read_text())
        self.last_checkpoint = self.started
        self.first_step = True
        self.result = None
        self.manual = False
        self.closed = False

    def initialize(self, cfg, state):
        import tensorflow as tf
        from igm.processes.iceflow.emulate.utils.architectures import Architectures
        from .models import ControlledBackbone, ScrambledDahuNet

        tf.config.experimental.enable_tensor_float_32_execution(False)

        # Native architecture extension points: only the declared scientific controls.
        def scrambled(*, input_names, Nz, network_params):
            model = ScrambledDahuNet(
                input_names=input_names,
                params=network_params,
                seed=self.spec.get("scramble_seed", 42),
                dtype="float32",
            )
            return model

        def backbone(*, input_names, Nz, network_params):
            params = dict(network_params)
            model = ControlledBackbone(
                input_names=input_names,
                params=params,
                features=params.pop("features"),
                dtype="float32",
            )
            return model

        Architectures.setdefault("online_scrambled", scrambled)
        Architectures.setdefault("online_backbone", backbone)
        if self.checkpoint is not None:
            for key, value in self.checkpoint.items():
                if key not in ("uv", "velocity", "time", "_solved", "online_smb_time"):
                    setattr(state, key, tf.convert_to_tensor(value))
            cfg.processes.time.start = float(self.checkpoint["time"])
            # Construct native optimizer variables cheaply, then restore them exactly.
            cfg.processes.iceflow.unified.nbit_init = 1
        # The native unified initializer reads tau_ref only. Convert legacy
        # input slidingco explicitly, matching the reference adapter; otherwise
        # it silently substitutes the case scalar and changes the physical model.
        initialize_material_fields(state, self.config)
        atomic_json(
            self.run.attempt / "native_config.json",
            __import__("omegaconf").OmegaConf.to_container(cfg, resolve=True),
        )

    def begin_step(self, state):
        if self.first_step:
            # The initial state is saved during initialization; count actual advances.
            state.it = self.progress["steps"] + 1
            self.first_step = False
        self.step_started = time.perf_counter()

    def initialized(self, cfg, state):
        import tensorflow as tf
        from igm.processes.iceflow.unified.evaluator import evaluate_iceflow

        actual = {
            name: np.asarray(getattr(state, name))
            for name in ("thk", "arrhenius", "tau_ref", "water_level")
            if hasattr(state, name)
        }
        assert_same_material_fields(actual, load_input(self.config, "fp32"), self.config)
        import hashlib

        atomic_json(
            self.run.attempt / "effective_material_fields.json",
            {
                name: dict(
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    minimum=float(value.min()),
                    maximum=float(value.max()),
                    sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                )
                for name, value in actual.items()
                if name != "thk"
            },
        )
        self.mask = (
            load_input(self.config, "fp32")["thk"]
            > self.config["metrics"]["historical_ice_threshold"]
        )
        if self.checkpoint is not None:
            extra = restore_weights(self.directory / "model_checkpoint.npz", state.iceflow_model)
            for index, variable in enumerate(state.iceflow.optimizer.optim_adam.variables):
                variable.assign(extra[f"optimizer_{index:04d}"])
            if "online_smb_time" in self.checkpoint:
                state.tlast_mb.assign(float(self.checkpoint["online_smb_time"]))
            if "smb" in self.checkpoint:
                state.smb = tf.convert_to_tensor(self.checkpoint["smb"], state.thk.dtype)
            evaluate_iceflow(cfg, state)
            np.asarray(state.ubar)  # include completed restore/evaluation work
            self.progress["initialization_seconds"] += time.perf_counter() - self.started
        state.time_save = tf.constant(self.times, state.t.dtype)
        state.itsave = max(0, len(self.progress["frames"]) - 1)
        if not self.progress["frames"]:
            self.observe(cfg, state, initial=True)

    def state_arrays(self, state):
        keys = [
            "x",
            "y",
            "dx",
            "topg",
            "usurf",
            "thk",
            "icemask",
            "arrhenius",
            "tau_ref",
            "water_level",
            "Href",
            "ice_area_fraction",
            "smb",
            "lsurf",
            "front_initial_extent",
        ]
        result = {key: np.asarray(getattr(state, key)) for key in keys if hasattr(state, key)}
        result["time"] = np.asarray(float(state.t), dtype=np.float64)
        return result

    def observe(self, cfg, state, initial=False):
        import tensorflow as tf

        arrays = self.state_arrays(state)
        velocity = np.stack([np.asarray(state.ubar), np.asarray(state.vbar)])
        if not np.isfinite(velocity).all():
            raise FloatingPointError("Nonfinite native velocity")
        now = time.perf_counter()
        if initial:
            self.progress["initialization_seconds"] = now - self.started
        else:
            self.progress["steady_seconds"] += now - self.step_started
            self.progress["steps"] = int(state.it)
        nbit = (
            self.config["online"]["nbit_init"]
            if initial
            else self.config["online"]["nbit"]
            if int(state.it) % self.config["online"]["retrain_freq"] == 0
            else 0
        )
        self.progress["solves"].append(
            dict(
                time=float(state.t),
                gradient_steps=nbit,
                seconds=now - (self.started if initial else self.step_started),
                phase="native_step",
            )
        )
        index = len(self.progress["frames"])
        saved = initial or bool(state.saveresult)
        checkpoint_due = (
            saved
            or self.run.stopped
            or now - self.last_checkpoint >= self.config["runtime"]["checkpoint_interval_seconds"]
        )
        if not checkpoint_due:
            return
        inputs = tf.stack(
            [getattr(state, name) for name in cfg.processes.iceflow.unified.inputs], -1
        )[None]
        uv = np.stack([np.asarray(value) for value in state.iceflow.mapping.get_UV(inputs)])
        if saved and index < len(self.times):
            target = self.times[index]
            # Native FP32 clocks reach the representable value of each requested time.
            tolerance = max(1e-7, abs(float(np.spacing(np.float32(target)))) * 2)
            if abs(float(state.t) - target) > tolerance:
                raise RuntimeError(
                    f"Native frame {float(state.t)} differs from requested time {target}"
                )
            tick = time.perf_counter()
            save_state(self.directory / "frames" / f"{frame_key(target)}.npz", arrays, uv, velocity)
            self.progress["frame_io_seconds"] += time.perf_counter() - tick
            row = dict(time=target, state_time=float(state.t), status="awaiting_reference")
            if self.canonical:
                target_state = load_state(self.canonical / "frames" / f"{frame_key(target)}.npz")
                if initial:
                    assert_same_material_fields(arrays, target_state, self.config)
                row.update(
                    compare(
                        velocity,
                        target_state["velocity"],
                        arrays["thk"],
                        target_state["thk"],
                        speed_offset=self.config["metrics"]["speed_offset"],
                        threshold=self.config["metrics"]["threshold_percent"],
                        historical_mask=historical_ice_mask(self.config, target_state, self.mask),
                        volume_thickness=arrays["thk"] + arrays.get("Href", 0.0),
                        reference_volume_thickness=target_state["thk"]
                        + target_state.get("Href", 0.0),
                    )
                )
            self.progress["frames"].append(row)
        tick = time.perf_counter()
        save_state(
            self.directory / "checkpoint.npz",
            arrays,
            uv,
            velocity,
            compressed=False,
            _solved=np.asarray(True),
            online_smb_time=np.asarray(float(state.tlast_mb)),
        )
        checkpoint_weights(
            self.directory / "model_checkpoint.npz",
            state.iceflow_model,
            compressed=False,
            **{
                f"optimizer_{i:04d}": np.asarray(v)
                for i, v in enumerate(state.iceflow.optimizer.optim_adam.variables)
            },
        )
        self.progress["checkpoint_seconds"] += time.perf_counter() - tick
        self.progress["checkpoint_count"] += 1
        atomic_json(self.directory / "progress.json", self.progress)
        self.last_checkpoint = time.perf_counter()
        self.run.checkpoint(time=float(state.t), steps=self.progress["steps"])

    def finish(self, state):
        from igm.processes.iceflow.unified.evaluator.evaluator import evaluator_iceflow
        from .models import count

        if len(self.progress["frames"]) != len(self.times):
            raise RuntimeError("Native trajectory did not save the complete frame grid")
        elapsed = time.perf_counter() - self.started
        previous = sum(
            json.loads(p.read_text()).get("elapsed_seconds", 0.0)
            for p in (self.directory / "attempts").glob("*/status.json")
            if p.parent != self.run.attempt
        )
        self.result = dict(
            workflow="online",
            execution_backend="igm_run",
            validated=False,
            case=self.config["case"],
            profile=self.config["profile"],
            architecture=self.name,
            precision="fp32",
            seed=self.args.seed,
            validation_only=self.config.get("validation_only", False),
            parameters=count(state.iceflow_model),
            total_seconds=self.context["startup_seconds"] + elapsed + previous,
            attempt_wall_seconds=elapsed,
            previous_attempt_wall_seconds=previous,
            resumed=previous > 0,
            startup_seconds=self.context["startup_seconds"],
            computation_seconds=self.progress["initialization_seconds"]
            + self.progress["steady_seconds"],
            **{
                k: self.progress[k]
                for k in [
                    "initialization_seconds",
                    "steady_seconds",
                    "checkpoint_seconds",
                    "checkpoint_count",
                    "frame_io_seconds",
                    "steps",
                    "frames",
                ]
            },
            newton_iterations=0,
            cg_iterations=0,
            device=self.context["device"],
            packages=self.context["packages"],
            source_commit=self.context["dependency"]["commit"],
            evaluator=dict(
                implementation="IGM evaluator_iceflow",
                mapping_traces=state.iceflow.mapping.get_UV.experimental_get_tracing_count(),
                evaluator_traces=evaluator_iceflow.experimental_get_tracing_count(),
            ),
            trajectory=None if self.deferred else trajectory_summary(self.progress["frames"]),
            scoring_status="pending" if self.deferred else "complete",
        )
        atomic_json(self.directory / "result.json", self.result)
        if self.result["trajectory"] is not None:
            state.score = dict(
                cost_accuracy=self.result["trajectory"]["mean_speed_rmse"],
                cost_time=self.result["steady_seconds"],
            )
        if self.manual:
            self.run.__exit__(None, None, None)
            self.closed = True

    def fail(self, error):
        if self.manual and not self.closed:
            self.run.__exit__(type(error), error, error.__traceback__)
            self.closed = True


def run(args, config, context, name, spec, directory):
    """Run an ordinary igm_run command in a fresh process for each trajectory."""
    directory = Path(directory)
    status_path = directory / "status.json"
    previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    case = directory / "native"
    (case / "experiment").mkdir(parents=True, exist_ok=True)
    if not (case / "processes").exists():
        (case / "processes").symlink_to(
            root() / "searches/runs/processes", target_is_directory=True
        )
    if not (case / "user").exists():
        (case / "user").symlink_to(root() / "searches/runs/user", target_is_directory=True)
    gpu_lock = None
    physical_gpu = str(args.gpu)
    if physical_gpu != "cpu":
        import fcntl

        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + physical_gpu,
                "--query-gpu=uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        uuid, memory, utilization = [v.strip() for v in raw.split(",")]
        gpu_lock = open("/tmp/igm-paper-" + uuid + ".lock", "a")
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("GPU occupied by another paper worker") from None
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
        )
        if uuid in apps or float(memory) > 300 or float(utilization) > 5:
            gpu_lock.close()
            raise RuntimeError("GPU occupied; retry when idle")
        physical_gpu = uuid
    native_config = configuration(args, config, name, spec, directory)
    native_config["study"].update(
        gpu=str(getattr(args, "gpu", "cpu")),
        seed=args.seed,
        namespace=args.namespace,
        case=config["case"],
        profile=config["profile"],
        output=str(args.output),
    )
    (case / "experiment/params.yaml").write_text(
        "# @package _global_\n" + yaml.safe_dump(native_config, sort_keys=False)
    )
    from .runtime import activate

    activate()
    import sysconfig

    entry = Path(sysconfig.get_path("scripts")) / "igm_run"
    if not entry.exists():
        raise RuntimeError("igm_run is missing from this Python environment")
    command = [sys.executable, str(entry), "+experiment=params"]
    env = dict(os.environ, PYTHONPATH=str(root() / "_shared"))
    env["CUDA_VISIBLE_DEVICES"] = "-1" if args.gpu == "cpu" else physical_gpu
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=case, env=env)
    handlers = {}

    def stop(signum, _frame):
        if process.poll() is None:
            process.send_signal(signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, stop)
        code = process.wait()
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    seconds = time.perf_counter() - started
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    already_complete = previous_status.get("status") == "complete" and status == previous_status
    atomic_json(
        directory / ("native_resume_check.json" if already_complete else "native_execution.json"),
        dict(command=command, returncode=code, process_seconds=seconds, case_directory=str(case)),
    )
    if code != 0:
        from .artifacts import Paused

        # Occupancy is checked before the observer opens its Run. An imported
        # checkpoint may still say "paused"; that old status is not this exit.
        if code == 3:
            raise RuntimeError("GPU occupied; native trajectory remains queued")
        if status.get("status") == "paused" and status != previous_status:
            raise Paused("Native IGM checkpointed its interrupted trajectory")
        raise RuntimeError(f"Native igm_run failed with exit {code}; see worker log")
    result = json.loads((directory / "result.json").read_text())
    if already_complete:
        # The native observer checked the complete run's full signature. Its
        # startup-only verification must not replace the measured solve cost.
        return result
    result["native_observer_total_seconds"] = result["total_seconds"]
    result["native_process_seconds"] = seconds
    result["total_seconds"] = seconds + result.get("previous_attempt_wall_seconds", 0.0)
    atomic_json(directory / "result.json", result)
    return result


def active(cfg=None):
    global _ACTIVE
    if _ACTIVE is None:
        if cfg is None:
            raise RuntimeError("Native experiment observer has not been initialized")
        from omegaconf import OmegaConf
        from .runtime import activate, provenance, tree_hash
        import igm
        import tensorflow as tf

        settings = OmegaConf.to_container(cfg.study, resolve=True)
        args = SimpleNamespace(**settings["args"])
        dependency = activate()
        tf.config.experimental.enable_tensor_float_32_execution(False)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        selected = visible[0] if visible[0] else str(cfg.core.hardware.visible_gpus[0])
        device = dict(
            requested=str(args.gpu), hostname=socket.gethostname(), tensor_float_32_execution=False
        )
        if tf.config.list_physical_devices("GPU"):
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={selected}",
                    "--query-gpu=index,uuid,name,pci.bus_id,driver_version,power.limit,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            index, uuid, name, bus, driver, power, temperature = [
                v.strip() for v in raw.strip().split(",")
            ]
            device.update(
                index=int(index),
                uuid=uuid,
                name=name,
                pci_bus_id=bus,
                driver=driver,
                power_limit_W=float(power),
                temperature_C=float(temperature),
            )
        context = provenance(device, dependency)
        from hydra.utils import get_original_cwd

        context["native_hooks_sha256"] = tree_hash(Path(get_original_cwd()) / "user")
        try:
            # Linux process age includes Python/TF imports before native IGM starts.
            fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
            process_started = float(fields[19]) / os.sysconf("SC_CLK_TCK")
            context["startup_seconds"] = time.clock_gettime(time.CLOCK_BOOTTIME) - process_started
        except (OSError, ValueError, IndexError, AttributeError):
            context["startup_seconds"] = 0.0
        _ACTIVE = Observer(
            args,
            settings["config"],
            context,
            settings["architecture"],
            settings["architecture_spec"],
            settings["directory"],
        )
        _ACTIVE.manual = True
        _ACTIVE.run.__enter__()
        if _ACTIVE.run.skip:
            _ACTIVE.run.__exit__(None, None, None)
            _ACTIVE.closed = True
            raise SystemExit(0)

        def unfinished(observer=_ACTIVE):
            if not observer.closed:
                error = RuntimeError("Native IGM exited before finalization; see engine logs")
                observer.run.__exit__(RuntimeError, error, None)
                observer.closed = True

        atexit.register(unfinished)
    return _ACTIVE
