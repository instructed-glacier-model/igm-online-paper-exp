"""Release-native architectures and two explicitly named experimental controls."""
from __future__ import annotations
import copy
import numpy as np
import tensorflow as tf


def build(spec, inputs, input_names, seed=0, precision="fp32"):
    from igm.processes.iceflow.emulate.utils.architectures import Architectures
    from igm.processes.iceflow.emulate.utils.normalizations import AdaptiveAffineLayer

    tf.keras.utils.set_random_seed(seed)
    dtype = "float64" if precision == "fp64" else "float32"
    tf.keras.backend.set_floatx(dtype)
    spec = copy.deepcopy(spec)
    params = spec["params"]
    family = spec["family"]
    if family == "identity":
        raise ValueError("Identity uses the IGM mapping directly, not a neural model")
    if family in ("cnn", "mlp"):
        params["precision"] = dtype
    if family == "dahunet":
        params.update(backend=spec["backend"], features=spec["features"])
    cls = Architectures["dahunet" if family == "controlled_backbone" else family]
    if family == "controlled_backbone":
        model = ControlledBackbone(
            input_names=input_names,
            params=params,
            features=spec["features"],
            backend=spec.get("backend", "cnn"),
            scramble=spec.get("scramble", False),
            scramble_seed=spec.get("scramble_seed", 42),
            dtype=dtype,
        )
    elif spec.get("scramble"):
        model = ScrambledDahuNet(
            input_names=input_names, params=params, seed=spec["scramble_seed"], dtype=dtype
        )
    else:
        model = cls(input_names=input_names, Nz=2, network_params=params, dtype=dtype)
    normalizer = AdaptiveAffineLayer(nb_channels=len(input_names), dtype=dtype)
    x = tf.cast(inputs, dtype)
    means, variance = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
    normalizer.set_stats(means, variance)
    model.input_normalizer = normalizer
    model(x)
    return model


class ControlledBackbone(tf.keras.Model):
    """One native hidden backend with controlled raw/physical/scrambled inputs."""

    def __init__(
        self,
        *,
        input_names,
        params,
        features,
        backend="cnn",
        scramble=False,
        scramble_seed=42,
        **kwargs,
    ):
        from igm.processes.iceflow.emulate.utils.architectures.dahunet import (
            _CNNBackend,
            _FNOBackend,
            FEATURES,
        )

        super().__init__(**kwargs)
        self.input_names = list(input_names)
        self.nb_inputs = len(input_names)
        self.features = list(features)
        self.feature_fns = [FEATURES[name] for name in features]
        self.scramble = scramble
        self.scramble_seed = scramble_seed
        self.input_normalizer = None
        self.dummy_shape = (8, 8)
        if backend == "cnn":
            self.backbone = _CNNBackend(
                len(input_names) + len(features),
                4,
                n_filters=params["nb_out_filter"],
                n_layers=params["nb_layers"],
                ker_size=params.get("conv_ker_size", 3),
                use_residual=params.get("residual", True),
                dtype=self.compute_dtype,
                name="backbone",
            )
        elif backend == "fno":
            self.backbone = _FNOBackend(
                len(input_names) + len(features),
                4,
                width=params["width"],
                modes1=params["modes1"],
                modes2=params["modes2"],
                padding=params.get("padding", 9),
                use_grid=params.get("use_grid", True),
                dtype=self.compute_dtype,
                name="backbone",
            )
            self.dummy_shape = (max(16, params["modes1"] + 1), max(16, 2 * params["modes2"] + 2))
        else:
            raise ValueError(f"Unsupported controlled backbone: {backend}")

    def build(self, input_shape):
        if not self.built:
            self.call(tf.zeros((1, *self.dummy_shape, self.nb_inputs), dtype=self.compute_dtype))
            super().build(input_shape)

    def call(self, inputs, training=False):
        from igm.processes.iceflow.emulate.utils.architectures.dahunet import _dict_proxies

        raw = tf.cast(inputs, self.compute_dtype)
        norm = self.input_normalizer(raw) if self.input_normalizer is not None else raw
        ctx = _dict_proxies(
            raw,
            self.input_names.index("thk"),
            self.input_names.index("usurf"),
            self.input_names.index("dX"),
        )
        fields = [fn(ctx) for fn in self.feature_fns]
        if self.scramble and fields:
            stacked = tf.concat(fields, axis=-1)
            ny, nx, nf = int(raw.shape[1]), int(raw.shape[2]), len(fields)
            noise = np.random.default_rng(self.scramble_seed).normal(size=(nf, ny, nx))
            phases = np.exp(1j * np.angle(np.fft.rfft2(noise)))
            phases[:, 0, 0] = 1.0
            phase = tf.constant(phases, tf.complex128 if raw.dtype == tf.float64 else tf.complex64)
            shuffled = tf.signal.irfft2d(
                tf.signal.rfft2d(tf.transpose(stacked, [0, 3, 1, 2])) * phase, fft_length=[ny, nx]
            )
            fields = [tf.transpose(shuffled, [0, 2, 3, 1])]
        joined = tf.concat([*fields, norm], axis=-1) if fields else norm
        return self.backbone(joined, training=training)


class ScrambledDahuNet(tf.keras.Model):
    """Fixed random Fourier phases applied only to the physical features.

    The phase field is generated from a real random raster, preserving conjugate
    symmetry and the feature DC component. It is constant for a shape and seed.
    """

    def __init__(self, *, input_names, params, seed, **kwargs):
        from igm.processes.iceflow.emulate.utils.architectures.dahunet import DahuNet

        super().__init__(**kwargs)
        self.native = DahuNet(
            input_names=input_names, Nz=2, network_params=params, dtype=self.compute_dtype
        )
        self.nb_inputs = len(input_names)
        self.input_normalizer = None
        self.seed = seed
        self.phase = None

    def build(self, input_shape):
        if not self.built:
            self.call(tf.zeros((1, 8, 8, self.nb_inputs), dtype=self.compute_dtype))
            super().build(input_shape)

    def call(self, inputs, training=False):
        from igm.processes.iceflow.emulate.utils.architectures.dahunet import _dict_proxies

        raw = tf.cast(inputs, self.compute_dtype)
        ctx = _dict_proxies(raw, self.native.idx_thk, self.native.idx_usurf, self.native.idx_dX)
        fields = tf.concat([fn(ctx) for fn in self.native.feature_fns], axis=-1)
        ny, nx, nf = int(inputs.shape[1]), int(inputs.shape[2]), int(fields.shape[-1])
        noise = np.random.default_rng(self.seed).normal(size=(nf, ny, nx))
        phases = np.exp(1j * np.angle(np.fft.rfft2(noise)))
        phases[:, 0, 0] = 1.0
        phase = tf.constant(phases, tf.complex128 if raw.dtype == tf.float64 else tf.complex64)
        scrambled = tf.signal.irfft2d(
            tf.signal.rfft2d(tf.transpose(fields, [0, 3, 1, 2])) * phase, fft_length=[ny, nx]
        )
        normalized = self.input_normalizer(raw) if self.input_normalizer is not None else raw
        with tf.name_scope("scrambled_backbone"):
            return self.native._backend(
                tf.concat([tf.transpose(scrambled, [0, 2, 3, 1]), normalized], -1),
                training=training,
            )


def forward(model, inputs):
    from igm.processes.iceflow.utils.data_preprocessing import Y_to_UV

    return tf.stack(Y_to_UV(2, model(inputs, training=False)))


def count(model):
    return sum(int(np.prod(v.shape)) for v in model.trainable_variables)


def optimizer(config, lr):
    online = config["online"]
    return tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=online["beta1"],
        beta_2=online["beta2"],
        clipnorm=online["clipnorm"],
    )


def training_step(problem, model, inputs, opt=None, lr=None):
    with tf.GradientTape() as tape:
        uv = forward(model, inputs)
        value = problem.energy(uv, inputs)
    params = model.trainable_variables
    grads = tape.gradient(value, params)
    if opt is not None:
        opt.apply_gradients(zip(grads, params))
    else:
        for parameter, g in zip(params, grads):
            parameter.assign_sub(tf.cast(lr, parameter.dtype) * g)
    return value
