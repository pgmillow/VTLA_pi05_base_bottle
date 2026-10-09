"""JAX/NNX tactile encoder for the right Inspire hand.

Ports `/data/tactile_encoder.py` for the 11 right-hand taxel patches. Fusion into
Pi0 uses a separate scalar `tanh(alpha)` gate (alpha starts at 0).
"""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

TACTILE_IDS: tuple[str, ...] = ("12", "13", "22", "23", "32", "33", "42", "43", "52", "54", "61")
NUM_SENSORS = len(TACTILE_IDS)
CANVAS_HW: tuple[int, int] = (14, 14)
DEFAULT_NUM_FRAMES = 16
TACTILE_PATCH_SHAPE: dict[str, tuple[int, int]] = {
    "12": (12, 8),
    "13": (10, 8),
    "22": (12, 8),
    "23": (10, 8),
    "32": (12, 8),
    "33": (10, 8),
    "42": (12, 8),
    "43": (10, 8),
    "52": (12, 8),
    "54": (12, 8),
    "61": (14, 8),
}


def spatial_valid_mask(canonical_size: tuple[int, int] = CANVAS_HW) -> jnp.ndarray:
    canvas_h, canvas_w = canonical_size
    mask = np.zeros((NUM_SENSORS, canvas_h, canvas_w), dtype=np.float32)
    for idx, taxel_id in enumerate(TACTILE_IDS):
        patch_h, patch_w = TACTILE_PATCH_SHAPE[taxel_id]
        mask[idx, :patch_h, :patch_w] = 1.0
    return jnp.asarray(mask)


@dataclasses.dataclass(frozen=True)
class TactileEncoderConfig:
    num_sensors: int = NUM_SENSORS
    canonical_size: tuple[int, int] = CANVAS_HW
    num_frames: int = DEFAULT_NUM_FRAMES
    embed_dim: int = 256
    baseline_frames: int = 16
    scale_percentile: float = 99.5
    cnn_widths: tuple[int, ...] = (32, 64, 128)
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    max_frames: int = 64
    preprocessing: str = "window_median"


class TactileConst(nnx.Variable):
    """Non-trainable constant stored in the NNX graph (not a Param)."""


def _drop_fully_padded_rows(pad: jnp.ndarray) -> jnp.ndarray:
    """pad is True where the token should be ignored."""
    return pad & ~jnp.all(pad, axis=-1, keepdims=True)


class _MHA(nnx.Module):
    def __init__(self, dim: int, num_heads: int, rngs: nnx.Rngs):
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} must divide num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nnx.Linear(dim, dim, rngs=rngs)
        self.k = nnx.Linear(dim, dim, rngs=rngs)
        self.v = nnx.Linear(dim, dim, rngs=rngs)
        self.o = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, pad: jnp.ndarray | None = None) -> jnp.ndarray:
        batch, seq, _ = x.shape
        heads = self.num_heads
        q = self.q(x).reshape(batch, seq, heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(batch, seq, heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(batch, seq, heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (self.head_dim**-0.5)
        if pad is not None:
            pad = _drop_fully_padded_rows(pad)
            attn = jnp.where(pad[:, None, None, :], jnp.asarray(-1e9, dtype=attn.dtype), attn)
        weights = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        return self.o(out.transpose(0, 2, 1, 3).reshape(batch, seq, x.shape[-1]))


class TinyTactileCNN(nnx.Module):
    def __init__(self, in_channels: int, widths: tuple[int, ...], embed_dim: int, rngs: nnx.Rngs):
        convs = {}
        norms = {}
        channels = in_channels
        for index, width in enumerate(widths):
            stride = 2 if index > 0 else 1
            key = f"layer{index}"
            convs[key] = nnx.Conv(
                channels,
                width,
                kernel_size=(3, 3),
                strides=stride,
                padding="SAME",
                use_bias=False,
                rngs=rngs,
            )
            norms[key] = nnx.GroupNorm(width, num_groups=min(8, width), rngs=rngs)
            channels = width
        self.convs = nnx.Dict(convs)
        self.norms = nnx.Dict(norms)
        self.n_layers = len(widths)
        self.proj = nnx.Linear(channels, embed_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for index in range(self.n_layers):
            key = f"layer{index}"
            x = jax.nn.gelu(self.norms[key](self.convs[key](x)))
        return self.proj(jnp.mean(x, axis=(1, 2)))


class FactorisedBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, rngs: nnx.Rngs):
        self.norm_temporal = nnx.LayerNorm(dim, rngs=rngs)
        self.attn_temporal = _MHA(dim, num_heads, rngs)
        self.norm_spatial = nnx.LayerNorm(dim, rngs=rngs)
        self.attn_spatial = _MHA(dim, num_heads, rngs)
        self.norm_mlp = nnx.LayerNorm(dim, rngs=rngs)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        sensor_pad: jnp.ndarray | None,
        frame_pad: jnp.ndarray | None,
        sample_pad: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        batch, sensors, frames, dim = x.shape
        residual = x
        temporal = self.norm_temporal(x).reshape(batch * sensors, frames, dim)
        temporal_pad = None
        if frame_pad is not None:
            temporal_pad = jnp.broadcast_to(frame_pad[:, None, :], (batch, sensors, frames)).reshape(
                batch * sensors, frames
            )
        if sample_pad is not None:
            temporal_pad = sample_pad.reshape(batch * sensors, frames)
        x = residual + self.attn_temporal(temporal, temporal_pad).reshape(batch, sensors, frames, dim)

        residual = x
        spatial = jnp.swapaxes(self.norm_spatial(x), 1, 2).reshape(batch * frames, sensors, dim)
        spatial_pad = None
        if sensor_pad is not None:
            spatial_pad = jnp.broadcast_to(sensor_pad[:, None, :], (batch, frames, sensors)).reshape(
                batch * frames, sensors
            )
        if sample_pad is not None:
            spatial_pad = jnp.swapaxes(sample_pad, 1, 2).reshape(batch * frames, sensors)
        x = residual + jnp.swapaxes(
            self.attn_spatial(spatial, spatial_pad).reshape(batch, frames, sensors, dim), 1, 2
        )
        return x + self.fc2(jax.nn.gelu(self.fc1(self.norm_mlp(x))))


class TactileEncoder(nnx.Module):
    """Raw taxels [B, S, T, H, W] -> [B, S, embed_dim] tokens."""

    def __init__(self, config: TactileEncoderConfig, rngs: nnx.Rngs):
        self.config = config
        if config.embed_dim % config.num_heads != 0:
            raise ValueError("embed_dim must divide num_heads")
        self.spatial_mask = TactileConst(spatial_valid_mask(config.canonical_size))
        self.cnn = TinyTactileCNN(1, config.cnn_widths, config.embed_dim, rngs)
        self.sensor_embed = nnx.Embed(config.num_sensors, config.embed_dim, rngs=rngs)
        self.time_embed = nnx.Param(
            jax.random.truncated_normal(
                rngs.params(), -2.0, 2.0, (1, 1, config.max_frames, config.embed_dim)
            )
            * 0.02
        )
        self.blocks = nnx.Dict(
            {
                f"layer{index}": FactorisedBlock(config.embed_dim, config.num_heads, config.mlp_ratio, rngs)
                for index in range(config.depth)
            }
        )
        self.depth = config.depth
        self.norm = nnx.LayerNorm(config.embed_dim, rngs=rngs)
        self.pool_query = nnx.Param(
            jax.random.truncated_normal(rngs.params(), -2.0, 2.0, (1, 1, config.embed_dim)) * 0.02
        )
        self.pool_q = nnx.Linear(config.embed_dim, config.embed_dim, rngs=rngs)
        self.pool_k = nnx.Linear(config.embed_dim, config.embed_dim, rngs=rngs)
        self.pool_v = nnx.Linear(config.embed_dim, config.embed_dim, rngs=rngs)
        self.pool_o = nnx.Linear(config.embed_dim, config.embed_dim, rngs=rngs)
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

    def _preprocess(self, taxels: jnp.ndarray, baseline=None, scale=None, sample_valid=None) -> jnp.ndarray:
        taxels = taxels.astype(jnp.float32)
        if self.config.preprocessing == "episode_baseline":
            if baseline is None or scale is None:
                raise ValueError("episode_baseline needs tactile_baseline and tactile_scale")
            pressure = (taxels - baseline[:, :, None]) / jnp.maximum(scale[:, :, None, None, None], 1e-3)
            valid = self.spatial_mask.value[None, :, None].astype(jnp.bool_)
            if sample_valid is not None:
                valid = valid & sample_valid[..., None, None]
            # Fixed no-contact noise scale preserves sustained pressure and amplitude.
            # asinh compresses large readings without the saturation of tanh.
            return jnp.arcsinh(jnp.where(valid, pressure, 0.0))
        if self.config.preprocessing != "window_median":
            raise ValueError(f"Unknown tactile preprocessing: {self.config.preprocessing}")
        frames_used = min(self.config.baseline_frames, taxels.shape[2])
        baseline = jnp.median(taxels[:, :, :frames_used], axis=2)
        pressure = taxels - baseline[:, :, None, :, :]
        mask = self.spatial_mask.value.astype(pressure.dtype)
        pressure = pressure * mask[None, :, None, :, :]
        batch_size, n_sensors, n_frames = pressure.shape[:3]
        flat = jnp.abs(pressure).reshape(batch_size, n_sensors, -1)
        spatial = mask.reshape(1, n_sensors, 1, -1)
        weights = jnp.broadcast_to(spatial, (1, n_sensors, n_frames, spatial.shape[-1])).reshape(1, n_sensors, -1)
        weights = jnp.broadcast_to(weights, flat.shape)
        flat = jnp.where(weights > 0, flat, 0.0)
        scale = jnp.quantile(flat, self.config.scale_percentile / 100.0, axis=-1)
        scale = jax.lax.stop_gradient(jnp.maximum(scale, 1e-3))
        return jnp.tanh(pressure / scale[:, :, None, None, None])

    def _pool(self, x: jnp.ndarray, frame_pad: jnp.ndarray | None, sample_pad=None) -> jnp.ndarray:
        batch, sensors, frames, dim = x.shape
        heads = self.num_heads
        seq = x.reshape(batch * sensors, frames, dim)
        query = jnp.broadcast_to(self.pool_query.value, (batch * sensors, 1, dim))
        q = self.pool_q(query).reshape(batch * sensors, 1, heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.pool_k(seq).reshape(batch * sensors, frames, heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.pool_v(seq).reshape(batch * sensors, frames, heads, self.head_dim).transpose(0, 2, 1, 3)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (self.head_dim**-0.5)
        if frame_pad is not None or sample_pad is not None:
            pad = (sample_pad.reshape(batch * sensors, frames) if sample_pad is not None else
                   jnp.broadcast_to(frame_pad[:, None, :], (batch, sensors, frames)).reshape(batch * sensors, frames))
            pad = _drop_fully_padded_rows(pad)
            attn = jnp.where(pad[:, None, None, :], jnp.asarray(-1e9, dtype=attn.dtype), attn)
        weights = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        out = self.pool_o(out.transpose(0, 2, 1, 3).reshape(batch * sensors, 1, dim))
        return out.reshape(batch, sensors, dim)

    def __call__(
        self,
        taxels: jnp.ndarray,
        sensor_valid: jnp.ndarray | None = None,
        frame_valid: jnp.ndarray | None = None,
        sample_valid: jnp.ndarray | None = None,
        baseline: jnp.ndarray | None = None,
        scale: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        batch, sensors, frames = taxels.shape[:3]
        if sensors != self.config.num_sensors:
            raise ValueError(f"expected {self.config.num_sensors} sensors, got {sensors}")
        if sensor_valid is None:
            sensor_valid = jnp.ones((batch, sensors), dtype=jnp.bool_)
        sensor_valid = sensor_valid.astype(jnp.bool_)

        if frames > self.config.max_frames:
            raise ValueError("Too many tactile history frames")
        if sample_valid is not None:
            sample_valid = sample_valid & sensor_valid[:, :, None]
        normalised = self._preprocess(taxels, baseline, scale, sample_valid)
        canvas_h, canvas_w = normalised.shape[-2:]
        cnn_in = normalised.reshape(batch * sensors * frames, canvas_h, canvas_w, 1)
        embedded = self.cnn(cnn_in).reshape(batch, sensors, frames, self.config.embed_dim)
        embedded = embedded * sensor_valid[:, :, None, None].astype(embedded.dtype)
        embedded = embedded + self.sensor_embed(jnp.arange(sensors)).reshape(1, sensors, 1, -1)
        embedded = embedded + self.time_embed.value[:, :, :frames]

        sensor_pad = jnp.logical_not(sensor_valid)
        frame_pad = None if frame_valid is None else jnp.logical_not(frame_valid.astype(jnp.bool_))
        for index in range(self.depth):
            embedded = self.blocks[f"layer{index}"](
                embedded, sensor_pad, frame_pad, None if sample_valid is None else ~sample_valid,
            )
        embedded = self.norm(embedded)
        pooled = self._pool(embedded, frame_pad, None if sample_valid is None else ~sample_valid)
        return pooled * sensor_valid[:, :, None].astype(pooled.dtype)
