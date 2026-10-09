#!/usr/bin/env python3
"""Spatio-temporal tactile encoder for the VTLA model.

Data path (one stage per box in the architecture diagram):

    raw taxels        [B, 22, T, Hc, Wc]
      -> TactilePreprocessor    baseline removal, robust scaling, canonical padding
      -> TinyTactileCNN         shared per-sensor CNN, [1, 14, 14] -> 256
      -> concat 22 streams      [B, 22, T, 256]
      -> SpatioTemporalEncoder  factorised temporal / cross-sensor attention
      -> tactile tokens         [B, 22, 256]
      -> ZeroInitGate           zero-initialised injection into the VLA backbone

The Inspire hand exposes 11 taxel patches per hand in three physical sizes, given
here as (height, width) exactly as the sensor_msgs/Image fields report them:
12x8 for ids 12/22/32/42/52/54, 10x8 for 13/23/33/43, and 14x8 for 61.
Patches are zero-padded into a common canvas; the per-sensor embedding lets the
network resolve which region of the canvas is real for a given sensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

TACTILE_IDS: tuple[str, ...] = ("12", "13", "22", "23", "32", "33", "42", "43", "52", "54", "61")
TACTILE_LAYOUT_VERSION = "inspire_hand_v1"
HANDS: tuple[str, ...] = ("left", "right")

# Native (height, width) of every taxel patch, as recorded in the MCAP streams.
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

# Flattened sensor order used by every tensor in this module: left hand first.
SENSOR_ORDER: tuple[tuple[str, str], ...] = tuple(
    (hand, taxel_id) for hand in HANDS for taxel_id in TACTILE_IDS
)
NUM_SENSORS = len(SENSOR_ORDER)

# 30 Hz capture; 16 frames ~= 533 ms, which matches the measured lead of the
# teleoperation command over the achieved joint state.
DEFAULT_NUM_FRAMES = 16


def sensor_index(hand: str, taxel_id: str) -> int:
    return SENSOR_ORDER.index((hand, taxel_id))


def spatial_valid_mask(canonical_size: tuple[int, int]) -> torch.Tensor:
    """Per-sensor mask marking which canvas pixels carry a real taxel."""
    canvas_h, canvas_w = canonical_size
    mask = torch.zeros(NUM_SENSORS, canvas_h, canvas_w)
    for idx, (_hand, taxel_id) in enumerate(SENSOR_ORDER):
        patch_h, patch_w = TACTILE_PATCH_SHAPE[taxel_id]
        if patch_h > canvas_h or patch_w > canvas_w:
            raise ValueError(
                f"canonical_size {canonical_size} is too small for patch {taxel_id} "
                f"of shape {(patch_h, patch_w)}"
            )
        mask[idx, :patch_h, :patch_w] = 1.0
    return mask


def pad_to_canvas(
    patches: dict[tuple[str, str], torch.Tensor],
    canonical_size: tuple[int, int] = (14, 14),
) -> torch.Tensor:
    """Assemble native-shape patches into a [B, 22, T, Hc, Wc] canvas.

    Each value is a [B, T, H, W] tensor for one (hand, taxel_id) sensor. Missing
    sensors are left as zeros; pass a sensor_valid mask to the encoder to mark them.
    """
    canvas_h, canvas_w = canonical_size
    reference = next(iter(patches.values()))
    batch, frames = reference.shape[:2]
    canvas = reference.new_zeros(batch, NUM_SENSORS, frames, canvas_h, canvas_w)
    for (hand, taxel_id), patch in patches.items():
        patch_h, patch_w = TACTILE_PATCH_SHAPE[taxel_id]
        if patch.shape[-2:] != (patch_h, patch_w):
            raise ValueError(
                f"sensor {(hand, taxel_id)} has shape {tuple(patch.shape[-2:])}, "
                f"expected {(patch_h, patch_w)}"
            )
        canvas[:, sensor_index(hand, taxel_id), :, :patch_h, :patch_w] = patch
    return canvas


@dataclass
class TactileEncoderConfig:
    num_sensors: int = NUM_SENSORS
    canonical_size: tuple[int, int] = (14, 14)
    num_frames: int = DEFAULT_NUM_FRAMES
    embed_dim: int = 256

    # Preprocessing
    baseline_frames: int = 30
    scale_percentile: float = 99.5
    squash: str = "tanh"  # "tanh" | "clamp" | "none"
    use_temporal_delta: bool = False

    # Tiny CNN
    cnn_widths: tuple[int, ...] = (32, 64, 128)

    # Spatio-temporal transformer
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    temporal_pool: str = "attention"  # "attention" | "mean" | "last"
    max_frames: int = 64

    # Injection into the VLA backbone
    backbone_dim: int | None = None
    num_inject_layers: int = 1
    gate_per_channel: bool = False


@dataclass
class TactileEncoderOutput:
    tokens: torch.Tensor  # [B, S, embed_dim]
    sensor_valid: torch.Tensor  # [B, S] bool
    backbone_tokens: torch.Tensor | None = None  # [B, S, backbone_dim]
    frame_tokens: torch.Tensor | None = None  # [B, S, T, embed_dim]
    aux: dict[str, torch.Tensor] = field(default_factory=dict)


class TactilePreprocessor(nn.Module):
    """Baseline removal and robust per-sensor scaling.

    Raw taxel counts are mono16. Patch 54 reads two orders of magnitude hotter
    than the others and saturates, so the normalised signal is squashed to keep
    a single sensor from dominating the shared CNN. Dead channels are common and
    vary per episode, hence the explicit sensor_valid mask rather than a global
    channel list.
    """

    def __init__(self, config: TactileEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("spatial_mask", spatial_valid_mask(config.canonical_size), persistent=False)
        # Dataset-level scales; overwrite with set_patch_scale() once calibrated.
        self.register_buffer("patch_scale", torch.ones(config.num_sensors))
        self.register_buffer("scale_is_calibrated", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def set_patch_scale(self, scale: torch.Tensor) -> None:
        if scale.shape != self.patch_scale.shape:
            raise ValueError(f"expected scale of shape {tuple(self.patch_scale.shape)}, got {tuple(scale.shape)}")
        self.patch_scale.copy_(scale.to(self.patch_scale))
        self.scale_is_calibrated.fill_(True)

    def _estimate_baseline(self, taxels: torch.Tensor) -> torch.Tensor:
        frames = min(self.config.baseline_frames, taxels.shape[2])
        return taxels[:, :, :frames].median(dim=2).values

    def _per_sample_scale(self, pressure: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, sensors, frames = pressure.shape[:3]
        flat = pressure.abs().reshape(batch, sensors, -1)
        # The spatial mask is per sensor; tile it across the time axis so padded
        # taxels cannot inflate the high percentile.
        weights = mask.reshape(1, sensors, 1, -1).expand(1, sensors, frames, -1).reshape(1, sensors, -1)
        flat = torch.where(weights > 0, flat, torch.zeros_like(flat))
        quantile = torch.quantile(flat, self.config.scale_percentile / 100.0, dim=-1)
        return quantile.clamp_min(1e-3)

    def forward(
        self,
        taxels: torch.Tensor,
        baseline: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, S, T, Hc, Wc] raw counts -> [B, S, T, C, Hc, Wc] normalised."""
        if taxels.dim() != 5:
            raise ValueError(f"expected taxels of shape [B, S, T, H, W], got {tuple(taxels.shape)}")
        taxels = taxels.float()
        if baseline is None:
            baseline = self._estimate_baseline(taxels)
        pressure = taxels - baseline.unsqueeze(2)

        mask = self.spatial_mask.to(pressure.dtype)
        pressure = pressure * mask.unsqueeze(0).unsqueeze(2)

        if bool(self.scale_is_calibrated):
            scale = self.patch_scale.reshape(1, -1, 1, 1, 1)
        else:
            scale = self._per_sample_scale(pressure, mask).reshape(*pressure.shape[:2], 1, 1, 1)
        normalised = pressure / scale

        if self.config.squash == "tanh":
            normalised = torch.tanh(normalised)
        elif self.config.squash == "clamp":
            normalised = normalised.clamp(-4.0, 4.0)
        elif self.config.squash != "none":
            raise ValueError(f"unknown squash mode: {self.config.squash}")

        channels = [normalised]
        if self.config.use_temporal_delta:
            delta = torch.zeros_like(normalised)
            delta[:, :, 1:] = normalised[:, :, 1:] - normalised[:, :, :-1]
            channels.append(delta)
        return torch.stack(channels, dim=3)


class TinyTactileCNN(nn.Module):
    """Per-sensor encoder shared across all 22 patches and all timesteps."""

    def __init__(self, in_channels: int, widths: tuple[int, ...], embed_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for index, width in enumerate(widths):
            stride = 2 if index > 0 else 1
            layers += [
                nn.Conv2d(channels, width, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(min(8, width), width),
                nn.GELU(),
            ]
            channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[N, C, H, W] -> [N, embed_dim]."""
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class FactorisedBlock(nn.Module):
    """Temporal attention within a sensor, then attention across the 22 sensors."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm_temporal = nn.LayerNorm(dim)
        self.attn_temporal = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_spatial = nn.LayerNorm(dim)
        self.attn_spatial = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        sensor_pad: torch.Tensor | None,
        frame_pad: torch.Tensor | None,
    ) -> torch.Tensor:
        """x: [B, S, T, D]; *_pad are True where the entry should be ignored."""
        batch, sensors, frames, dim = x.shape

        residual = x
        temporal = self.norm_temporal(x).reshape(batch * sensors, frames, dim)
        temporal_pad = None
        if frame_pad is not None:
            temporal_pad = frame_pad.unsqueeze(1).expand(batch, sensors, frames).reshape(batch * sensors, frames)
            temporal_pad = _drop_fully_padded_rows(temporal_pad)
        temporal, _ = self.attn_temporal(temporal, temporal, temporal, key_padding_mask=temporal_pad)
        x = residual + temporal.reshape(batch, sensors, frames, dim)

        residual = x
        spatial = self.norm_spatial(x).transpose(1, 2).reshape(batch * frames, sensors, dim)
        spatial_pad = None
        if sensor_pad is not None:
            spatial_pad = sensor_pad.unsqueeze(1).expand(batch, frames, sensors).reshape(batch * frames, sensors)
            spatial_pad = _drop_fully_padded_rows(spatial_pad)
        spatial, _ = self.attn_spatial(spatial, spatial, spatial, key_padding_mask=spatial_pad)
        x = residual + spatial.reshape(batch, frames, sensors, dim).transpose(1, 2)

        return x + self.mlp(self.norm_mlp(x))


def _drop_fully_padded_rows(pad: torch.Tensor) -> torch.Tensor:
    """Un-mask rows that are entirely padded, which would otherwise yield NaN."""
    return pad & ~pad.all(dim=-1, keepdim=True)


class SpatioTemporalEncoder(nn.Module):
    """[B, S, T, D] -> [B, S, D] via factorised attention and temporal pooling."""

    def __init__(self, config: TactileEncoderConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.sensor_embed = nn.Embedding(config.num_sensors, dim)
        self.time_embed = nn.Parameter(torch.zeros(1, 1, config.max_frames, dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        self.blocks = nn.ModuleList(
            FactorisedBlock(dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(dim)
        if config.temporal_pool == "attention":
            self.pool_query = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            self.pool_attn = nn.MultiheadAttention(dim, config.num_heads, batch_first=True)

    def forward(
        self,
        x: torch.Tensor,
        sensor_pad: torch.Tensor | None,
        frame_pad: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sensors, frames, dim = x.shape
        if frames > self.config.max_frames:
            raise ValueError(f"got {frames} frames, max_frames is {self.config.max_frames}")

        x = x + self.sensor_embed.weight.reshape(1, sensors, 1, dim)
        x = x + self.time_embed[:, :, :frames]
        for block in self.blocks:
            x = block(x, sensor_pad, frame_pad)
        x = self.norm(x)

        if self.config.temporal_pool == "mean":
            pooled = x.mean(dim=2)
        elif self.config.temporal_pool == "last":
            pooled = x[:, :, -1]
        else:
            sequence = x.reshape(batch * sensors, frames, dim)
            query = self.pool_query.expand(batch * sensors, 1, dim)
            pool_pad = None
            if frame_pad is not None:
                pool_pad = frame_pad.unsqueeze(1).expand(batch, sensors, frames).reshape(batch * sensors, frames)
                pool_pad = _drop_fully_padded_rows(pool_pad)
            pooled, _ = self.pool_attn(query, sequence, sequence, key_padding_mask=pool_pad)
            pooled = pooled.reshape(batch, sensors, dim)
        return pooled, x


class ZeroInitGate(nn.Module):
    """Zero-initialised learnable gate, one scalar (or vector) per inject layer.

    tanh(0) == 0, so the tactile branch contributes nothing at step zero and the
    pretrained backbone starts from its original behaviour.
    """

    def __init__(self, dim: int, num_layers: int = 1, per_channel: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        shape = (num_layers, dim) if per_channel else (num_layers, 1)
        self.alpha = nn.Parameter(torch.zeros(shape))

    def gate(self, layer_index: int = 0) -> torch.Tensor:
        return torch.tanh(self.alpha[layer_index])

    def forward(self, x: torch.Tensor, layer_index: int = 0) -> torch.Tensor:
        return x * self.gate(layer_index)


class TactileEncoder(nn.Module):
    """Full tactile branch: preprocessing, Tiny CNN, spatio-temporal transformer, gate."""

    def __init__(self, config: TactileEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or TactileEncoderConfig()
        in_channels = 2 if self.config.use_temporal_delta else 1
        self.preprocess = TactilePreprocessor(self.config)
        self.sensor_cnn = TinyTactileCNN(in_channels, self.config.cnn_widths, self.config.embed_dim)
        self.spatio_temporal = SpatioTemporalEncoder(self.config)
        self.gate = ZeroInitGate(
            self.config.embed_dim if self.config.backbone_dim is None else self.config.backbone_dim,
            num_layers=self.config.num_inject_layers,
            per_channel=self.config.gate_per_channel,
        )
        self.to_backbone: nn.Module = nn.Identity()
        if self.config.backbone_dim is not None:
            self.to_backbone = nn.Linear(self.config.embed_dim, self.config.backbone_dim)

    def forward(
        self,
        taxels: torch.Tensor,
        sensor_valid: torch.Tensor | None = None,
        baseline: torch.Tensor | None = None,
        frame_valid: torch.Tensor | None = None,
        return_frame_tokens: bool = False,
    ) -> TactileEncoderOutput:
        """taxels: [B, S, T, Hc, Wc] raw counts."""
        batch, sensors, frames = taxels.shape[:3]
        if sensors != self.config.num_sensors:
            raise ValueError(f"expected {self.config.num_sensors} sensors, got {sensors}")

        if sensor_valid is None:
            sensor_valid = taxels.new_ones(batch, sensors, dtype=torch.bool)
        sensor_valid = sensor_valid.bool()

        normalised = self.preprocess(taxels, baseline=baseline)
        channels, canvas_h, canvas_w = normalised.shape[3:]
        flat = normalised.reshape(batch * sensors * frames, channels, canvas_h, canvas_w)
        embedded = self.sensor_cnn(flat).reshape(batch, sensors, frames, self.config.embed_dim)

        # Dead channels must not leak their (all-zero, but bias-shifted) embedding.
        embedded = embedded * sensor_valid.reshape(batch, sensors, 1, 1)

        sensor_pad = ~sensor_valid
        frame_pad = ~frame_valid.bool() if frame_valid is not None else None
        tokens, frame_tokens = self.spatio_temporal(embedded, sensor_pad, frame_pad)
        tokens = tokens * sensor_valid.unsqueeze(-1)

        backbone_tokens = None
        if self.config.backbone_dim is not None:
            backbone_tokens = self.to_backbone(tokens) * sensor_valid.unsqueeze(-1)

        return TactileEncoderOutput(
            tokens=tokens,
            sensor_valid=sensor_valid,
            backbone_tokens=backbone_tokens,
            frame_tokens=frame_tokens if return_frame_tokens else None,
            aux={"gate": self.gate.gate(0).detach()},
        )

    def inject(self, tokens: torch.Tensor, layer_index: int = 0) -> torch.Tensor:
        """Scale tactile tokens by the zero-initialised gate before fusion."""
        return self.gate(tokens, layer_index)


def _smoke_test() -> None:
    torch.manual_seed(0)
    config = TactileEncoderConfig(backbone_dim=512, num_inject_layers=4)
    encoder = TactileEncoder(config)
    params = sum(p.numel() for p in encoder.parameters())

    batch, frames = 4, config.num_frames
    patches = {
        (hand, taxel_id): torch.randint(0, 4000, (batch, frames, *TACTILE_PATCH_SHAPE[taxel_id])).float()
        for hand, taxel_id in SENSOR_ORDER
    }
    taxels = pad_to_canvas(patches, config.canonical_size)

    sensor_valid = torch.ones(batch, NUM_SENSORS, dtype=torch.bool)
    sensor_valid[0, sensor_index("right", "23")] = False  # dead channel
    sensor_valid[1, sensor_index("left", "52")] = False

    out = encoder(taxels, sensor_valid=sensor_valid, return_frame_tokens=True)
    print(f"params            {params/1e6:.2f}M")
    print(f"input             {tuple(taxels.shape)}")
    print(f"frame tokens      {tuple(out.frame_tokens.shape)}")
    print(f"tactile tokens    {tuple(out.tokens.shape)}")
    print(f"backbone tokens   {tuple(out.backbone_tokens.shape)}")
    print(f"dead token norm   {out.tokens[0, sensor_index('right', '23')].abs().max().item():.6f}")
    print(f"live token norm   {out.tokens[0, sensor_index('left', '12')].norm().item():.4f}")

    gated = encoder.inject(out.backbone_tokens, layer_index=0)
    print(f"gate at init      {encoder.gate.gate(0).item():.6f}")
    print(f"gated output max  {gated.abs().max().item():.6f}")

    gated.sum().backward()
    missing = [name for name, p in encoder.named_parameters() if p.grad is None]
    print(f"params w/o grad   {missing}")

    assert torch.isfinite(out.tokens).all(), "non-finite tokens"
    assert out.tokens[0, sensor_index("right", "23")].abs().max() == 0, "dead channel leaked"
    assert gated.abs().max() == 0, "gate is not zero at init"
    assert not missing, f"parameters never reached by backward: {missing}"
    print("ok")


if __name__ == "__main__":
    _smoke_test()
