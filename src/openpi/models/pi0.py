import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tactile_encoder as _tactile
from openpi.models.tactile_adapter import TactileActionAdapter
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.continuous_state_input = config.continuous_state_input
        self.loss_action_dim = config.loss_action_dim or config.action_dim
        self.loss_action_dims = config.loss_action_dims
        self.image_aug_crop_enabled = config.image_aug_crop_enabled
        self.image_aug_crop_ratio = config.image_aug_crop_ratio
        self.image_aug_rotate_deg = config.image_aug_rotate_deg
        self.image_aug_brightness = config.image_aug_brightness
        self.image_aug_contrast = config.image_aug_contrast
        self.image_aug_saturation = config.image_aug_saturation
        self.image_aug_hue = config.image_aug_hue
        self.image_aug_prob = config.image_aug_prob
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        if config.continuous_state_input or not config.pi05:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if not config.pi05:
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.tactile_enabled = config.tactile_enabled
        self.tactile_fusion = config.tactile_fusion
        self.contact_loss_weight = config.contact_loss_weight
        if config.tactile_enabled:
            if getattr(config, 'tactile_layout', None) == 'o10_268':
                from openpi.vtla.tactile import O10TactileEncoder
                self.tactile_encoder = O10TactileEncoder(config.tactile_embed_dim,rngs,num_frames=config.tactile_num_frames)
            else:
                self.tactile_encoder = _tactile.TactileEncoder(
                    _tactile.TactileEncoderConfig(
                        num_sensors=config.tactile_num_sensors,
                        num_frames=config.tactile_num_frames,
                        embed_dim=config.tactile_embed_dim,
                        baseline_frames=config.tactile_num_frames,
                        preprocessing="episode_baseline" if config.tactile_fusion == "action_adapter" else "window_median",
                    ), rngs=rngs,
                )
            if self.tactile_fusion == "action_adapter":
                self.tactile_adapter = TactileActionAdapter(
                    config.tactile_embed_dim, action_expert_config.width, rngs,
                )
            else:
                self.tactile_to_backbone = nnx.Linear(config.tactile_embed_dim, paligemma_config.width, rngs=rngs)
                # Legacy mode retains extra prefix tokens even when this gain is zero.
                self.tactile_gate = nnx.Param(jnp.zeros(()))

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        if self.tactile_enabled and self.tactile_fusion == "prefix" and obs.tactile is not None:
            tactile_tokens = self.tactile_encoder(
                obs.tactile,
                sensor_valid=obs.tactile_valid,
                frame_valid=obs.tactile_frame_valid,
            )
            tactile_tokens = self.tactile_to_backbone(tactile_tokens)
            gate = jnp.tanh(self.tactile_gate.value)
            tactile_tokens = tactile_tokens * gate
            if obs.tactile_valid is not None:
                tactile_tokens = tactile_tokens * obs.tactile_valid[:, :, None].astype(tactile_tokens.dtype)
                tactile_mask = obs.tactile_valid
            else:
                tactile_mask = jnp.ones(tactile_tokens.shape[:2], dtype=jnp.bool_)
            tokens.append(tactile_tokens)
            input_mask.append(tactile_mask)
            ar_mask += [False] * tactile_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def encode_tactile_condition(self, obs):
        if not self.tactile_enabled or self.tactile_fusion != "action_adapter" or obs.tactile is None:
            return None
        if obs.tactile_baseline is None or obs.tactile_scale is None or obs.tactile_sample_valid is None:
            raise ValueError("Use TactileContactPreprocessor to supply baseline, scale and per-frame validity")
        valid = obs.tactile_sample_valid[:, :, -1]
        if obs.tactile_valid is not None:
            valid = valid & obs.tactile_valid
        tokens = self.tactile_encoder(obs.tactile, sensor_valid=valid,
                                     frame_valid=obs.tactile_frame_valid,
                                     sample_valid=obs.tactile_sample_valid,
                                     baseline=obs.tactile_baseline, scale=obs.tactile_scale)
        mask = self.tactile_encoder.spatial_mask.value[None] * valid[..., None, None]
        current = (obs.tactile[:, :, -1] - obs.tactile_baseline) / jnp.maximum(obs.tactile_scale[..., None, None], 1e-3)
        current = jnp.where(mask > 0, current, 0)
        magnitude = jnp.abs(current)
        denominator = jnp.maximum(jnp.sum(mask, axis=(1, 2, 3)), 1)
        delta = jnp.zeros_like(current)
        if obs.tactile.shape[2] > 1:
            pair_valid = valid & obs.tactile_sample_valid[:, :, -2]
            delta = (obs.tactile[:, :, -1] - obs.tactile[:, :, -2]) / jnp.maximum(obs.tactile_scale[..., None, None], 1e-3)
            delta = jnp.where((mask > 0) & pair_valid[..., None, None], delta, 0)
        stats = jnp.stack([
            jnp.log1p(jnp.sum(magnitude, axis=(1, 2, 3)) / denominator),
            jnp.log1p(jnp.max(magnitude, axis=(1, 2, 3))),
            jnp.sum((magnitude >= 3) * mask, axis=(1, 2, 3)) / denominator,
            jnp.log1p(jnp.sum(jnp.abs(delta), axis=(1, 2, 3)) / denominator),
        ], axis=-1)
        logits, gate = self.tactile_adapter.contact(tokens, valid, stats)
        return tokens, valid, logits, gate

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"],
        tactile_condition: tuple | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if self.continuous_state_input or not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        if tactile_condition is not None:
            tactile_tokens, tactile_valid, _, contact_gate = tactile_condition
            action_expert_tokens = self.tactile_adapter(
                action_expert_tokens, tactile_tokens, tactile_valid, contact_gate,
            )
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self.compute_loss_and_metrics(rng, observation, actions, train=train)
        return loss

    def compute_loss_and_metrics(self, rng, observation, actions, *, train=False):
        loss, metrics, _ = self._flow_loss_with_context(rng, observation, actions, train=train)
        return loss, metrics

    def _flow_loss_with_context(self, rng, observation, actions, *, train=False):
        """Expose the already-computed prefix cache for a detached slow rollout."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            image_aug_crop_enabled=self.image_aug_crop_enabled,
            image_aug_crop_ratio=self.image_aug_crop_ratio,
            image_aug_rotate_deg=self.image_aug_rotate_deg,
            image_aug_brightness=self.image_aug_brightness,
            image_aug_contrast=self.image_aug_contrast,
            image_aug_saturation=self.image_aug_saturation,
            image_aug_hue=self.image_aug_hue,
            image_aug_prob=self.image_aug_prob,
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        if self.loss_action_dims is not None:
            action_dim_mask = jnp.isin(jnp.arange(self.action_dim), jnp.array(self.loss_action_dims))
        else:
            action_dim_mask = jnp.arange(self.action_dim) < self.loss_action_dim
        actions = jnp.where(action_dim_mask, actions, 0.0)
        noise = jnp.where(action_dim_mask, noise, 0.0)
        action_dim_count = jnp.sum(action_dim_mask)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        tactile_condition = self.encode_tactile_condition(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time, tactile_condition)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        sq_err = jnp.square(v_t.astype(jnp.float32) - u_t.astype(jnp.float32))
        active_sq_err = jnp.where(action_dim_mask, sq_err, 0.0)
        flow_loss = jnp.sum(active_sq_err, axis=-1) / action_dim_count
        contact_loss = jnp.asarray(0.0)
        metrics = {"flow_loss": jnp.mean(flow_loss)}
        if tactile_condition is not None:
            _, valid, logits, gate = tactile_condition
            if observation.tactile_contact is None or observation.tactile_contact_valid is None:
                if train:
                    raise ValueError("Contact-supervised training requires contact labels and label validity")
            else:
                labels = observation.tactile_contact
                known = observation.tactile_contact_valid & jnp.any(valid, axis=-1)
                bce = jax.nn.softplus(logits) - logits * labels
                count = jnp.maximum(jnp.sum(known), 1)
                contact_loss = jnp.sum(jnp.where(known, bce, 0)) / count
                metrics["contact_label_fraction"] = jnp.mean(known.astype(jnp.float32))
                metrics["contact_positive_fraction"] = jnp.sum(jnp.where(known, labels, 0)) / count
                metrics["contact_accuracy"] = jnp.sum(((gate >= 0.5) == (labels >= 0.5)) * known) / count
            metrics["contact_gate"] = jnp.mean(gate)
            metrics["contact_loss"] = contact_loss
        context = (observation, prefix_mask, kv_cache, tactile_condition)
        return flow_loss + self.contact_loss_weight * contact_loss, metrics, context

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # Encode the latest tactile history once per action sampling call, not once per flow step.
        tactile_condition = self.encode_tactile_condition(observation)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size), tactile_condition
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
