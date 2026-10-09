"""Visual-only pi05 adaptation with the same split and terminal masks as VTLA."""
import dataclasses

from flax import nnx
import jax.numpy as jnp

from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config


def masked_flow_loss(model, rng, observation, actions, *, train=False):
    loss, metrics = Pi0.compute_loss_and_metrics(model, rng, observation, actions, train=train)
    valid = getattr(observation, "action_valid", None)
    if valid is None:
        if train:
            raise ValueError("Staged training requires episode-end action masks")
        return loss, metrics
    weights = actions.shape[1] / jnp.maximum(valid.sum(-1, keepdims=True), 1)
    loss = jnp.where(valid, loss, 0) * weights
    contact = model.contact_loss_weight * metrics.get("contact_loss", 0.)
    flow = jnp.where(valid, loss - weights * contact, 0).mean()
    return loss, {**metrics, "flow_loss": flow}


@dataclasses.dataclass(frozen=True)
class Pi05VisualAdaptConfig(Pi0Config):
    pi05: bool = True
    tactile_enabled: bool = False
    use_frame_cache: bool = True
    training_objective: str = "flow"
    max_delay_frames: int = 0
    validation_fraction: float = 0.05
    split_seed: int = 42
    fps: float = 29.994

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or self.tactile_enabled or self.training_objective != "flow":
            raise ValueError("Visual adaptation is pure pi05 flow matching without tactile modules")

    def create(self, rng):
        return Pi05VisualAdapt(self, nnx.Rngs(rng))


class Pi05VisualAdapt(Pi0):
    compute_loss_and_metrics = masked_flow_loss
