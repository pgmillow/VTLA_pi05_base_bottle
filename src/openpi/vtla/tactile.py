"""Fresh native-vector encoder: channel MLP followed by a causal temporal conv."""
from flax import nnx
import jax
import jax.numpy as jnp


class O10TactileEncoder(nnx.Module):
    def __init__(self, embed_dim, rngs, *, num_frames=16):
        self.input = nnx.Linear(268,embed_dim,rngs=rngs)
        self.time_embedding = nnx.Param(jax.random.normal(rngs.params(),(num_frames,embed_dim))*.02)
        self.temporal = nnx.Conv(embed_dim,embed_dim,kernel_size=(3,),padding=((2,0),),rngs=rngs)
        self.norm = nnx.LayerNorm(embed_dim,rngs=rngs)

    def __call__(self, history, valid, baseline, scale):
        raw = history[:,0,:,0,:]
        normalized = (raw-baseline[:,0,0,None,:])/jnp.maximum(scale[:,0,None,None],1.)
        values = jnp.where(valid[...,None],jnp.tanh(normalized/5.),0.)
        tokens = jax.nn.gelu(self.input(values))+self.time_embedding.value
        tokens = jnp.where(valid[...,None],tokens,0.)
        tokens = self.norm(tokens+jax.nn.gelu(self.temporal(tokens)))
        return jnp.where(valid[...,None],tokens,0.)
