import haiku as hk
import jax.numpy as jnp
import jax.random as random
import numpy as np


# L=nb_hidden_layers #TODO: repeating comment from OG repo
def mlp_readout(x, input_dim, output_dim, L=2):
  output_sizes = [input_dim // 2**l for l in range(1, L + 1)] + [output_dim]
  # TODO: Are intiailisation procedures for MLPs in Haiku the same as in
  # PyTorch?
  mlp = hk.nets.MLP(output_sizes)
  return mlp(x)
