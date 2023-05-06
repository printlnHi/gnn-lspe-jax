import haiku as hk


class MLPReadout(hk.Module):
  """A multi-layer perception readout layer which halves the number of nodes at each layer"""

  def __init__(self, input_dim: int, output_dim: int, L: int = 2, name=None):
    """Constructs an MLPReadout module

    Args:
      input_dim: The number of input dimensions
      output_dim: The number of output dimensions
      L: The number of layers in the MLP
      name: Optional name of the module.
    """
    super().__init__(name=name)
    output_sizes = [input_dim // 2**l for l in range(1, L + 1)] + [output_dim]
    self.mlp = hk.nets.MLP(output_sizes)

  def __call__(self, x):
    return self.mlp(x)
