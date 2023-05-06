import haiku as hk
import jax.numpy as jnp


class MultiEmbedder(hk.Module):
  """ An embedder for multiple discrete dimensions into a shared embedding space.
  Maintaing a seperate set of weights into the space for each dimension. Each dimension's
  embedding is summed to produce the output embedding.
  """

  def __init__(self, vocab_sizes, embed_dim, name=None, **kwargs):
    """Constructs a MultiEmbedder module/

    Args:
      vocab_sizes: A list of vocab sizes for each dimension.
      embed_dim: The number of dimensions of the shared embedding space
      name: Optional name of the module.
      **kwargs: Additional keyword arguments passed to `hk.Embed`.
      """
    super().__init__(name=name)
    self.embedders = [hk.Embed(vocab_size, embed_dim, **kwargs)
                      for vocab_size in vocab_sizes]

  def __call__(self, ids):
    embeddings = jnp.array([embedder(ids[:, i])
                           for i, embedder in enumerate(self.embedders)])
    combined = jnp.sum(embeddings, axis=0)
    return combined
