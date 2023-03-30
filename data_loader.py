import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


class DataLoaderIterator:
  def __init__(self, dataset, batch_indicies):
    self.dataset = dataset
    self.batch_indicies = batch_indicies
    self.batch_index = 0

  def __next__(self):
    if self.batch_index >= len(self.batch_indicies):
      raise StopIteration
    batch = [self.dataset[index]
             for index in self.batch_indicies[self.batch_index]]
    self.batch_index += 1
    return batch


class DataLoader:
  def __init__(self, dataset: np.ndarray, batch_size: int,
               rng: Optional[jax.random.KeyArray] = None):
    """Create a batched data loader
    params:
      dataset: a list of data points
      batch_size: the size of each batch
      rng: a jax.random.KeyArray to shuffle the dataset or None to disable shuffling
    """
    self.dataset = dataset
    self.batch_size = batch_size
    self.rng = rng

  def __iter__(self):
    n = len(self.dataset)
    if self.rng is not None:
      self.rng, subkey = jax.random.split(self.rng)
      indicies = jax.random.permutation(subkey, n, independent=True)
    else:
      indicies = jnp.arange(n)
    split_points = jnp.arange(self.batch_size, n, self.batch_size)
    batch_indicies = np.split(indicies, split_points)
    return DataLoaderIterator(self.dataset, batch_indicies)
