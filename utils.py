from functools import partial
import time
from typing import (Any, Callable, Collection, Dict, Iterator, NewType,
                    Optional, Tuple)

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from type_aliases import LabelledGraph

GraphsSize = NewType("GraphsSize", Tuple[int, int, int])
# TODO: Consider deleting this
PaddingScheme = Callable[[GraphsSize], GraphsSize]

# TODO: Consider splitting this file out


def add_lapPE(graph, pos_enc_dim):
  nodes, edges, senders, receivers, globals, n_node, n_edge = graph
  dim = nodes['feat'].shape[0]
  A = np.zeros((dim, dim))
  A[senders, receivers] = 1
  in_degrees = np.bincount(senders, minlength=dim)
  N = np.diag(in_degrees ** -0.5)
  D = np.eye(dim)
  L = D - N @ A @ N
  # TODO: Why does Dwivedi normalise with indegree matrix?
  eigValues, eigVectors = np.linalg.eig(L)
  idx = eigValues.argsort()
  eigValues, eigVectors = eigValues[idx], eigVectors[:, idx]
  # All vectors should be real, should I check this?
  pe = eigVectors[:, 1:pos_enc_dim + 1]
  return graph._replace(nodes=nodes | {'pe': pe, "eigvec": pe})


def add_jnp_lapPE(graph, pos_enc_dim):
  nodes, edges, senders, receivers, globals, n_node, n_edge = graph
  dim = nodes['feat'].shape[0]
  A = jnp.zeros((dim, dim)).at[senders, receivers].set(1)
  in_degrees = jnp.bincount(senders, minlength=dim)
  N = jnp.diag(in_degrees ** -0.5)
  D = jnp.eye(dim)
  L = D - N @ A @ N
  # TODO: Why does Dwivedi normalise with indegree matrix?
  eigValues, eigVectors = jnp.linalg.eig(L)
  idx = eigValues.argsort()
  eigValues, eigVectors = eigValues[idx], eigVectors[:, idx]
  # All vectors should be real, should I check this?
  eigVectors = jnp.real(eigVectors)
  pe = eigVectors[:, 1:pos_enc_dim + 1]
  return graph._replace(nodes=nodes | {'pe': pe, "eigvec": pe})


def _next_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than or equal to  x for padding.
  """
  y = 2
  while y < x:
    y *= 2
  return y


def get_GraphsSize(graph: jraph.GraphsTuple) -> GraphsSize:
  """Returns the triple representing the size of the GraphsTuple: (n_nodes, n_edges, n_graphs)"""
  return GraphsSize((jnp.sum(graph.n_node).item(), jnp.sum(
    graph.n_edge).item(), graph.n_node.shape[0]))


def fixed_batch_power_of_two_padding(
  size: GraphsSize, batch_size: int = 1) -> GraphsSize:
  """Pads the number of edges and nodes to the next power of two (+1 for nodes).
  Only adds a single padding graph to reach batch_size+1.
  Adds +1 is for nodes and graphs as `jraph.pad_graphs` requires at least one padding node and graph.
  Args:
    size: GraphsSize with number of nodes equal to batch_size
    batch_size: The size of the batch. Default is singleton batch.
  """
  n_nodes, n_edges, n_graphs = size
  assert(n_graphs == batch_size)
  return GraphsSize((_next_power_of_two(n_nodes) + 1,
                    _next_power_of_two(n_edges), n_graphs + 1))


def power_of_two_padding(size: GraphsSize, batch_size=None) -> GraphsSize:
  """Pads the number of edges and nodes to the next power of two (+1 for nodes).
  Adds a single graph if batch_size is None, otherwise adds enough graphs to reach batch_size+1."""
  n_nodes, n_edges, n_graphs = size
  n_nodes = _next_power_of_two(n_nodes) + 1
  n_edges = _next_power_of_two(n_edges)
  if batch_size == None:
    n_graphs += 1
  else:
    assert(n_graphs <= batch_size)
    n_graphs = batch_size + 1

  return GraphsSize((n_nodes, n_edges, n_graphs))


def monotonic_power_of_two_padding(
  size: GraphsSize, last_padding: GraphsSize, batch_size=None) -> GraphsSize:
  """Power of two padding except that the number of nodes and edges is monotonically increasing."""
  n_nodes, n_edges, n_graphs = size
  last_n_nodes, last_n_edges, last_n_graphs = last_padding
  n_nodes = max(_next_power_of_two(n_nodes) + 1, last_n_nodes)
  n_edges = max(_next_power_of_two(n_edges), last_n_edges)
  if batch_size == None:
    n_graphs = max(n_graphs + 1, last_n_graphs)
  else:
    assert(n_graphs <= batch_size)
    n_graphs = max(batch_size + 1, last_n_graphs)

  return GraphsSize((n_nodes, n_edges, n_graphs))


def pad_labelled_graph(labelled_graph: LabelledGraph,
                       padding_strategy: PaddingScheme) -> LabelledGraph:
  """
  Pads a `LabelledGraph` to the size specified by the padding strategy.
  Pads `jraph.GraphsTuple` by calling `jraph.pad_with_graphs`.
  Pads the label to the number of graphs with zeros.
  Args:
    labelled_graph: a batched LabelledGraph (can be batch size 1).
    padding_strategy: a function (possibly impure) that takes a GraphsSize and returns a GraphsSize.
  Returns:
    A LabelledGraph consisting of a padded `jraph.GraphsTuple` and a padded label.
  """
  graphs_tuple, label = labelled_graph
  original_size = get_GraphsSize(graphs_tuple)
  n_nodes, n_edges, n_graphs = padding_strategy(original_size)
  padded_graphs_tuple = jraph.pad_with_graphs(
    graphs_tuple, n_nodes, n_edges, n_graphs)
  label_padding_shape = (n_graphs - original_size[2], ) + label.shape[1:]
  padded_label = jnp.concatenate([label, jnp.zeros(label_padding_shape)])

  return padded_graphs_tuple, padded_label


def create_optimizer(
  hyper_params: Dict[str, Any]) -> optax.GradientTransformation:
  # TODO: replace learning_rate with a scheduler that uses appropriate hyper parameters
  # TODO: understand what params "weight decay" is
  steps_per_epoch = (
      10000 + hyper_params["batch_size"] - 1) // hyper_params["batch_size"]
  print("steps per epoch", steps_per_epoch)
  lr = optax.exponential_decay(
      init_value=hyper_params["init_lr"],
      transition_steps=hyper_params["transition_epochs"] * steps_per_epoch,
      decay_rate=hyper_params["lr_reduce_factor"],
      end_value=hyper_params["init_lr"] * 1e-2)
  # lr =hyper_params["init_lr"]
  return optax.adam(learning_rate=lr)


def __get__unpadded_batch(batch):
  batch_graphs = [x[0] for x in batch]
  batch_labels = [x[1] for x in batch]
  labelled_graph = (
      jraph.batch_np(batch_graphs),
      jnp.concatenate(batch_labels, axis=0))
  return labelled_graph, len(batch)


def flat_data_loader(dataset, batch_size, padding_strategy, rng):
  start_time = time.time()
  n = len(dataset)
  length = (n + batch_size - 1) // batch_size
  if rng is not None:

    '''batch_indicies = jax.random.permutation(rng, n)
    dataset = [dataset[i] for i in batch_indicies]'''
    rng = np.random.default_rng(int(rng[0]))
    rng.shuffle(dataset)
  else:
    batch_indicies = jnp.arange(n)
  shuffle_time = time.time()
  unpadded = [
      __get__unpadded_batch(
          dataset[i * batch_size:(i + 1) * batch_size]
      ) for i in range(length)]
  unpadded_time = time.time()
  batches = [(pad_labelled_graph(x[0], padding_strategy), x[1])
             for x in unpadded]
  batches_time = time.time()
  print(f"total time: {batches_time - start_time} = shuffle {shuffle_time-start_time} + unpadded: {unpadded_time - shuffle_time} + batches: {batches_time - unpadded_time}")
  return batches


class DataLoaderIterator:
  def __init__(self, dataset, batch_indicies, batch_size, padding_strategy):
    self.dataset = dataset
    self.batch_indicies = batch_indicies
    self.batch_index = 0
    self.batch_size = batch_size
    self.padding_strategy = padding_strategy

  def __next__(self) -> Tuple[LabelledGraph, int]:
    if self.batch_index >= len(self.batch_indicies):
      raise StopIteration
    batch_graphs = [self.dataset[index][0]
                    for index in self.batch_indicies[self.batch_index]]
    batch_labels = [self.dataset[index][1]
                    for index in self.batch_indicies[self.batch_index]]
    labelled_graph = (jraph.batch(batch_graphs), jnp.concatenate(batch_labels))
    self.batch_index += 1
    return (pad_labelled_graph(labelled_graph,
            self.padding_strategy), len(batch_graphs))

  def __iter__(self) -> Iterator[Tuple[LabelledGraph, int]]:
    return self


class DataLoader:
  def __init__(self, dataset: np.ndarray, batch_size: int,
               rng: Optional[jax.random.KeyArray] = None, padding_strategy: PaddingScheme = power_of_two_padding):
    """Create a batched data loader
    params:
      dataset: a list of data points
      batch_size: the size of each batch
      rng: a jax.random.KeyArray to shuffle the dataset or None to disable shuffling
      padding_stategy: a potentially stateful function to determine each batch's padding
    """
    self.dataset = dataset  # TODO: Consider whether this should be a jax array, storing this on device and whether we should pre-produce the batches
    self.batch_size = batch_size
    self.rng = rng
    self.length = (len(dataset) + batch_size - 1) // batch_size
    self.padding_strategy = padding_strategy

  def __iter__(self) -> Iterator[Tuple[LabelledGraph, int]]:
    n = len(self.dataset)
    if self.rng is not None:
      self.rng, subkey = jax.random.split(self.rng)
      indicies = jax.random.permutation(subkey, n, independent=True)
    else:
      indicies = jnp.arange(n)
    split_points = jnp.arange(self.batch_size, n, self.batch_size)
    batch_indicies = np.split(indicies, split_points)
    return DataLoaderIterator(
      self.dataset, batch_indicies, self.batch_size, self.padding_strategy)

  def __len__(self) -> int:
    return self.length


class HaikuDebug(hk.Module):
  def __init__(self, name=None, label=None, enable: bool = True):
    self.name = name
    self.label = label
    if self.name is None and self.label is None:
      self.name = self.label = "HaikuDebug"
    elif self.name is None:
      self.name = self.label
    elif self.label is None:
      self.label = self.name
    self.enable = enable

    super().__init__(name=name)

  def __call__(self, x):
    if self.enable:
      print(f"<{self.label}> {x} </{self.label}>")
