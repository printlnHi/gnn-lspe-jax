from typing import Callable, NewType, Tuple

import jraph
import numpy as np

from types_and_aliases import LabelledGraph


def _next_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than or equal to  x for padding.
  """
  y = 2
  while y < x:
    y *= 2
  return y


GraphsSize = NewType("GraphsSize", Tuple[int, int, int])


def get_GraphsSize(graph: jraph.GraphsTuple) -> GraphsSize:
  """Returns the triple representing the size of the GraphsTuple: (n_nodes, n_edges, n_graphs)"""
  return GraphsSize((np.sum(graph.n_node).item(), np.sum(
    graph.n_edge).item(), graph.n_node.shape[0]))


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


PaddingScheme = Callable[[GraphsSize], GraphsSize]


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
  padded_label = np.concatenate([label, np.zeros(label_padding_shape)])

  return padded_graphs_tuple, padded_label
