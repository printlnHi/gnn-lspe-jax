import jraph
import jax.numpy as jnp
import optax

from typing import Any, Dict
from type_aliases import LabelledGraphs


def _nearest_bigger_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than x for padding.
  Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
  """
  y = 2
  while y < x:
    y *= 2
  return y


def pad_graph_to_nearest_power_of_two(
        graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Pads a batched `GraphsTuple` to the nearest power of two.
  For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
  would pad the `GraphsTuple` nodes and edges:
    7 nodes --> 8 nodes (2^3)
    5 edges --> 8 edges (2^3)
  And since padding is accomplished using `jraph.pad_with_graphs`, an extra
  graph and node is added:
    8 nodes --> 9 nodes
    3 graphs --> 4 graphs
  Args:
    graphs_tuple: a batched `GraphsTuple` (can be batch size 1).
  Returns:
    A graphs_tuple batched to the nearest power of two.
  Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
  """
  # Add 1 since we need at least one padding node for pad_with_graphs.
  pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
  pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
  return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                               pad_graphs_to)


def pad_all(ds: LabelledGraphs) -> LabelledGraphs:
  """Pads all graphs in a dataset to the nearest power of two."""
  # Jax will re-jit your graphnet every time a new graph shape is encountered.
  # In the limit, this means a new compilation every training step, which
  # will result in *extremely* slow training. To prevent this, pad each
  # batch of graphs to the nearest power of two. Since jax maintains a cache
  # of compiled programs, the compilation cost is amortized.

  padded = []
  for graph, target in ds:
    graph = pad_graph_to_nearest_power_of_two(graph)
    # Since padding is implemented with pad_with_graphs, an extra graph has
    # been added to the batch, which means there should be an extra label.
    target = jnp.concatenate([target, jnp.array([0])])
    padded.append((graph, target))
  return padded


def create_optimizer(
  hyper_params: Dict[str, Any]) -> optax.GradientTransformation:

  # TODO: replace learning_rate with a scheduler that uses appropriate hyper parameters
  # TODO: understand what params "weight decay" is
  return optax.adam(learning_rate=hyper_params["init_lr"])
