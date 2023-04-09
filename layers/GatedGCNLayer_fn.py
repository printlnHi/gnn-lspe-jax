from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph

from utils import HaikuDebug


def GatedGCNLayer(graph: jraph.GraphsTuple, output_dim, weight_on_edges=True, residual=True,
                  dropout=0.0, debug=False, is_training=True) -> jraph.GraphsTuple:
  """Returns a method that applies a GatedGCN layer.

  Args:
    output_dim: the dimension of output node and edge feature vectors
    weight_on_edges: if True the soft attention for nodes is over on edge outputs, otherwise over the intermediate eta values
    residual: whether we have a residual connection, as the original gated GCN did (TODO: Verify). Requires output_dim=input_dim
    dropout: the dropout rate to apply on node and edge outputs
  Returns:
    A function that applies a GatedGCN layer.
  """

  # TODO: find another paper perhaps, list paper
  A = hk.Linear(output_dim, name="i_multiplication_edge_logits")
  B = hk.Linear(output_dim, name="j_multiplication_edge_logits")
  C = hk.Linear(output_dim, name="edge_multiplication_edge_logits")
  U = hk.Linear(output_dim, name="i_multiplication_node")
  V = hk.Linear(output_dim, name="j_multiplication_node")
  batch_norm_edge = hk.BatchNorm(
      create_scale=True,
      create_offset=True,
      decay_rate=0.9, name="batch_norm_edge")
  batch_norm_node = hk.BatchNorm(
      create_scale=True,
      create_offset=True,
      decay_rate=0.9, name="batch_norm_node")
  # TODO: Check if batch norm parameters are the same as those of pytorch

  nodes, edges, receivers, senders, _, _, _ = graph
  h = nodes['feat']
  e = edges['feat']
  # Does this sender<-> i receiver<->j correspondence make sense? Do we only
  # care about undirected graph
  i = senders
  j = receivers

  # Equivalent to the sum of n_node, but statically known.
  try:
    sum_n_node = h.shape[0]
  except IndexError:
    raise IndexError('GatedGCN requires node features')
  assert(edges is not None)

  if residual and h.shape[1] != output_dim:
    raise ValueError(f"For residual connections the output_dim ({output_dim}) must match"
                     f"the node feature dimension {h.shape[1]}")
  if residual and e.shape[1] != output_dim:
    raise ValueError(f"For residual connections the output_dim ({output_dim}) must match"
                     f"the edge feature dimension {e.shape[1]}")

  assert(
      not residual or (
          h.shape[1] == output_dim and e.shape[1] == output_dim))

  # Pass nodes through the attention query function to transform
  # node features, e.g. with an MLP.
  total_num_nodes = tree.tree_leaves(h)[0].shape[0]

  eta = A(h[i]) + B(h[j]) + C(e)
  HaikuDebug("eta", enable=debug)(eta)
  edge_layer_features = jax.nn.relu(batch_norm_edge(eta, is_training))
  HaikuDebug("edge_layer_features", enable=debug)(edge_layer_features)

  if residual:
    e = e + edge_layer_features
  else:
    e = edge_layer_features

  if weight_on_edges:
    w_sigma = jax.nn.sigmoid(e)
  else:
    w_sigma = jax.nn.sigmoid(eta)

  w_sigma_sum = jax.ops.segment_sum(
      w_sigma, segment_ids=i, num_segments=sum_n_node) + 1e-6

  # TODO: check this is Hadamard product
  unattnd_messages = V(h[j]) * w_sigma
  agg_unattnd_messages = jax.ops.segment_sum(
      unattnd_messages, i, num_segments=sum_n_node)  # i or j?
  node_layer_features = U(h) + agg_unattnd_messages / w_sigma_sum
  HaikuDebug("node_before", enable=debug)(node_layer_features)
  node_layer_features = jax.nn.relu(
    batch_norm_node(node_layer_features, is_training))
  HaikuDebug("node_layer_features", enable=debug)(node_layer_features)

  if residual:
    h = h + node_layer_features
  else:
    h = node_layer_features

  if is_training:
    h = hk.dropout(hk.next_rng_key(), dropout, h)
    e = hk.dropout(hk.next_rng_key(), dropout, e)

  nodes = dict(nodes | {'feat': h})
  edges = dict(edges | {'feat': e})
  output = graph._replace(nodes=nodes, edges=edges)
  HaikuDebug("output", enable=debug)(output)
  return output
