from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph

from utils import HaikuDebug
import masked


class GatedGCNLayer(hk.Module):

  def __init__(self, output_dim, weight_on_edges=True,
               residual=True, dropout=0.0, mask_batch_norm=True, debug=False, name: Optional[str] = None):
    super().__init__(name=name)
    self.output_dim = output_dim
    self.weight_on_edges = weight_on_edges
    self.residual = residual
    self.dropout = dropout
    self.mask_batch_norm = mask_batch_norm
    self.debug = debug

  def __call__(self, graph: jraph.GraphsTuple,
               is_training=True) -> jraph.GraphsTuple:
    output_dim = self.output_dim
    weight_on_edges = self.weight_on_edges
    residual = self.residual
    dropout = self.dropout
    debug = self.debug

    """Applies a GatedGCN layer."""

    # TODO: find another paper perhaps, list paper
    A = hk.Linear(output_dim, name="i_multiplication_edge_logits")
    B = hk.Linear(output_dim, name="j_multiplication_edge_logits")
    C = hk.Linear(output_dim, name="edge_multiplication_edge_logits")
    U = hk.Linear(output_dim, name="i_multiplication_node")
    V = hk.Linear(output_dim, name="j_multiplication_node")

    nodes, edges, receivers, senders, _, _, _ = graph
    h = nodes['feat']
    e = edges['feat']
    # TODO: Do we only care about undirected graph
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

    # TODO: Check if batch norm parameters are the same as those of pytorch
    if self.mask_batch_norm:
      edge_mask = jraph.get_edge_padding_mask(graph)
      edge_layer_features = masked.BatchNorm(
          create_scale=True,
          create_offset=True,
          decay_rate=0.9, name="masked_batch_norm_edge")(eta, edge_mask, is_training)
    else:
      edge_layer_features = hk.BatchNorm(
          create_scale=True,
          create_offset=True,
          decay_rate=0.9, name="batch_norm_edge")(eta, is_training)
    edge_layer_features = jax.nn.relu(edge_layer_features)
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

    unattnd_messages = V(h[j]) * w_sigma
    agg_unattnd_messages = jax.ops.segment_sum(
        unattnd_messages, i, num_segments=sum_n_node)  # TODO: i or j?
    node_layer_features = U(h) + agg_unattnd_messages / w_sigma_sum
    HaikuDebug("node_before", enable=debug)(node_layer_features)
    if self.mask_batch_norm:
      node_mask = jraph.get_node_padding_mask(graph)
      node_layer_features = masked.BatchNorm(
          create_scale=True,
          create_offset=True,
          decay_rate=0.9, name="masked_batch_norm_node")(node_layer_features, node_mask, is_training)
    else:
      node_layer_features = hk.BatchNorm(
          create_scale=True,
          create_offset=True,
          decay_rate=0.9, name="batch_norm_node")(node_layer_features, is_training)

    node_layer_features = jax.nn.relu(node_layer_features)
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
