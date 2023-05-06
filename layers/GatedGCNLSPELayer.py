from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph

import masked
from lib.debug import HaikuDebug


class GatedGCNLSPELayer(hk.Module):

  def __init__(self, output_dim, weight_on_edges=True,
               residual=True, dropout=0.0, mask_batch_norm=True, graph_norm=True, debug=False, name: Optional[str] = None):
    super().__init__(name=name)
    self.output_dim = output_dim
    self.weight_on_edges = weight_on_edges
    self.residual = residual
    self.dropout = dropout
    self.mask_batch_norm = mask_batch_norm
    self.graph_norm = graph_norm
    self.debug = debug

  def __call__(self, graph: jraph.GraphsTuple,
               is_training=True) -> jraph.GraphsTuple:
    output_dim = self.output_dim
    weight_on_edges = self.weight_on_edges
    residual = self.residual
    dropout = self.dropout
    debug = self.debug

    """Applies a GatedGCN-LSPE layer."""

    A = hk.Linear(output_dim, name="i_multiplication_edge_logits")
    B = hk.Linear(output_dim, name="j_multiplication_edge_logits")
    C = hk.Linear(output_dim, name="edge_multiplication_edge_logits")
    U = hk.Linear(output_dim, name="i_multiplication_node")
    V = hk.Linear(output_dim, name="j_multiplication_node")
    X = hk.Linear(output_dim, name="i_multiplication_positional")
    Y = hk.Linear(output_dim, name="j_multiplication_positional")

    nodes, edges, receivers, senders, _, _, _ = graph
    h = nodes['feat']
    p = nodes['pos']
    e = edges['feat']
    i = senders
    j = receivers

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

    # TODO: Turn to jraph
    w_sigma_sum = jax.ops.segment_sum(
        w_sigma, segment_ids=i, num_segments=sum_n_node) + 1e-6

    unattnd_messages = V(jnp.concatenate([h[j], p[j]], axis=1)) * w_sigma
    agg_unattnd_messages = jax.ops.segment_sum(
        unattnd_messages, i, num_segments=sum_n_node)
    node_layer_features = U(jnp.concatenate(
      [h, p], axis=1)) + agg_unattnd_messages / w_sigma_sum
    if self.graph_norm:
      node_layer_features *= nodes['snorm_n'][:, None]
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

    unattnd_messages = Y(p[j]) * w_sigma
    agg_unattnd_messages = jax.ops.segment_sum(
        unattnd_messages, i, num_segments=sum_n_node)
    pos_layer_features = jax.nn.tanh(X(p) + agg_unattnd_messages / w_sigma_sum)
    HaikuDebug("pos_layer_features", enable=debug)(pos_layer_features)

    if residual:
      h = h + node_layer_features
      p = p + pos_layer_features
    else:
      h = node_layer_features
      p = pos_layer_features

    if is_training:
      h = hk.dropout(hk.next_rng_key(), dropout, h)
      e = hk.dropout(hk.next_rng_key(), dropout, e)
      p = hk.dropout(hk.next_rng_key(), dropout, p)

    nodes = dict(nodes | {'feat': h} | {'pos': p})
    edges = dict(edges | {'feat': e})
    output = graph._replace(nodes=nodes, edges=edges)
    HaikuDebug("output", enable=debug)(output)
    return output
