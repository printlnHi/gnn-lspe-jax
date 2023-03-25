import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
from typing import Callable

def GatedGCNLayer(output_dim, weight_on_edges=True, residual=True, dropout=0.0) -> Callable:
  """Returns a method that applies a GatedGCN layer.

  Args:
    output_dim: the dimension of output node and edge feature vectors
    weight_on_edges: if True the soft attention for nodes is over on edge outputs, otherwise over the intermediate eta values
    residual: whether we have a residual connection, as the original gated GCN did (TODO: Verify). Requires output_dim=input_dim 
    dropout=0.0: 
  Returns:
    A function that applies a GatedGCN layer.
  """
  

  def _ApplyGatedGCN(graph: jraph.GraphsTuple, is_training=True) -> jraph.GraphsTuple:
    """Applies a GatedGCN layer."""

    #TODO: find another paper perhaps, list paper
    A = hk.Linear(output_dim, name="i_multiplication_edge_logits")
    B = hk.Linear(output_dim, name="j_multiplication_edge_logits")
    C = hk.Linear(output_dim, name="edge_multiplication_edge_logits")
    U = hk.Linear(output_dim, name="i_multiplication_node")
    V = hk.Linear(output_dim, name="j_multiplication_node")
    batch_norm_edge = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)
    batch_norm_node = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)
    #TODO: Check if batch norm parameters are the same as those of pytorch 

    nodes, edges, receivers, senders, _ , _, _ = graph
    #Does this sender<-> i receiver<->j correspondence make sense? Do we only care about undirected graph
    i = senders
    j = receivers

    # Equivalent to the sum of n_node, but statically known.
    try:
      sum_n_node = nodes.shape[0]
    except IndexError:
      raise IndexError('GatedGCN requires node features')
    assert(edges is not None)

    if residual and nodes.shape[1]!=output_dim:
      raise ValueError(f"For residual connections the output_dim ({output_dim}) must match"
      f"the node feature dimension {nodes.shape[1]}")
    if residual and edges.shape[1]!=output_dim:
      raise ValueError(f"For residual connections the output_dim ({output_dim}) must match"
      f"the edge feature dimension {edges.shape[1]}")


    assert(not residual or (nodes.shape[1]==output_dim and edges.shape[1]==output_dim))

    # Pass nodes through the attention query function to transform
    # node features, e.g. with an MLP.
    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
    print(f"GatedGCN total_num_nodes: {total_num_nodes}")
    hi = nodes[i]
    hj = nodes[j]

    eta = A(hi) + B(hj) + C(edges)
    edge_layer_features = jax.nn.relu(batch_norm_edge(eta, is_training))

    if residual:
      edges = edges + edge_layer_features
    else:
      edges = edge_layer_features

    if weight_on_edges:
      w_sigma = jax.nn.sigmoid(edges)
    else:
      w_sigma = jax.nn.sigmoid(eta)
                               
    w_sigma_sum = jax.ops.segment_sum(w_sigma, segment_ids=i, num_segments=sum_n_node) + 1e-6

    unattnd_messages = V(hj) * w_sigma #TODO: check this is Hadamard product
    agg_unattnd_messages = jax.ops.segment_sum(unattnd_messages, i, num_segments=sum_n_node) # i or j?
    node_layer_features = U(nodes) + agg_unattnd_messages / w_sigma_sum
    node_layer_features = jax.nn.relu(batch_norm_node(node_layer_features, is_training))

    if residual:
     nodes = nodes + node_layer_features
    else:
      nodes = node_layer_features

    if is_training:
      nodes = hk.dropout(hk.next_rng_key(), dropout, nodes)
      edges = hk.dropout(hk.next_rng_key(), dropout, edges)
    return graph._replace(nodes=nodes, edges=edges)

  # pylint: enable=g-long-lambda  
  return _ApplyGatedGCN