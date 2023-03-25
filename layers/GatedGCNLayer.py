import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
from typing import Callable

def GatedGCNLayer(output_dim, residual=True, dropout=0.0) -> Callable:
  """Returns a method that applies a GatedGCN layer.

  Args:
    output_dim: the dimension of output node and edge feature vectors
    residual: whether we have a residual connection, as the original gated GCN did (TODO: Verify). Requires output_dim=input_dim 

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
    Ah = A(hi)
    Bh = B(hj)
    Ce = C(edges)

    edge_layer_features = Ah + Bh + Ce
    edge_layer_features = jax.nn.relu(batch_norm_edge(edge_layer_features, is_training))

    if residual:
      edges = edges + edge_layer_features
    else:
      edges = edge_layer_features


    #GNN-LSPE does batch norm after softmaxing!
    attn_softmax_logits = jax.nn.sigmoid(edges)
    #Does this work for multiple dimensions? I think so
    #The GNN-LSPE repo has an epsilon of 1e-6, what repo does 
    attn_weights = jraph.segment_softmax(
        attn_softmax_logits, segment_ids=i, num_segments=sum_n_node)

    Uh = U(nodes)
    Vh = V(hj)
    messages = Vh * attn_weights #TODO: check this is Hadamard product?

    node_layer_features = Uh + jax.ops.segment_sum(messages, i, num_segments=sum_n_node) # i or j?
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