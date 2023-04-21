from typing import Any, Callable, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import jraph

# TODO: Should I rename to not be cased like a class?
from layers.GatedGCNLayer_hk import GatedGCNLayer
from layers.GatedGCNLSPELayer import GatedGCNLSPELayer
from layers.mlp_readout_layer import mlp_readout
from type_aliases import GraphClassifierInput, GraphClassifierOutput
from utils import HaikuDebug

g_init, g_apply = hk.transform_with_state(GatedGCNLayer)


def gnn_model(net_params: Dict[str, Any],
              debug: bool = False) -> Callable[GraphClassifierInput, GraphClassifierOutput]:
  num_atom_type = net_params['num_atom_type']
  num_bond_type = net_params['num_bond_type']
  hidden_dim = net_params['hidden_dim']
  out_dim = net_params['out_dim']
  in_feat_dropout = net_params['in_feat_dropout']
  dropout = net_params['dropout']
  n_layers = net_params['L']
  readout = net_params['readout']
  batch_norm = net_params['batch_norm']
  graph_norm = net_params['graph_norm']
  residual = net_params['residual']
  edge_feat = net_params['edge_feat']
  pe_init = net_params['pe_init']
  mask_batch_norm = net_params['mask_batch_norm']

  pos_enc_dim = net_params['pos_enc_dim']

  def gated_gcn_net(graph: jraph.GraphsTuple,
                    is_training: bool) -> GraphClassifierOutput:
    """A gatedGCN model."""
    nodes, edges, receivers, senders, globals, n_node, n_edge = graph
    sum_n_node = nodes['feat'].shape[0]
    num_graphs = n_node.shape[0]

    h = hk.Embed(vocab_size=num_atom_type, embed_dim=hidden_dim)(nodes['feat'])
    if is_training:
      h = hk.dropout(hk.next_rng_key(), in_feat_dropout, h)

    if pe_init == "lap_pe":
      # Combine the node features and the Laplacian PE in the embedding space
      h += hk.Linear(hidden_dim, name="pe_embedding")(nodes['pe'])
      pass
    elif pe_init == "rand_walk":
      p = hk.Linear(hidden_dim, name="pe_embedding")(nodes['pe'])
      nodes = nodes | {'pos': p}

    if edge_feat:
      embedding_e = hk.Embed(vocab_size=num_bond_type, embed_dim=hidden_dim)
    else:
      embedding_e = hk.Linear(hidden_dim)
      edges = {'feat': jnp.ones([n_edge, 1])}
    e = embedding_e(edges['feat'])

    nodes = nodes | {'feat': h}
    edges = edges | {'feat': e}
    updated_graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        receivers=receivers,
        senders=senders,
        n_node=n_node,
        n_edge=n_edge,
        globals=globals)

    layer_args = {'output_dim': hidden_dim, 'residual': residual,
                  'dropout': dropout, 'mask_batch_norm': mask_batch_norm, 'graph_norm': graph_norm}
    final_layer_args = layer_args | {'output_dim': out_dim}

    if pe_init == 'rand_walk':
      for _ in range(n_layers - 1):
        updated_graph = GatedGCNLSPELayer(
          **layer_args)(updated_graph, is_training=is_training)
      updated_graph = GatedGCNLSPELayer(
        **final_layer_args)(updated_graph, is_training=is_training)
    else:
      for _ in range(n_layers - 1):
        updated_graph = GatedGCNLayer(
          **layer_args)(updated_graph, is_training=is_training)
      updated_graph = GatedGCNLayer(
        **final_layer_args)(updated_graph, is_training=is_training)

    nodes, edges, _, _, _, _, _ = updated_graph

    HaikuDebug("updated_graph", enable=debug)(updated_graph)
    h = nodes['feat']

    graph_indicies = jnp.repeat(
        jnp.arange(
            n_node.shape[0]),
        n_node,
        total_repeat_length=sum_n_node)

    if pe_init == 'rand_walk':
      p = nodes['pos']
      p = hk.Linear(pos_enc_dim, name="pe_out")(p)
      p = jraph.segment_normalize(p, graph_indicies, num_segments=num_graphs)
      h = hk.Linear(out_dim, name="Whp")(jnp.concatenate([h, p], axis=1))
      nodes = nodes | {'final_p': p, 'final_h': h}

    # readout
    HaikuDebug("graph_indicies", enable=debug)(graph_indicies)
    if readout == 'sum':
      hg = jraph.segment_sum(h, graph_indicies, num_segments=num_graphs)
    elif readout == 'max':
      # TODO: Should consider using aux value of -inf instead of 0 for max
      hg = jraph.segment_max(h, graph_indicies, num_segments=num_graphs)
    else:
      # mean
      hg = jraph.segment_mean(h, graph_indicies, num_segments=num_graphs)
    hg = jnp.nan_to_num(hg)
    HaikuDebug("hg", enable=debug)(hg)
    mlp_result = mlp_readout(hg, input_dim=out_dim, output_dim=1)
    HaikuDebug("mlp_result", enable=debug)(mlp_result)

    updated_graph = updated_graph._replace(globals=hg, nodes=nodes)
    return jnp.squeeze(mlp_result), updated_graph

  return gated_gcn_net
