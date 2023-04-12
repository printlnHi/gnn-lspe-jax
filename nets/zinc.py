from typing import Any, Callable, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import jraph

# TODO: Should I rename to not be cased like a class?
# from layers.GatedGCNLayer import GatedGCNLayer
#from layers.GatedGCNLayer_fn import GatedGCNLayer
from layers.GatedGCNLayer_hk import GatedGCNLayer
from layers.mlp_readout_layer import mlp_readout
from type_aliases import GraphClassifierFn
from utils import HaikuDebug

g_init, g_apply = hk.transform_with_state(GatedGCNLayer)


def gnn_model(net_params: Dict[str, Any],
              debug: bool = False) -> GraphClassifierFn:
  num_atom_type = net_params['num_atom_type']
  num_bond_type = net_params['num_bond_type']
  hidden_dim = net_params['hidden_dim']
  out_dim = net_params['out_dim']
  in_feat_dropout = net_params['in_feat_dropout']
  dropout = net_params['dropout']
  n_layers = net_params['L']
  readout = net_params['readout']
  batch_norm = net_params['batch_norm']
  residual = net_params['residual']
  edge_feat = net_params['edge_feat']
  # device = net_params['device'] #TODO: What do I do with this?
  pe_init = net_params['pe_init']

  use_lapeig_loss = net_params['use_lapeig_loss']
  lambda_loss = net_params['lambda_loss']
  alpha_loss = net_params['alpha_loss']

  pos_enc_dim = net_params['pos_enc_dim']

  def gated_gcn_net(graph: jraph.GraphsTuple,
                    is_training: bool) -> jnp.ndarray:
    """A gatedGCN model."""
    nodes, edges, receivers, senders, globals, n_node, n_edge = graph
    sum_n_node = nodes['feat'].shape[0]
    num_graphs = n_node.shape[0]

    h = hk.Embed(vocab_size=num_atom_type, embed_dim=hidden_dim)(nodes['feat'])
    if is_training:
      h = hk.dropout(hk.next_rng_key(), in_feat_dropout, h)

    if pe_init in ['rand_walk', 'lap_pe']:
      hk.Linear(hidden_dim)  # embedding_p to be applied
      raise NotImplementedError("Don't have a concept of node features p yet")

    if pe_init == 'lap_pe':
      # h = h + p
      raise NotImplementedError("Don't have a concept of node features p yet")

    if edge_feat:
      embedding_e = hk.Embed(vocab_size=num_bond_type, embed_dim=hidden_dim)
    else:
      embedding_e = hk.Linear(hidden_dim)
      edges = {'feat': jnp.ones([n_edge, 1])}
    e = embedding_e(edges['feat'])

    if pe_init == 'rand_walk':
      # LSPE
      raise NotImplementedError(
        "Random walk PE nor GatedGCNLSPELayer implemented yet")
    else:
      # NoPE or LapPE
      '''      layers = [
          GatedGCNLayer(
              hidden_dim,
              residual=residual,
              dropout=dropout) for _ in range(
              n_layers - 1)]
      layers.append(GatedGCNLayer(out_dim, residual=residual, dropout=dropout))
      layers = np.array(layers)
      '''

    # TODO: Will have to update this to propogate p features
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

    '''for layer in layers:
      updated_graph = layer(updated_graph, is_training=is_training)'''
    for _ in range(n_layers - 1):
      updated_graph = GatedGCNLayer(output_dim=hidden_dim, residual=residual, dropout=dropout,
                                    )(updated_graph, is_training=is_training)
    updated_graph = GatedGCNLayer(
        output_dim=out_dim,
        residual=residual,
        dropout=dropout)(
        updated_graph,
        is_training=is_training)

    nodes, edges, _, _, _, _, _ = updated_graph
    HaikuDebug("update_graph", enable=debug)(updated_graph)
    h = nodes['feat']

    if pe_init == 'rand_walk':
      p_out = hk.Linear(pos_enc_dim)
      Whp = hk.Linear(out_dim)
      '''
            p = self.p_out(p)
            ...'''
      raise NotImplementedError("Don't have a concept of node features p yet")

    # readout
    graph_indicies = jnp.repeat(
        jnp.arange(
            n_node.shape[0]),
        n_node,
        total_repeat_length=sum_n_node)
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
    return jnp.squeeze(mlp_result)

  return gated_gcn_net
