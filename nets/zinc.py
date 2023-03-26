from typing import Any, Callable, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import jraph

# TODO: Should I rename to not be cased like a class?
from layers.GatedGCNLayer import GatedGCNLayer
from layers.mlp_readout_layer import mlp_readout
from type_aliases import GraphClassifierFn


def gnn_model(net_params: Dict[str, Any]) -> GraphClassifierFn:
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

    h = hk.Embed(vocab_size=num_atom_type, embed_dim=hidden_dim)(nodes['feat'])
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
      layers = [
          GatedGCNLayer(
              hidden_dim,
              residual=residual,
              dropout=dropout) for _ in range(
              n_layers - 1)]
      layers.append(GatedGCNLayer(out_dim, residual=residual, dropout=dropout))
      layers = hk.Sequential(layers)

    # TODO: Will have to update this to propogate p features
    nodes.update({'feat': h})
    edges.update({'feat': e})
    updated_graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        receivers=receivers,
        senders=senders,
        n_node=n_node,
        n_edge=n_edge,
        globals=globals)
    updated_graph = layers(updated_graph, is_training=is_training)
    nodes, edges, _, _, _, _, _ = updated_graph

    if pe_init == 'rand_walk':
      p_out = hk.Linear(pos_enc_dim)
      Whp = hk.Linear(out_dim)
      '''
            p = self.p_out(p)
            ...'''
      raise NotImplementedError("Don't have a concept of node features p yet")

    # readout
    node_features = jnp.stack(
        jnp.split(
            nodes['feat'],
            jnp.cumsum(n_node)[
                :-1]),
        axis=0)
    reduction = {'sum': jnp.sum, 'max': jnp.max, 'mean': jnp.mean}.get(
      readout, jnp.mean)  # default readout is mean
    hg = reduction(node_features, axis=0)

    return mlp_readout(hg, input_dim=out_dim, output_dim=1)

  return gated_gcn_net
