import jax
import jraph
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Any, Callable
from layers import GatedGCNLayer, mlp_readout_layer

NetFnType = Callable[[jraph.GraphsTuple, bool], jraph.GraphsTuple]


def gnn_model(net_params: Dict[str, Any]) -> NetFnType:
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
                    is_training: bool) -> jraph.GraphsTuple:
    """A gatedGCN model."""
    nodes, edges, receivers, senders, n_node, n_edge, globals = graph

    h = hk.Embed(vocab_size=num_atom_type, embed_dim=hidden_dim)(nodes)
    h = hk.Dropout(keep_prob=0.9)

    if pe_init in ['rand_walk', 'lap_pe']:
      hk.Linear(hidden_dim)  # embedding_p to be applied
      raise NotImplementedError("Don't have a concept of node features p yet")

    if pe_init == 'lap_pe':
      #h = h + p
      raise NotImplementedError("Don't have a concept of node features p yet")

    if edge_feat:
      embedding_e = hk.Embed(vocab_size=num_bond_type, embed_dim=hidden_dim)
    else:
      embedding_e = hk.Linear(hidden_dim)
      edges = jnp.ones([n_edge, 1])
    e = embedding_e(edges)

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
    updated_graph = jraph.GraphTuple(
        nodes=h,
        edges=e,
        receivers=receivers,
        senders=senders,
        n_node=n_node,
        n_edge=n_edge,
        globals=globals)
    updated_graph = layers(updated_graph, is_training=is_training)
    nodes, edges, receivers, senders, n_node, n_edge, globals = updated_graph

    if pe_init == 'rand_walk':
      p_out = hk.Linear(pos_enc_dim)
      Whp = hk.Linear(out_dim)
      '''
            p = self.p_out(p)
            ...'''
      raise NotImplementedError("Don't have a concept of node features p yet")

    # readout
    node_features = jnp.split(nodes, jnp.cumsum(n_node)[:-1])
    reduction = {'sum': jnp.sum, 'max': jnp.max, 'mean': jnp.mean}.get(
      readout, jnp.mean)  # default readout is mean
    hg = reduction(node_features, axis=0)

    return mlp_readout_layer(hg, input_dim=out_dim, output_dim=1), g
