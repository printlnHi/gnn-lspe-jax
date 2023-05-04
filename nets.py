from typing import Any, Callable, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import jraph

# TODO: Should I rename to not be cased like a class?
from layers.GatedGCNLayer import GatedGCNLayer
from layers.GatedGCNLSPELayer import GatedGCNLSPELayer
from layers.mlp_readout_layer import mlp_readout
from types_and_aliases import GraphClassifierInput, GraphClassifierOutput
from utils import HaikuDebug
from multi_embedder import MultiEmbedder


def gated_gcn_net(net_params, h_encoder, e_encoder, task_out_dim, graph: jraph.GraphsTuple,
                  is_training: bool, debug: bool = False) -> GraphClassifierOutput:
  in_feat_dropout = net_params['in_feat_dropout']
  hidden_dim = net_params['hidden_dim']
  out_dim = net_params['out_dim']
  dropout = net_params['dropout']
  n_layers = net_params['L']
  readout = net_params['readout']
  graph_norm = net_params['graph_norm']
  residual = net_params['residual']
  pe_init = net_params['pe_init']
  mask_batch_norm = net_params['mask_batch_norm']
  weight_on_edges = net_params['weight_on_edges']

  pos_enc_dim = net_params['pos_enc_dim']
  """A gatedGCN model."""
  nodes, edges, receivers, senders, globals, n_node, n_edge = graph
  HaikuDebug("input_graph", enable=debug)(graph)
  sum_n_node = nodes['feat'].shape[0]
  num_graphs = n_node.shape[0]

  HaikuDebug("unencoded_h", enable=debug)(nodes['feat'])
  h = h_encoder(nodes['feat'])
  HaikuDebug("h_encoded", enable=debug)(h)
  if is_training:
    h = hk.dropout(hk.next_rng_key(), in_feat_dropout, h)

  if pe_init == "lap_pe":
    # Combine the node features and the Laplacian PE in the embedding space
    h += hk.Linear(hidden_dim, name="pe_embedding")(nodes['pe'])
    pass
  elif pe_init == "rand_walk":
    p = hk.Linear(hidden_dim, name="pe_embedding")(nodes['pe'])
    nodes = nodes | {'pos': p}

  e = e_encoder(edges['feat'])
  HaikuDebug("e_encoded", enable=debug)(e)

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
                'dropout': dropout, 'mask_batch_norm': mask_batch_norm, 'graph_norm': graph_norm, 'weight_on_edges': weight_on_edges}
  final_layer_args = layer_args | {'output_dim': out_dim}

  HaikuDebug("before_layers_updated_graph", enable=debug)(updated_graph)
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
    hg = jraph.segment_max(h, graph_indicies, num_segments=num_graphs)
  else:
    # mean
    hg = jraph.segment_mean(h, graph_indicies, num_segments=num_graphs)
  hg = jnp.nan_to_num(hg)
  HaikuDebug("hg", enable=debug)(hg)
  mlp_result = mlp_readout(hg, input_dim=out_dim, output_dim=task_out_dim)
  HaikuDebug("mlp_result", enable=debug)(mlp_result)

  updated_graph = updated_graph._replace(globals=hg, nodes=nodes)
  return jnp.squeeze(mlp_result), updated_graph


def zinc_model(task_dims: Dict[str, Any], net_params: Dict[str, Any],
               debug: bool = False) -> Callable[GraphClassifierInput, GraphClassifierOutput]:

  assert(len(task_dims['atom']) == 1) and (len(task_dims['bond']) == 1)
  num_atom_type = task_dims['atom'][0]
  num_bond_type = task_dims['bond'][0]
  hidden_dim = net_params['hidden_dim']

  def net(graph: jraph.GraphsTuple,
          is_training: bool) -> GraphClassifierOutput:
    h_encoder = hk.Embed(vocab_size=num_atom_type, embed_dim=hidden_dim)
    e_encoder = hk.Embed(vocab_size=num_bond_type, embed_dim=hidden_dim)
    task_out_dim = 1
    return gated_gcn_net(net_params, h_encoder, e_encoder, task_out_dim, graph, is_training, debug=debug)
  return net


def moltox21_model(task_dims: Dict[str, Any], net_params: Dict[str, Any], debug: bool = False) -> Callable[GraphClassifierInput, GraphClassifierOutput]:

  glorot_uniform = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
  hidden_dim = net_params['hidden_dim']
  task_out_dim = task_dims['classes']

  def net(graph: jraph.GraphsTuple, is_training: bool) -> GraphClassifierOutput:
    h_encoder = MultiEmbedder(
      task_dims['atom'], hidden_dim, w_init=glorot_uniform)
    e_encoder = MultiEmbedder(
      task_dims['bond'], hidden_dim, w_init=glorot_uniform)
    return gated_gcn_net(net_params, h_encoder, e_encoder, task_out_dim, graph, is_training, debug=debug)
  return net
