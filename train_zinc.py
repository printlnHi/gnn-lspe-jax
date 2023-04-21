import functools
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import jax.lax

from type_aliases import LabelledGraph, Metrics, TrainResult
from utils import graphLaplacian


def compute_loss(net: hk.TransformedWithState, params: hk.Params, state: hk.State,
                 batch: LabelledGraph, rng: jax.random.KeyArray, is_training: bool) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, hk.State]]:
  """Compute the loss for a given dataset."""
  graph, label = batch
  (scores, _graph), state = net.apply(
    params, state, rng, graph, is_training=is_training)
  mask = jraph.get_graph_padding_mask(graph)
  # L1 loss
  loss = jnp.sum(jnp.abs(scores - label) * mask) / jnp.sum(mask)
  return loss, (loss, state)


def compute_lapeig_inclusive_loss(net: hk.TransformedWithState, net_params, batch_size:int, params: hk.Params, state: hk.State,
                                  batch: LabelledGraph, rng: jax.random.KeyArray, is_training: bool) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, hk.State]]:

  graph, label = batch
  (scores, graph), state = net.apply(
      params, state, rng, graph, is_training=is_training)

  graph_mask = jraph.get_graph_padding_mask(graph)
  node_mask = jraph.get_node_padding_mask(graph)
  num_graphs = jnp.sum(graph_mask)
  num_nodes = jnp.sum(node_mask)

  task_loss = jnp.sum(jnp.abs(scores - label) * graph_mask) / num_graphs

  pos_enc_dim = net_params["pos_enc_dim"]
  alpha_loss = net_params["alpha_loss"]
  lambda_loss = net_params["lambda_loss"]
  p = graph.nodes['final_p']
  p *= node_mask[:, None]
  pT = jnp.transpose(p)

  L = graphLaplacian(graph, np_=jnp)
  trace = jnp.trace(pT @ L @ p)

  graph_indicies = jnp.concatenate([jnp.array([0]), jnp.cumsum(graph.n_node)])
  node_indicies = jnp.arange(graph.nodes['feat'].shape[0])

  def frob_norm_squared(i):
    mask = (node_indicies >= graph_indicies[i]) * \
        (node_indicies < graph_indicies[i + 1])
    pMasked = p * mask[:, None]
    frob_norm_squared = jnp.sum(jnp.square(
        jnp.transpose(pMasked) @ pMasked - jnp.eye(pos_enc_dim)))
    return frob_norm_squared * (i < num_graphs)

  frob_norms_squared_sum = jnp.sum(jax.lax.map(
    frob_norm_squared, jnp.arange(batch_size)))

  positional_loss = (trace + lambda_loss * frob_norms_squared_sum) / \
      (pos_enc_dim * num_graphs * num_nodes)
  loss = task_loss + alpha_loss * positional_loss

  return loss, (task_loss, state)


def train_epoch(loss_and_grad_fn, opt_update: optax.TransformUpdateFn, opt_apply_updates, pe_init, params: hk.Params, state: hk.State, rng: jax.random.KeyArray,
                opt_state: optax.OptState, dataloader) -> TrainResult:
  """Train for one epoch."""
  epoch_start_time = time.time()
  losses = []
  lengths = []
  MAEs = []
  loss_and_grad_times = []
  opt_update_times = []
  opt_apply_times = []

  rng, subkey = jax.random.split(rng)
  batches = list(dataloader(subkey))
  del subkey
  subkeys = jax.random.split(rng, 2 * len(batches)).reshape(-1, 2, 2)
  dataset_time = time.time() - epoch_start_time

  for (batch, length), subkey in zip(batches, subkeys):
    # As we've already produced the whole dataset time between iterations is
    # negligible
    if pe_init == "lap_pe":
      flip = jax.random.bernoulli(
          subkey[0], shape=batch[0].nodes['pe'].shape) * 2 - 1
      batch[0].nodes['pe'] = batch[0].nodes['pe'] * flip

    batch_start = time.time()
    (loss, (MAE, state)), grads = loss_and_grad_fn(
      params, state, batch, subkey[1])
    loss_end = time.time()

    updates, opt_state = opt_update(grads, opt_state, params)
    opt_update_end = time.time()

    params = opt_apply_updates(params, updates)
    opt_end = time.time()

    losses.append(loss)
    MAEs.append(MAE)
    lengths.append(length)
    loss_and_grad_times.append(loss_end - batch_start)
    opt_update_times.append(opt_update_end - loss_end)
    opt_apply_times.append(opt_end - opt_update_end)

  losses = jnp.asarray(losses)
  MAEs = jnp.asarray(MAEs)
  lengths = jnp.asarray(lengths)
  loss = jnp.sum(losses * lengths) / jnp.sum(lengths)
  MAE = jnp.sum(MAEs * lengths) / jnp.sum(lengths)

  loss_and_grad_times = jnp.asarray(loss_and_grad_times)
  opt_update_times = jnp.asarray(opt_update_times)
  opt_apply_times = jnp.asarray(opt_apply_times)
  total_batch_times = loss_and_grad_times + opt_update_times + opt_apply_times

  time_metrics = {
      "total_epoch_time_ALT": time.time() - epoch_start_time,
      "dataset_time": dataset_time}
  for times, name in zip([loss_and_grad_times, opt_update_times, opt_apply_times, total_batch_times], [
                         "loss_and_grad_times", "opt_update_times", "opt_apply_times", "total_batch_times", "pre_iter_times"]):
    time_metrics[name[:-1]] = float(jnp.sum(times))

  metrics = {"loss": float(loss), "MAE": float(MAE)} | time_metrics
  return params, state, opt_state, metrics


def evaluate_epoch(loss_fn, params: hk.Params,
                   state: hk.State, ds: Iterable[LabelledGraph]) -> Metrics:
  """Evaluate for one epoch."""

  losses = []
  lengths = []
  MAEs = []
  for batch, length in ds:
    # State shouldn't change during evaluation
    loss, (MAE, state_out) = loss_fn(params, state, batch)
    losses.append(loss)
    MAEs.append(MAE)
    lengths.append(length)
  losses = jnp.asarray(losses)
  MAEs = jnp.asarray(MAEs)
  lengths = jnp.asarray(lengths)
  loss = jnp.sum(losses * lengths) / jnp.sum(lengths)
  MAE = jnp.sum(MAEs * lengths) / jnp.sum(lengths)

  metrics = {"loss": float(loss), "MAE": float(MAE)}
  return metrics
