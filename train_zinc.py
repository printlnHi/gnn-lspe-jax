import functools
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from type_aliases import LabelledGraph, Metrics, TrainResult
from utils import graphLaplacian


def compute_loss(net: hk.TransformedWithState, params: hk.Params, state: hk.State,
                 batch: LabelledGraph, rng: jax.random.KeyArray, is_training: bool) -> Tuple[jnp.ndarray, hk.State]:
  """Compute the loss for a given dataset."""
  graph, label = batch
  scores, state = net.apply(params, state, rng, graph, is_training=is_training)
  mask = jraph.get_graph_padding_mask(graph)
  # L1 loss
  loss = jnp.sum(jnp.abs(scores - label) * mask) / jnp.sum(mask)
  return loss, state


def compute_lapeig_inclusive_loss(net: hk.TransformedWithState, net_params, params: hk.Params, state: hk.State,
                                  batch: LabelledGraph, rng: jax.random.KeyArray, is_training: bool) -> Tuple[jnp.ndarray, hk.State]:
  graph, label = batch
  (scores, p), state = net.apply(
    params, state, rng, graph, is_training=is_training)
  graph_mask = jraph.get_graph_padding_mask(graph)
  node_mask = jraph.get_node_padding_mask(graph)
  jraph.get_number_of_padding_with_graphs_edges
  num_graphs = jnp.sum(graph_mask)
  num_nodes = jnp.sum(node_mask)

  task_loss = jnp.sum(jnp.abs(scores - label) * graph_mask) / num_graphs

  L = graphLaplacian(graph, np_=jnp)
  pT = jnp.transpose(p)
  trace = jnp.trace(pT @ L @ p)
  print("L", L)
  print("p", p)
  print("trace", trace)

  positional_loss = trace / \
      (net_params["pos_enc_dim"] * num_graphs * num_nodes)

  loss = task_loss + net_params['alpha_loss'] * positional_loss
  return loss, state


def train_epoch(loss_and_grad_fn, opt_update: optax.TransformUpdateFn, opt_apply_updates, pe_init, params: hk.Params, state: hk.State, rng: jax.random.KeyArray,
                opt_state: optax.OptState, dataloader) -> TrainResult:
  """Train for one epoch."""
  epoch_start_time = time.time()
  losses = []
  lengths = []
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
    (loss, state), grads = loss_and_grad_fn(params, state, batch, subkey[1])
    loss_end = time.time()

    updates, opt_state = opt_update(grads, opt_state, params)
    opt_update_end = time.time()

    params = opt_apply_updates(params, updates)
    opt_end = time.time()

    losses.append(loss)
    lengths.append(length)
    loss_and_grad_times.append(loss_end - batch_start)
    opt_update_times.append(opt_update_end - loss_end)
    opt_apply_times.append(opt_end - opt_update_end)

  losses = jnp.asarray(losses)
  lengths = jnp.asarray(lengths)
  loss = jnp.sum(losses * lengths) / jnp.sum(lengths)

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

  metrics = {"loss": float(loss)} | time_metrics
  return params, state, opt_state, metrics


def evaluate_epoch(loss_fn, params: hk.Params,
                   state: hk.State, ds: Iterable[LabelledGraph]) -> Metrics:
  """Evaluate for one epoch."""

  losses = []
  lengths = []
  for batch, length in ds:
    # State shouldn't change during evaluation
    #print("=" * 80)
    loss, _ = loss_fn(params, state, batch)
    #print(batch[0].nodes['feat'].shape, loss)
    #print("=" * 80)
    loss, state_out = loss_fn(params, state, batch)
    losses.append(loss)
    lengths.append(length)
  losses = jnp.asarray(losses)
  lengths = jnp.asarray(lengths)
  loss = jnp.sum(losses * lengths) / jnp.sum(lengths)

  metrics = {"loss": float(loss)}
  return metrics
