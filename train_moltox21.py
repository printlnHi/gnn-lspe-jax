import functools
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple

import haiku as hk
import jax
import jax.lax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from sklearn.metrics import roc_auc_score

from types_and_aliases import LabelledGraph, LoadedData, Metrics, TrainResult


def compute_loss(net: hk.TransformedWithState, params: hk.Params, state: hk.State,
                 batch: LabelledGraph, rng: jax.random.KeyArray, is_training: bool) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, hk.State]]:
  """Compute the loss for a given dataset."""
  graph, label = batch
  mask = jraph.get_graph_padding_mask(graph)
  not_nan = ~jnp.isnan(label)
  combined_mask = mask[:, None] * not_nan
  zeroed_label = jnp.nan_to_num(label)

  (scores, _graph), state = net.apply(
    params, state, rng, graph, is_training=is_training)
  sigmoid_scores = jax.nn.sigmoid(scores)
  losses_old = - (zeroed_label * jnp.log(sigmoid_scores) +
                  (1 - zeroed_label) * jnp.log(1 - sigmoid_scores))
  M = -(jnp.clip(scores, a_max=0))
  losses = scores - scores * zeroed_label + M + \
      jnp.log(jnp.exp(-M) + jnp.exp(-scores - M))
  losses *= combined_mask
  loss = jnp.sum(losses) / jnp.sum(combined_mask)

  return loss, ((label, scores, combined_mask), state)


def roc_auc(label, scores, mask):
  rocauc_list = []
  for i in range(label.shape[1]):
    if np.sum(label[:, i] == 1) > 0 and np.sum(label[:, i] == 0) > 0:
      submask = mask[:, i]
      rocauc_list.append(roc_auc_score(
        label[submask, i], scores[submask, i]))
  return np.mean(rocauc_list)


def train_epoch(loss_and_grad_fn, opt_update: optax.TransformUpdateFn, opt_apply_updates, pe_init, params: hk.Params, state: hk.State, rng: jax.random.KeyArray,
                opt_state: optax.OptState, dataloader: Callable[[jax.random.KeyArray], LoadedData]) -> TrainResult:
  """Train for one epoch."""
  epoch_start_time = time.time()
  losses = []
  lengths = []
  ROC_AUCs = []
  loss_and_grad_times = []
  opt_update_times = []
  opt_apply_times = []

  rng, subkey = jax.random.split(rng)
  batches = list(dataloader(subkey))
  del subkey
  subkeys = jax.random.split(rng, 2 * len(batches)).reshape(-1, 2, 2)
  dataset_time = time.time() - epoch_start_time

  for (batch, length), subkey in zip(batches, subkeys):
    if pe_init == "lap_pe":
      flip = jax.random.bernoulli(
          subkey[0], shape=batch[0].nodes['pe'].shape[1]) * 2 - 1
      batch[0].nodes['pe'] = batch[0].nodes['pe'] * flip

    batch_start = time.time()
    (loss, ((label, scores, mask), state)), grads = loss_and_grad_fn(
      params, state, batch, subkey[1])
    ROC_AUC = roc_auc(label, scores, mask)
    loss_end = time.time()

    updates, opt_state = opt_update(grads, opt_state, params)
    opt_update_end = time.time()

    params = opt_apply_updates(params, updates)
    opt_end = time.time()

    losses.append(loss)
    ROC_AUCs.append(ROC_AUC)
    lengths.append(length)
    loss_and_grad_times.append(loss_end - batch_start)
    opt_update_times.append(opt_update_end - loss_end)
    opt_apply_times.append(opt_end - opt_update_end)

  losses = jnp.asarray(losses)
  ROC_AUCs = jnp.asarray(ROC_AUCs)
  lengths = jnp.asarray(lengths)
  loss = jnp.sum(losses * lengths) / jnp.sum(lengths)
  ROC_AUC = jnp.sum(ROC_AUCs * lengths) / jnp.sum(lengths)

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

  metrics = {"loss": float(loss), "ROC_AUC": float(ROC_AUC)} | time_metrics
  return params, state, opt_state, metrics


def evaluate_epoch(loss_fn, params: hk.Params, state: hk.State, ds: LoadedData) -> Metrics:
  """Evaluate for one epoch."""

  losses = []
  lengths = []
  ROC_AUCs = []
  for batch, length in ds:
    loss, ((label, scores, mask), state_out) = loss_fn(params, state, batch)
    ROC_AUC = roc_auc(label, scores, mask)
    losses.append(loss)
    ROC_AUCs.append(ROC_AUC)
    lengths.append(length)
  losses = jnp.asarray(losses)
  ROC_AUCs = jnp.asarray(ROC_AUCs)
  lengths = jnp.asarray(lengths)
  loss = jnp.sum(losses * lengths) / jnp.sum(lengths)
  ROC_AUC = jnp.sum(ROC_AUCs * lengths) / jnp.sum(lengths)

  metrics = {"loss": float(loss), "ROC_AUC": float(ROC_AUC)}
  return metrics
