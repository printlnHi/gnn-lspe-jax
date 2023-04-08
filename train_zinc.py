import functools
from typing import Any, Callable, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax

from type_aliases import (
  Metrics,
  TrainResult,
  TrainBatchResult,
    LabelledGraph)
from utils import DataLoader


def compute_loss(net: hk.TransformedWithState, params: hk.Params, state: hk.State,
                 batch: LabelledGraph, rng: jax.random.KeyArray, is_training: bool) -> Tuple[jnp.ndarray, hk.State]:
  """Compute the loss for a given dataset."""
  graph, label = batch
  scores, state = net.apply(params, state, rng, graph, is_training=is_training)
  mask = jraph.get_graph_padding_mask(graph)
  # L1 loss
  loss = jnp.sum(jnp.abs(scores - label) * mask) / jnp.sum(mask)
  """print("scores", scores.shape, scores)
  print("label", label.shape, label)
  print("mask", mask.shape, mask)
  print("scores-label", (scores - label).shape, scores - label)
  print(
      "abs(scores-label)",
      jnp.abs(
          scores -
          label).shape,
      jnp.abs(
          scores -
          label))
  print("abs(scores-label) * mask",
        (jnp.abs(scores - label) * mask).shape,
        jnp.abs(scores - label) * mask)
  print("loss", loss.shape, loss)"""

  return loss, state


def train_batch(loss_and_grad_fn, opt_update: optax.TransformUpdateFn, params: hk.Params, state: hk.State, rng: jax.random.KeyArray,
                opt_state: optax.OptState, batch: LabelledGraph) -> TrainBatchResult:
  (loss, state), grads = loss_and_grad_fn(
    params, state, batch, rng)
  updates, opt_state = opt_update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, state, opt_state, loss


def train_epoch_new(train_batch_fn, params: hk.Params, state: hk.State, rng: jax.random.KeyArray,
                    opt_state: optax.OptState, ds: DataLoader) -> TrainResult:
  """Train for one epoch."""

  losses = []
  lengths = []
  subkeys = jax.random.split(rng, len(ds))
  for (batch, length), subkey in zip(ds, subkeys):
    params, state, opt_state, loss = train_batch_fn(
        params, state, rng, opt_state, batch)
    losses.append(loss)
    lengths.append(length)
  losses = jnp.asarray(losses)
  lengths = jnp.asarray(lengths)
  loss = jnp.sum(losses * lengths) / jnp.sum(lengths)

  metrics = {"loss": float(loss)}
  return params, state, opt_state, metrics


def train_epoch(loss_and_grad_fn, opt_update: optax.TransformUpdateFn, params: hk.Params, state: hk.State, rng: jax.random.KeyArray,
                opt_state: optax.OptState, ds: DataLoader) -> TrainResult:
  """Train for one epoch."""

  losses = []
  lengths = []
  subkeys = jax.random.split(rng, len(ds))
  for (batch, length), subkey in zip(ds, subkeys):
    (loss, state), grads = loss_and_grad_fn(
      params, state, batch, subkey)
    updates, opt_state = opt_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    #print("mlp/~/linear_0: ", params['mlp/~/linear_0']['w'])
    losses.append(loss)
    lengths.append(length)
  losses = jnp.asarray(losses)
  lengths = jnp.asarray(lengths)
  loss = jnp.sum(losses * lengths) / jnp.sum(lengths)

  metrics = {"loss": float(loss)}
  return params, state, opt_state, metrics


def evaluate_epoch(loss_fn, params: hk.Params,
                   state: hk.State, ds: DataLoader) -> Metrics:
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
