import functools
from typing import Any, Callable, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax

from type_aliases import (Metrics, TrainResult, LabelledGraphs, LabelledGraph)


def compute_loss(net: hk.TransformedWithState, params: hk.Params, state: hk.State,
                 datapoint: LabelledGraph, rng: jax.random.KeyArray, is_training: bool) -> Tuple[jnp.ndarray, hk.State]:
  """Compute the loss for a given dataset."""
  graph, label = datapoint
  scores, state = net.apply(params, state, rng, graph, is_training=is_training)

  mask = jraph.get_graph_padding_mask(graph)
  # L1 loss
  loss = jnp.sum(jnp.abs(scores - label) * mask) / jnp.sum(mask)
  return loss, state,


def train_epoch(loss_and_grad_fn, params: hk.Params, state: hk.State, rng: jax.random.KeyArray,
                opt_state: optax.OptState, opt_update: optax.TransformUpdateFn, ds: LabelledGraphs) -> TrainResult:
  """Train for one epoch."""

  losses = []
  for graph, label in ds:
    rng, subkey = jax.random.split(rng)
    (loss, state), grads = loss_and_grad_fn(
      params, state, (graph, label), subkey)
    updates, opt_state = opt_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    losses.append(loss)

  metrics = {"loss": float(jnp.mean(jnp.asarray(losses)))}
  return params, state, opt_state, metrics


def evaluate_epoch(loss_fn, params: hk.Params,
                   state: hk.State, ds: LabelledGraphs) -> Metrics:
  """Evaluate for one epoch."""

  losses = []
  for graph, label in ds:
    # State shouldn't change during evaluation
    loss, _ = loss_fn(params, state, (graph, label))
    losses.append(loss)

  metrics = {"loss": float(jnp.mean(jnp.asarray(losses)))}
  return metrics
