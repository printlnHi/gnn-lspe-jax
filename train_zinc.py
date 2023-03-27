import functools
from typing import Any, Callable, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax

from type_aliases import (GraphClassifierFn, EvaluateFn, Metrics, TrainFn,
                          TrainResult, LabelledGraphs, LabelledGraph)


def compute_loss(net: hk.TransformedWithState, params: hk.Params, state: hk.State,
                 datapoint: LabelledGraph, rng: jax.random.KeyArray, is_training: bool) -> Tuple[jnp.ndarray, hk.State]:
  """Compute the loss for a given dataset."""
  graph, label = datapoint
  scores, state = net.apply(params, state, rng, graph, is_training=is_training)
  # L1 loss
  loss = jnp.mean(jnp.abs(scores - label))
  return loss, state


def train_epoch(net: hk.TransformedWithState, params: hk.Params, state: hk.State, rng: jax.random.KeyArray,
                opt_state: optax.OptState, opt_update: optax.TransformUpdateFn, ds: LabelledGraphs) -> TrainResult:
  """Train for one epoch."""
  compute_loss_fn = functools.partial(compute_loss, net, is_training=True)
  compute_loss_fn = jax.jit(jax.value_and_grad(compute_loss_fn, has_aux=True))

  losses = []
  for graph, label in ds:
    rng, subkey = jax.random.split(rng)
    (loss, state), grads = compute_loss_fn(
      params, state, (graph, label), subkey)
    updates, opt_state = opt_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    losses.append(loss)

  metrics = {"loss": float(jnp.mean(jnp.asarray(losses)))}
  return params, state, opt_state, metrics


def evaluate_epoch(net: hk.TransformedWithState, params: hk.Params,
                   state: hk.State, ds: LabelledGraphs) -> Metrics:
  """Evaluate for one epoch."""
  # Rng is only used for dropout, not needed for evaluation
  compute_loss_fn = functools.partial(
      compute_loss, net, rng=None, is_training=False)
  compute_loss_fn = jax.jit(compute_loss_fn)

  losses = []
  for graph, label in ds:
    # State shouldn't change during evaluation
    loss, _ = compute_loss_fn(params, state, (graph, label))
    losses.append(loss)

  metrics = {"loss": float(jnp.mean(jnp.asarray(losses)))}
  return metrics


def get_trainer_evaluator(
  net: hk.TransformedWithState) -> Tuple[TrainFn, EvaluateFn]:
  trainer = functools.partial(train_epoch, net)
  evaluator = functools.partial(evaluate_epoch, net)
  return trainer, evaluator
