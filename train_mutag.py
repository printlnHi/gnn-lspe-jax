import functools
from typing import Any, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax

from types_and_aliases import (MutagEvaluateFn, Metrics, MutagTrainFn,
                               MutagTrainResult, LabelledGraph)


def compute_loss(net: hk.Transformed, params: hk.Params, batch: LabelledGraph) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes loss and accuracy."""
  graph, label = batch
  pred_graph = net.apply(params, graph)
  preds = jax.nn.log_softmax(pred_graph.globals)
  targets = jax.nn.one_hot(label, 2)

  # Since we have an extra 'dummy' graph in our batch due to padding, we want
  # to mask out any loss associated with the dummy graph.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.
  mask = jraph.get_graph_padding_mask(pred_graph)

  # Cross entropy loss.
  loss = -jnp.mean(preds * targets * mask[:, None])

  # Accuracy taking into account the mask.
  accuracy = jnp.sum(
      (jnp.argmax(pred_graph.globals, axis=1) == label) * mask) / jnp.sum(mask)
  return loss, accuracy

  # Adapted from
  # https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py


def train_epoch(net: hk.Transformed, params: hk.Params, opt_state: optax.OptState, opt_update: optax.TransformUpdateFn,
                ds: List[LabelledGraph]) -> MutagTrainResult:

  compute_loss_fn = functools.partial(compute_loss, net)
  # We jit the computation of our loss, since this is the main computation.
  # Using jax.jit means that we will use a single accelerator. If you want
  # to use more than 1 accelerator, use jax.pmap. More information can be
  # found in the jax documentation.
  compute_loss_fn = jax.jit(jax.value_and_grad(
      compute_loss_fn, has_aux=True))

  losses = []
  accuracies = []
  print("type of ds: ", type(ds))
  print("type of ds[0]: ", type(ds[0]))
  print("type of ds[0][0]: ", type(ds[0][0]))
  print("type of ds[0][1]: ", type(ds[0][1]))
  for graph, label in ds:
    (loss, accuracy), grads = compute_loss_fn(params, graph, label)
    updates, opt_state = opt_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    losses.append(loss)
    accuracies.append(accuracy)

  metrics = {"loss": float(jnp.mean(jnp.asarray(losses))),
             "accuracy": float(jnp.mean(jnp.asarray(accuracies)))}
  return params, opt_state, metrics

# Assumes ds already padded


def evaluate_epoch(net: hk.Transformed,
                   params: hk.Params, ds: List[LabelledGraph]) -> Metrics:
  compute_loss_fn = functools.partial(compute_loss, net)
  compute_loss_fn = jax.jit(compute_loss_fn)

  losses = []
  accuracies = []
  for graph, label in ds:
    loss, accuracy = compute_loss_fn(params, graph, label)
    losses.append(loss)
    accuracies.append(accuracy)

  metrics = {"loss": float(jnp.mean(jnp.asarray(losses))),
             "accuracy": float(jnp.mean(jnp.asarray(accuracies)))}
  return metrics


def get_trainer_evaluator(
  net: hk.Transformed) -> Tuple[MutagTrainFn, MutagEvaluateFn]:
  trainer = functools.partial(train_epoch, net)
  evaluator = functools.partial(evaluate_epoch, net)
  return trainer, evaluator
