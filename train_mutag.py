import jax
import jax.numpy as jnp
import jraph
import haiku as hk
import optax
import functools

from typing import Tuple, List, Dict, Any, Callable


def compute_loss(params: hk.Params, graph: jraph.GraphsTuple, label: jnp.ndarray,
                 net: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes loss and accuracy."""
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

  # Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py

# Assume ds already padded
def train_epoch(params: hk.Params, opt_state: optax.OptState, opt_update: optax.TransformUpdateFn, ds: List[Dict[str, Any]], net: jraph.GraphsTuple) -> Tuple[hk.Params, optax.OptState, Dict[str, Any]]:
    
  compute_loss_fn = functools.partial(compute_loss, net=net)
  # We jit the computation of our loss, since this is the main computation.
  # Using jax.jit means that we will use a single accelerator. If you want
  # to use more than 1 accelerator, use jax.pmap. More information can be
  # found in the jax documentation.
  compute_loss_fn = jax.jit(jax.value_and_grad(
      compute_loss_fn, has_aux=True))    

  losses = []
  accuracies = []
  for graph, label in ds:
    (loss, accuracy), grads = compute_loss_fn(params, graph, label)
    updates, opt_state = opt_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    losses.append(loss)
    accuracies.append(accuracy)
  
  metrics = {"loss": jnp.mean(jnp.asarray(losses)), "accuracy":jnp.mean(jnp.asarray(accuracies))}
  return params, opt_state, metrics

# Assumes ds already padded
def evaluate_epoch(params: hk.Params, ds: List[Dict[str, Any]], net: jraph.GraphsTuple) -> Dict[str, Any]:
  compute_loss_fn = functools.partial(compute_loss, net=net)
  compute_loss_fn = jax.jit(compute_loss_fn)

  losses = []
  accuracies = []
  for graph, label in ds:
    loss, accuracy = compute_loss_fn(params, graph, label)
    losses.append(loss)
    accuracies.append(accuracy)
  
  metrics = {"loss": jnp.mean(jnp.asarray(losses)), "accuracy":jnp.mean(jnp.asarray(accuracies))}
  return metrics


def get_trainer_evaluator(net: jraph.GraphsTuple) -> Tuple[Callable[[hk.Params, optax.OptState, optax.TransformUpdateFn, List[Dict[str, Any]]],Tuple[hk.Params, optax.OptState, Dict[str, Any]]], Callable[[hk.Params, List[Dict[str, Any]]], Dict[str, Any]]]:
  trainer = functools.partial(train_epoch, net=net)
  evaluator = functools.partial(evaluate_epoch, net=net)
  return trainer, evaluator

