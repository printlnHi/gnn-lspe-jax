
from functools import partial
import time
from typing import (Any, Callable, Collection, Dict, Iterator, NamedTuple, NewType,
                    Optional, Tuple, Union)
import typing_extensions
import chex

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax


def create_optimizer_with_learning_rate_hyperparam(hyper_params: Dict[str, Any]) -> optax.GradientTransformation:
  init_lr = hyper_params["init_lr"]
  return optax.inject_hyperparams(optax.adamw)(learning_rate=init_lr, weight_decay=hyper_params["weight_decay"])


def create_lr_exponential_decay(hyper_params: Dict[str, Any]) -> optax.Schedule:
  steps_per_epoch = (
      10000 + hyper_params["batch_size"] - 1) // hyper_params["batch_size"]
  print("steps per epoch", steps_per_epoch)
  lr = optax.exponential_decay(
      init_value=hyper_params["init_lr"],
      transition_steps=hyper_params["transition_epochs"] * steps_per_epoch,
      decay_rate=hyper_params["lr_reduce_factor"],
      end_value=hyper_params["init_lr"] * 1e-2)
  return lr


'''MetaState = chex.ArrayTree


class MetaTransformInitFn(typing_extensions.Protocol):
  """A pure function that initializes the MetaSate."""

  def __call__(self, *args, **kwargs) -> MetaState:
    """Initializes the MetaState

    Args:
      args: Arguments to the initialization function.
      kwargs: Keyword arguments to the initialization function.
    Returns:
      The initialized MetaState
    """


class MetaTransformUpdateFn(typing_extensions.Protocol):
  """A pure function that transforms the OptState and updates the MetaState"""

  def __call__(self, metaState: MetaState, optaxState: optax.OptState, *args, **kwargs) -> Tuple[MetaState, optax.OptState]:
    """Updates the OptState

    Args:
      metaState: The MetaState
      optaxState: The optax.OptState
      args: Arguments to the update function.
      kwargs: Keyword arguments to the update function.

    Returns:
      The updated MetaState
    """


class MetaTransformation(NamedTuple):
  """A pair of pure functions implementing a transformation on the optimizer state."""
  init: MetaTransformInitFn
  update: MetaTransformUpdateFn'''


class ReduceLROnPlateau(object):
  def __init__(self, init_lr: float, reduce: float = 0.5, patience: int = 10, min_lr: float = 0):
    self.reduce = reduce
    self.patience = patience
    self.threshold = 0e-4
    self.lr = init_lr
    self.min_lr = min_lr
    self.best = float('inf')
    self.consecutive_bad_epoch = -1

  def step(self, score: float) -> float:
    if score < self.best * (1 - self.threshold):
      self.best = score
      self.consecutive_bad_epoch = 0
    else:
      self.consecutive_bad_epoch += 1

    if self.consecutive_bad_epoch > self.patience:
      if self.lr > self.min_lr:
        print("Reducing learning rate:", self.lr, end='')
        self.lr = max(self.min_lr, self.reduce * self.lr)
        print("->", self.lr)
      self.consecutive_bad_epoch = 0
    return self.lr

  __call__ = step


def create_reduce_lr_on_plateau(hyper_params: Dict[str, Any]) -> Callable[[float], float]:
  return ReduceLROnPlateau(
    hyper_params["init_lr"], hyper_params["lr_reduce_factor"], hyper_params["lr_schedule_patience"], hyper_params["min_lr"])


'''def create_optimizer_and_lr_tracker(hyper_params: Dict[str, Any]) -> Tuple[optax.GradientTransformation, Callable[[float, optax.OptState], optax.OptState]]:
  optimizer = optax.inject_hyperparams(optax.adamw)(
    learning_rate=hyper_params["init_lr"], weight_decay=hyper_params["weight_decay"])
  print(hyper_params['min_lr'])
  reduceOnPlateau = ReduceLROnPlateau(
    hyper_params["init_lr"], hyper_params["lr_reduce_factor"], hyper_params["lr_schedule_patience"], hyper_params["min_lr"])

  def tracker(score, opt_state):
    opt_state.hyperparams['learning_rate'] = reduceOnPlateau.step(score)
    return opt_state

  return optimizer, tracker


optax.GradientTransformation
optax.chain'''
