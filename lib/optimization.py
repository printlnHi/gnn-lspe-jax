
from typing import Any, Callable, Dict

import optax


def create_optimizer(hyper_params: Dict[str, Any]) -> optax.GradientTransformation:
  return optax.adamw(learning_rate=hyper_params["init_lr"], weight_decay=hyper_params["weight_decay"])


def create_optimizer_with_learning_rate_hyperparam(hyper_params: Dict[str, Any]) -> optax.GradientTransformation:
  return optax.inject_hyperparams(optax.adamw)(learning_rate=hyper_params["init_lr"],
                                               weight_decay=hyper_params["weight_decay"])


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
  # We use min_lr as stopping point rather than a simple floor
  return ReduceLROnPlateau(
    hyper_params["init_lr"], hyper_params["lr_reduce_factor"], hyper_params["lr_schedule_patience"], min_lr=0)
