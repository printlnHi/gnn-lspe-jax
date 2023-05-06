import argparse
from typing import Any, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

import datasets
import wandb
from lib.optimization import create_optimizer
from nets.mutag import net_fn
from train_mutag import get_trainer_evaluator


def train_val_pipeline(
  dataset, hyper_params: Dict[str, Any], net_params: Dict[str, Any], dirs, wandb_enabled=False):

  ds_train, ds_val, ds_test = dataset
  # Assumed datasets already padded

  # TODO: Choose from multiple different nets using "model name" or enum or similar as parameter
  # TODO: Use net_params to initialize net
  net = hk.without_apply_rng(hk.transform(net_fn))

  params = net.init(jax.random.PRNGKey(hyper_params["seed"]), ds_train[0][0])
  opt_init, opt_update = create_optimizer(hyper_params)
  opt_state = opt_init(params)

  train_epoch, evaluate_epoch = get_trainer_evaluator(net)

  for epoch in range(hyper_params["epochs"]):
    # Train for one epoch.
    params, opt_state, train_metrics = train_epoch(
      params, opt_state, opt_update, ds_train)
    print(
      f'Epoch {epoch} - train loss: {train_metrics["loss"]}, train accuracy: {train_metrics["accuracy"]}')

    # Evaluate on the validation set.
    val_metrics = evaluate_epoch(params, ds_val)
    print(
      f'Epoch {epoch} - val loss: {val_metrics["loss"]}, val accuracy: {val_metrics["accuracy"]}')

    if wandb_enabled:
      wandb.log({"epoch": epoch} | train_metrics | val_metrics)

  # Evaluate on the test set.
  metrics = evaluate_epoch(params, ds_test)
  return (params, metrics)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--wandb", action="store_true")
  parser.add_argument("--wandb_entity", type=str, default="marcushandley")
  parser.add_argument("--wandb_project", type=str, default="Part II")
  parser.add_argument("--wandb_run_name", type=str, default="test_main_mutag")

  parser.add_argument("--epochs", type=int)

  args = parser.parse_args()

  hyper_params = {
      "seed": 42,
      "epochs": 1000,
      "batch_size": 256,
      "init_lr": 1e-4,  # 0.001 for others
      "lr_reduce_factor": 0.5,
      "lr_schedule_patience": 25,
      "min_lr": 1e-5,
      "weight_decay": 0.0,
      "print_epoch_interval": 1,
      "max_time": 48
    }

  print("jax backend:", jax.lib.xla_bridge.get_backend().platform)
  print("jax devices:", jax.devices())

  if args.epochs:
    hyper_params["epochs"] = args.epochs

  dataset = datasets.mutag()

  if args.wandb:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name)
  try:
    params = train_val_pipeline(
        dataset,
        hyper_params,
        {},
        {},
        wandb_enabled=args.wandb)
    # finish wandb normally
    wandb.finish()
  except Exception as e:
    # finish wandb noting error then reraise exception
    wandb.finish(exit_code=1)
    raise e
