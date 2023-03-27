import argparse
import functools
import json
import time
from typing import Any, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax

import datasets
import wandb
from nets.zinc import gnn_model
from train_zinc import get_trainer_evaluator
from utils import create_optimizer

if __name__ == "__main__":
  print("jax backend:", jax.lib.xla_bridge.get_backend().platform)
  print("jax devices:", jax.devices())
  print()

  parser = argparse.ArgumentParser()

  parser.add_argument("--wandb", action="store_true")
  parser.add_argument("--wandb_entity", type=str, default="marcushandley")
  parser.add_argument("--wandb_project", type=str, default="Part II")
  parser.add_argument("--wandb_run_name", type=str, default="proto_zinc")

  parser.add_argument(
      '--config',
      help="Please give a config.json file with training/model/data/param details")

  parser.add_argument("--epochs", type=int)

  args = parser.parse_args()

  with open(args.config) as f:
    config = json.load(f)

  # hyperparameters
  hyper_params = config["params"]
  if args.epochs:
    hyper_params["epochs"] = args.epochs

  rng = jax.random.PRNGKey(hyper_params["seed"])

  # network parameters
  net_params = config["net_params"]

  dataset = datasets.zinc()
  train, val, test = dataset.train, dataset.val, dataset.test
  # TODO: pad dataset
  train_trunc = train[:1]

  net_params["num_atom_type"] = dataset.num_atom_type
  net_params["num_bond_type"] = dataset.num_bond_type

  if args.wandb:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name)

  try:
    net_fn = gnn_model(net_params=net_params)
    net = hk.transform_with_state(net_fn)

    rng, subkey = jax.random.split(rng)
    params, state = net.init(subkey, train[0][0], is_training=True)
    del subkey
    opt_init, opt_update = create_optimizer(hyper_params)
    opt_state = opt_init(params)
    train_epoch, evaluate_epoch = get_trainer_evaluator(net)

    start_time = time.time()

    for epoch in range(hyper_params["epochs"]):
      # Train for one epoch.
      rng, subkey = jax.random.split(rng)
      params, state, opt_state, train_metrics = train_epoch(
          params, state, subkey, opt_state, opt_update, train_trunc)
      print(
          f'Epoch {epoch} - train loss: {train_metrics["loss"]}')

      # Evaluate on the validation set.
      val_metrics = evaluate_epoch(params, state, train_trunc)
      print(
          f'Epoch {epoch} - val loss: {val_metrics["loss"]}')

      if args.wandb:
        time_elapsed = time.time() - start_time
        wandb.log({'epoch': epoch,
                   'time': time_elapsed} | {'train ' + k: v for k,
                                            v in train_metrics.items()} | {'val ' + k: v for k,
                                                                           v in val_metrics.items()})

    # finish wandb normally
    wandb.finish()
  except Exception as e:
    # finish wandb noting error then reraise exception
    wandb.finish(exit_code=1)
    raise e
