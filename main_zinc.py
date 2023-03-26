import argparse
import functools
import json
from typing import Any, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax

import datasets
import wandb
from nets.zinc import gnn_model
from utils import create_optimizer

if __name__ == "__main__":
  print("jax backend:", jax.lib.xla_bridge.get_backend().platform)
  print("jax devices:", jax.devices())

  parser = argparse.ArgumentParser()

  parser.add_argument("--wandb", action="store_true")
  parser.add_argument("--wandb_entity", type=str, default="marcushandley")
  parser.add_argument("--wandb_project", type=str, default="Part II")
  parser.add_argument("--wandb_run_name", type=str, default="test_main_mutag")

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

  # network parameters
  net_params = config["net_params"]

  dataset = datasets.zinc()
  train, val, test = dataset.train, dataset.val, dataset.test

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
    params, state = net.init(
        jax.random.PRNGKey(
            hyper_params["seed"]), train[0][0], is_training=True)

    # params = train_val_pipeline(dataset, hyper_params, {}, {}, wandb_enabled=args.wandb)
    # finish wandb normally
    wandb.finish()
  except Exception as e:
    # finish wandb noting error then reraise exception
    wandb.finish(exit_code=1)
    raise e
