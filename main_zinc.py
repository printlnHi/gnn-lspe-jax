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
import numpy as np

import datasets
import wandb
from nets.zinc import gnn_model
from train_zinc import train_epoch, evaluate_epoch, compute_loss
from utils import create_optimizer, pad_all, DataLoader

if __name__ == "__main__":
  print("jax backend:", jax.lib.xla_bridge.get_backend().platform)
  print("jax devices:", jax.devices())
  print()

  parser = argparse.ArgumentParser()

  parser.add_argument("--wandb", action="store_true")
  parser.add_argument("--wandb_entity", type=str, default="marcushandley")
  parser.add_argument("--wandb_project", type=str, default="Part II")
  parser.add_argument("--wandb_run_name", type=str, default="proto_zinc")
  parser.add_argument("--print_every", type=int, default=100)

  parser.add_argument("--no_jit", action="store_true")
  parser.add_argument("--no_update_jit", action="store_true")
  parser.add_argument("--truncate_to", type=int, default=None)
  parser.add_argument("--no_pad", action="store_true")

  parser.add_argument(
      '--config',
      help="Please give a config.json file with training/model/data/param details")

  parser.add_argument("--epochs", type=int)
  parser.add_argument("--seed", type=int)

  args = parser.parse_args()

  with open(args.config) as f:
    config = json.load(f)

  # hyperparameters
  hyper_params = config["params"]
  if args.epochs:
    hyper_params["epochs"] = args.epochs
  if args.seed:
    hyper_params["seed"] = args.seed

  rng = jax.random.PRNGKey(hyper_params["seed"])

  # network parameters
  net_params = config["net_params"]

  dataset = datasets.zinc()
  train, val, test = dataset.train, dataset.val, dataset.test
  if not args.no_pad:
    train = pad_all(train)
    val = pad_all(val)
    test = pad_all(test)
  else:
    raise NotImplementedError(
      "Compute_loss currently required padded GraphsTuple ")

  if args.truncate_to:
    train = train[:args.truncate_to]
    val = val[:args.truncate_to]

  rng, subkey = jax.random.split(rng)
  trainloader = DataLoader(
      np.asarray(train, dtype=object),
      hyper_params["batch_size"],
      rng=subkey)
  valloader = DataLoader(
      np.asarray(
          val,
          dtype=object),
      hyper_params["batch_size"],
      rng=None)

  net_params["num_atom_type"] = dataset.num_atom_type
  net_params["num_bond_type"] = dataset.num_bond_type

  if args.wandb:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name)

  try:
    start_time = time.time()

    net_fn = gnn_model(net_params=net_params)
    net = hk.transform_with_state(net_fn)

    rng, subkey = jax.random.split(rng)
    params, state = net.init(subkey, train[0][0], is_training=True)
    del subkey
    opt_init, opt_update = create_optimizer(hyper_params)
    opt_state = opt_init(params)

    if not args.no_update_jit:
      opt_update = jax.jit(opt_update)

    train_loss_and_grad_fn = jax.value_and_grad(
      functools.partial(compute_loss, net, is_training=True), has_aux=True)
    # Rng only used for dropout, not needed for eval
    eval_loss_fn = functools.partial(
        compute_loss, net, rng=None, is_training=False)

    if not args.no_jit:
      train_loss_and_grad_fn = jax.jit(train_loss_and_grad_fn)
      eval_loss_fn = jax.jit(eval_loss_fn)

    print("Training...")
    for epoch in range(hyper_params["epochs"]):
      print_epoch_metrics = epoch % args.print_every == 0 or epoch == hyper_params[
        "epochs"] - 1
      # Train for one epoch.
      rng, subkey = jax.random.split(rng)
      params, state, opt_state, train_metrics = train_epoch(
          train_loss_and_grad_fn, params, state, subkey, opt_state, opt_update, train)
      if print_epoch_metrics:
        print(
            f'Epoch {epoch} - train loss: {train_metrics["loss"]}')

      # Evaluate on the validation set.
      val_metrics = evaluate_epoch(eval_loss_fn, params, state, val)
      if print_epoch_metrics:
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
