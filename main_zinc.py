import argparse
import functools
import json
import time

import haiku as hk
import jax
import numpy as np

import datasets
import wandb
from nets.zinc import gnn_model
from train_zinc import train_epoch, evaluate_epoch, compute_loss
from utils import create_optimizer, DataLoader

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
  parser.add_argument("--batch_size", type=int)

  args = parser.parse_args()

  with open(args.config) as f:
    config = json.load(f)

  # hyperparameters
  hyper_params = config["params"]
  if args.epochs:
    hyper_params["epochs"] = args.epochs
  if args.seed:
    hyper_params["seed"] = args.seed
  if args.batch_size:
    hyper_params["batch_size"] = args.batch_size
  if args.truncate_to:
    hyper_params["truncate_to"] = args.truncate_to

  rng = jax.random.PRNGKey(hyper_params["seed"])

  # network parameters
  net_params = config["net_params"]

  dataset = datasets.zinc()
  train, val, test = dataset.train, dataset.val, dataset.test
  if args.no_pad:
    raise NotImplementedError(
      "Compute_loss currently required padded GraphsTuple ")

  if "truncate_to" in hyper_params and hyper_params["truncate_to"] is not None:
    train = train[:hyper_params["truncate_to"]]
    val = val[:hyper_params["truncate_to"]]

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
  testloader = DataLoader(
    np.asarray(test, dtype=object), hyper_params["batch_size"]
  )

  net_params["num_atom_type"] = dataset.num_atom_type
  net_params["num_bond_type"] = dataset.num_bond_type

  if args.wandb:
    tags = ["zinc"]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name, config=hyper_params, tags=tags)

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
          train_loss_and_grad_fn, params, state, subkey, opt_state, opt_update, trainloader)
      if print_epoch_metrics:
        print(
            f'Epoch {epoch} - train loss: {train_metrics["loss"]}')

      # Evaluate on the validation set.
      val_metrics = evaluate_epoch(eval_loss_fn, params, state, valloader)
      if print_epoch_metrics:
        print(
            f'Epoch {epoch} - val loss: {val_metrics["loss"]}')

      if args.wandb:
        time_elapsed = time.time() - start_time
        wandb.log({'epoch': epoch,
                   'time': time_elapsed} | {'train ' + k: v for k, v in train_metrics.items()} | {'val ' + k: v for k, v in val_metrics.items()})

    if args.truncate_to:
      valloader = DataLoader(
          np.asarray(
              dataset.val,
              dtype=object),
          1,
          rng=None)

      final_val_metrics = evaluate_epoch(
          eval_loss_fn, params, state, valloader)
    else:
      final_val_metrics = val_metrics

    final_test_metrics = evaluate_epoch(
        eval_loss_fn, params, state, testloader)

    final_metrics = {
        'final val ' + k: v for k,
        v in final_val_metrics.items()} | {
        'final test ' + k: v for k,
        v in final_test_metrics.items()}
    print("===Final metrics (untruncated)===")
    print(final_metrics)
    if args.wandb:
      wandb.log(final_metrics)

    if args.wandb:
      # finish wandb normally
      wandb.finish()
  except Exception as e:
    # finish wandb noting error then reraise exception
    wandb.finish(exit_code=1)
    raise e
