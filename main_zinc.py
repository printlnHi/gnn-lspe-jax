import argparse
from collections import Counter
import functools
import json
import os
import time

import haiku as hk
import jax
from jax.config import config
import numpy as np
import optax

import datasets
import wandb
from nets.zinc import gnn_model
from train_zinc import train_epoch, evaluate_epoch, compute_loss
from utils import create_optimizer, power_of_two_padding, GraphsSize, PaddingScheme, flat_data_loader, lapPE, RWPE

if __name__ == "__main__":
  # config.update("jax_log_compiles", True)
  '''
  import logging
  logging.getLogger("jax").setLevel(logging.DEBUG)
  '''

  print("jax backend:", jax.lib.xla_bridge.get_backend().platform)
  print("jax devices:", jax.devices())
  print()

  # ==================== Load parameters ====================
  # Parameters are loaded from config and can be overwritten from command line

  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--config',
      help="Please give a config.json file with training/model/data/param details")
  parser.add_argument("--out_dir", type=str, default=os.getcwd() + "/out")

  # Run parameters
  parser.add_argument("--wandb", action="store_true")
  parser.add_argument("--wandb_entity", type=str, default="marcushandley")
  parser.add_argument("--wandb_project", type=str, default="Part II")
  parser.add_argument("--wandb_run_name", type=str, default="proto_zinc")
  parser.add_argument("--print_every", type=int, default=100)
  # Hyperparameters
  parser.add_argument("--seed", type=int)
  parser.add_argument("--batch_size", type=int)
  parser.add_argument("--epochs", type=int)
  parser.add_argument("--transition_epochs", type=int, default=150)
  # Network parameters
  parser.add_argument("--pe_init", type=str)
  parser.add_argument("--no_mask_batch_norm", action="store_true")
  # Development parameters
  parser.add_argument("--truncate_to", type=int, default=None)
  parser.add_argument("--profile", action="store_true")

  args = parser.parse_args()

  with open(args.config) as f:
    config = json.load(f)

  # hyperparameters
  hyper_params = config["params"]
  if args.epochs is not None:
    hyper_params["epochs"] = args.epochs
  if args.seed is not None:
    hyper_params["seed"] = args.seed
  if args.batch_size:
    hyper_params["batch_size"] = args.batch_size
  hyper_params["transition_epochs"] = args.transition_epochs

  # development parameters
  hyper_params["truncate_to"] = args.truncate_to

  # network parameters
  net_params = config["net_params"]
  if args.pe_init:
    net_params["pe_init"] = args.pe_init
  net_params["mask_batch_norm"] = not args.no_mask_batch_norm

  # ==================== Data ====================
  dataset = datasets.zinc()
  net_params["num_atom_type"] = dataset.num_atom_type
  net_params["num_bond_type"] = dataset.num_bond_type

  if net_params["pe_init"] == "lap_pe":
    print("adding lap PE ...", end=" ", flush=True)
    pe_func = functools.partial(lapPE, pos_enc_dim=net_params["pos_enc_dim"])
    dataset.add_PE(pe_func, ["pe", "eigvec"])
    print("done")
  elif net_params["pe_init"] == "rand_walk":
    print("adding RW PE ...", end=" ", flush=True)
    pe_func = functools.partial(RWPE, pos_enc_dim=net_params["pos_enc_dim"])
    dataset.add_PE(pe_func, ["pe"])
    print("done")

  train, val, test = dataset.train, dataset.val, dataset.test
  if hyper_params["truncate_to"] is not None:
    train = train[:hyper_params["truncate_to"]]
    val = val[:hyper_params["truncate_to"]]

  # We do power of two padding and count graph sizes along the way
  padded_graph_sizes = Counter()

  def padding_strategy(size: GraphsSize) -> GraphsSize:
    padded_size = power_of_two_padding(
        size, batch_size=hyper_params["batch_size"])
    # TODO: Delete print statement
    if padded_size not in padded_graph_sizes:
      print(padded_size)
    padded_graph_sizes[padded_size] += 1
    return padded_size

  rng = jax.random.PRNGKey(hyper_params["seed"])
  trainloader = functools.partial(flat_data_loader,
                                  train, hyper_params["batch_size"], padding_strategy)
  valloader = flat_data_loader(
      val,
      hyper_params["batch_size"],
      padding_strategy,
      None)

  testloader = flat_data_loader(
      test,
      hyper_params["batch_size"],
      padding_strategy,
      None)
  # ============= Model, Train and Eval functions =============

  net_fn = gnn_model(net_params=net_params)
  net = hk.transform_with_state(net_fn)

  rng, subkey = jax.random.split(rng)
  params, state = net.init(subkey, train[0][0], is_training=True)
  del subkey

  opt_init, opt_update = create_optimizer(hyper_params)
  opt_state = opt_init(params)

    compute_loss_fn = functools.partial(compute_lapeig_inclusive_loss, net)
  train_loss_and_grad_fn = jax.value_and_grad(
    functools.partial(compute_loss, net, is_training=True), has_aux=True)

  train_epoch_fn = functools.partial(
      train_epoch, jax.jit(train_loss_and_grad_fn), jax.jit(opt_update), jax.jit(optax.apply_updates), net_params["pe_init"])

  # Rng only used for dropout, not needed for eval
  eval_loss_fn = jax.jit(functools.partial(
      compute_loss, net, rng=None, is_training=False))

  # ==================== Training ====================
  if args.wandb:
    tags = ["zinc"]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name, config=hyper_params | net_params, tags=tags)

  try:

    # ==================== Training loop ====================

    print("Training...")
    val_metrics = None
    total_train_time = 0
    total_val_time = 0

    if args.profile:
      jax.profiler.start_trace("out", False, True)

    for epoch in range(hyper_params["epochs"]):
      print_epoch_metrics = epoch % args.print_every == 0 or epoch == hyper_params[
        "epochs"] - 1
      epoch_started = time.time()
      seen_sizes = len(padded_graph_sizes)

      # Train for one epoch.
      rng, subkey = jax.random.split(rng)
      params, state, opt_state, train_metrics = train_epoch_fn(
        params, state, subkey, opt_state, trainloader)
      del subkey  # TODO: Do I want to keep deleting subkey? Looks ugly but prevents issues
      if print_epoch_metrics:
        print(
            f'Epoch {epoch} - train loss: {train_metrics["loss"]}')
        print(train_metrics)
      train_finished = time.time()

      # Evaluate on the validation set.
      val_metrics = evaluate_epoch(eval_loss_fn, params, state, valloader)
      if print_epoch_metrics:
        print(
            f'Epoch {epoch} - val loss: {val_metrics["loss"]}')
      val_finished = time.time()

      new_sizes = len(padded_graph_sizes) - seen_sizes

      train_time = train_finished - epoch_started
      val_time = val_finished - train_finished
      epoch_time = train_time + val_time
      total_train_time += train_time
      total_val_time += val_time

      timing_metrics = {
          'new_sizes': new_sizes,
          'epoch_time': epoch_time,
          'train_time': train_time,
          'val_time': val_time}
      if print_epoch_metrics:
        print(
          f'Epoch {epoch} - time: {epoch_time} (train: {train_time}, val: {val_time})')
        if new_sizes > 0:
          print(" " * 10 + f'>{new_sizes} new sizes!')

      if args.wandb:
        train_metrics = {'train ' + k: v for k, v in train_metrics.items()}
        wandb.log({"epoch": epoch} | timing_metrics |
                  train_metrics | {'val ' + k: v for k, v in val_metrics.items()})

    if args.profile:
      jax.profiler.stop_trace()

    # ==================== Final evaluation ====================
    # ~We want padded graph sizes from the training loop only~
    padded_graph_sizes_stringified = {
        str(size): count for size,
        count in padded_graph_sizes.items()}
    graph_sizes_seen = len(padded_graph_sizes)

    if args.truncate_to:
      valloader = flat_data_loader(
          dataset.val,
          hyper_params["batch_size"],
          padding_strategy,
          None)
      final_val_metrics = evaluate_epoch(
          eval_loss_fn, params, state, valloader)
    elif val_metrics is None:
      final_val_metrics = evaluate_epoch(
          eval_loss_fn, params, state, valloader)
    else:
      final_val_metrics = val_metrics

    print("ABOUT TO EVALUATE TEST")
    final_test_metrics = evaluate_epoch(
        eval_loss_fn, params, state, testloader)

    final_timing_metrics = {
        'graph_sizes_seen': graph_sizes_seen,
        'total_train_time': total_train_time,
        'total_val_time': total_val_time,
        'total_epoch_time': total_train_time + total_val_time}

    final_val_metrics = {
        'final val ' + k: v for k,
        v in final_val_metrics.items()}
    final_test_metrics = {
        'final test ' + k: v for k,
        v in final_test_metrics.items()}

    final_metrics = final_timing_metrics | final_val_metrics | final_test_metrics

    print("===Final metrics ===")
    print(final_metrics)

    if args.wandb:
      wandb.log(final_metrics)

      # Only save the model or graph sizes if we are using wandb
      out_dir = os.path.join(args.out_dir, f"{run.name}-{run.id}")
      os.makedirs(out_dir, exist_ok=False)
      # TODO: Save the model

      # json encode the graph sizes
      sizes_path = os.path.join(out_dir, "padded_graph_sizes.json")
      sizes_s = json.dumps(padded_graph_sizes_stringified, sort_keys=True)

      with open(sizes_path, "w") as f:
        f.write(sizes_s)

      wandb.save(sizes_path)

      # finish wandb normally
      wandb.finish()

  except Exception as e:
    # finish wandb then reraise exception
    wandb.finish(exit_code=1)
    raise e
