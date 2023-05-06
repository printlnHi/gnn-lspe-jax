import argparse
from typing import Any, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

import datasets
import wandb
from lib.optimization import create_optimizer
from train_mutag import get_trainer_evaluator

# Adapted from https://github.com/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb
import jax
import jraph
import jax.numpy as jnp
import haiku as hk


@jraph.concatenated_args
def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Edge update function for graph net."""
  net = hk.Sequential(
      [hk.Linear(128), jax.nn.relu,
       hk.Linear(128)])
  return net(feats)


@jraph.concatenated_args
def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Node update function for graph net."""
  net = hk.Sequential(
      [hk.Linear(128), jax.nn.relu,
       hk.Linear(128)])
  return net(feats)


@jraph.concatenated_args
def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Global update function for graph net."""
  # MUTAG is a binary classification task, so output pos neg logits.
  net = hk.Sequential(
      [hk.Linear(128), jax.nn.relu,
       hk.Linear(2)])
  return net(feats)


def net_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
  # Add a global paramater for graph classification.
  graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))
  embedder = jraph.GraphMapFeatures(
      hk.Linear(128), hk.Linear(128), hk.Linear(128))
  net = jraph.GraphNetwork(
      update_node_fn=node_update_fn,
      update_edge_fn=edge_update_fn,
      update_global_fn=update_global_fn)
  return net(embedder(graph))


def train_val_pipeline(
  dataset, hyper_params: Dict[str, Any], net_params: Dict[str, Any], dirs, wandb_enabled=False):

  ds_train, ds_val, ds_test = dataset
  # Pre pad the graphs

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

  dataset = datasets.load('mutag')

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
