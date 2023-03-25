import functools
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
import haiku as hk
import optax
import numpy as onp
from typing import Any, Callable, Dict, List, Optional, Tuple

import wandb

from utils import pad_graph_to_nearest_power_of_two, pad_all
import nets
import datasets

train_zinc_ds, val_zinc_ds, test_zinc_ds, num_atom_types, num_bond_types = datasets.zinc()

train_mutag_ds, test_mutag_ds = datasets.mutag()

train_ds = train_mutag_ds
val_ds = test_mutag_ds
test_ds = test_mutag_ds

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

def train(train_ds: List[Dict[str, Any]], val_ds: Optional[List[Dict[str, Any]]], num_train_steps: int, net_fn: Callable[[jraph.GraphsTuple], jraph.GraphsTuple]) -> hk.Params:
  """Training loop."""

  # Transform impure `net_fn` to pure functions with hk.transform.
  net = hk.without_apply_rng(hk.transform(net_fn))
  # Get a candidate graph and label to initialize the network.
  graph = train_ds[0]['input_graph']

  # Initialize the network.
  params = net.init(jax.random.PRNGKey(42), graph)
  # Initialize the optimizer.
  opt_init, opt_update = optax.adam(1e-4)
  opt_state = opt_init(params)

  compute_loss_fn = functools.partial(compute_loss, net=net)
  # We jit the computation of our loss, since this is the main computation.
  # Using jax.jit means that we will use a single accelerator. If you want
  # to use more than 1 accelerator, use jax.pmap. More information can be
  # found in the jax documentation.
  compute_loss_fn = jax.jit(jax.value_and_grad(
      compute_loss_fn, has_aux=True))


  padded_train_ds = pad_all(train_ds)
  print(f'Training dataset size: {len(padded_train_ds)}')

  # If not None we will evaluate the model on the validation set.
  if val_ds is not None:
    padded_val_ds = pad_all(val_ds)
    print(f'Validation dataset size: {len(padded_val_ds)}')
  else:
    padded_val_ds = None
    print('No validation dataset provided')
  
  for idx in range(num_train_steps):
    graph = padded_train_ds[idx % len(train_ds)]['input_graph']
    label = padded_train_ds[idx % len(train_ds)]['target']

    (loss, acc), grad = compute_loss_fn(params, graph, label)
    updates, opt_state = opt_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    wandb.log({'loss': loss, 'acc': acc, 'idx': idx})
    if idx % 50 == 0:
      print(f'idx: {idx}, loss: {loss}, acc: {acc}')
  print('Training finished')
  return params
  
def train_batched(train_ds: List[Dict[str, Any]], val_ds: Optional[List[Dict[str, Any]]], num_epochs: int, net_fn: Callable[[jraph.GraphsTuple], jraph.GraphsTuple], batch_size: int = 32) -> hk.Params:
  """Training loop."""

  # Transform impure `net_fn` to pure functions with hk.transform.
  net = hk.without_apply_rng(hk.transform(net_fn))
  # Get a candidate graph and label to initialize the network.
  graph = train_ds[0]['input_graph']

  # Initialize the network.
  params = net.init(jax.random.PRNGKey(42), graph)
  # Initialize the optimizer.
  opt_init, opt_update = optax.adam(1e-4)
  opt_state = opt_init(params)

  compute_loss_fn = functools.partial(compute_loss, net=net)
  # We jit the computation of our loss, since this is the main computation.
  # Using jax.jit means that we will use a single accelerator. If you want
  # to use more than 1 accelerator, use jax.pmap. More information can be
  # found in the jax documentation.
  compute_loss_fn_jit = jax.jit(jax.value_and_grad(
      compute_loss_fn, has_aux=True))
  #pmap compute_loss_fn instead across graph and label
  compute_loss_fn = jax.pmap(compute_loss_fn_jit, in_axes=(None, 0, 0))


  print('Unpadded training dataset size: ', len(train_ds))
  padded_train_ds = pad_all(train_ds)
  print(f'Training dataset size: {len(padded_train_ds)}')

  # If not None we will evaluate the model on the validation set.
  if val_ds is not None:
    padded_val_ds = pad_all(val_ds)
    print(f'Validation dataset size: {len(padded_val_ds)}')
  else:
    padded_val_ds = None
    print('No validation dataset provided')
  
  print_every = (num_epochs+20-1)//20
  for epoch in range(num_epochs):
    for idx in range(0, len(padded_train_ds), batch_size):
      graphs = [padded_train_ds[i % len(train_ds)]['input_graph'] for i in range(idx, idx + batch_size)]
      labels = [padded_train_ds[i % len(train_ds)]['target'] for i in range(idx, idx + batch_size)]

      (loss, acc), grad = compute_loss_fn(params, graphs, labels)
      updates, opt_state = opt_update(grad, opt_state, params)
      params = optax.apply_updates(params, updates)
      wandb.log({'loss': loss, 'acc': acc, 'idx': idx})
      if epoch % print_every == 0 or epoch == num_epochs-1:
        print(f'epoch: {epoch}, idx: {idx}, loss: {loss}, acc: {acc}')





def evaluate(dataset: List[Dict[str, Any]],
             params: hk.Params,
             net_fn: Callable[[jraph.GraphsTuple], jraph.GraphsTuple]) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Evaluation Script."""
  # Transform impure `net_fn` to pure functions with hk.transform.
  net = hk.without_apply_rng(hk.transform(net_fn))
  # Get a candidate graph and label to initialize the network.
  graph = dataset[0]['input_graph']
  accumulated_loss = 0
  accumulated_accuracy = 0
  compute_loss_fn = jax.jit(functools.partial(compute_loss, net=net))
  for idx in range(len(dataset)):
    graph = dataset[idx]['input_graph']
    label = dataset[idx]['target']
    graph = pad_graph_to_nearest_power_of_two(graph)
    label = jnp.concatenate([label, jnp.array([0])])
    loss, acc = compute_loss_fn(params, graph, label)
    accumulated_accuracy += acc
    accumulated_loss += loss
    if idx % 100 == 0:
      print(f'Evaluated {idx + 1} graphs')
  print('Completed evaluation.')
  loss = accumulated_loss / idx
  accuracy = accumulated_accuracy / idx
  print(f'Eval loss: {loss}, accuracy {accuracy}')
  return loss, accuracy

wandb.init(project="Part II", entity="marcushandley", name="test")

try:
  params = train_batched(train_mutag_ds, test_mutag_ds, num_epochs=5, net_fn=networks.net_fn)
  evaluate(test_mutag_ds, params, net_fn=networks.net_fn)
  #finish wandb normally
  wandb.finish()
except Exception as e:
  #finish wandb noting error then reraise exception
  wandb.finish(exit_code=1)
  raise e
  
