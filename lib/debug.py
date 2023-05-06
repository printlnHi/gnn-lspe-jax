
import haiku as hk
import jax
import jax.numpy as jnp


class HaikuDebug(hk.Module):
  def __init__(self, name=None, label=None, enable: bool = True):
    self.name = name
    self.label = label
    if self.name is None and self.label is None:
      self.name = self.label = "HaikuDebug"
    elif self.name is None:
      self.name = self.label
    elif self.label is None:
      self.label = self.name
    self.enable = enable

    super().__init__(name=name)

  def __call__(self, x):
    if self.enable:
      print(f"<{self.label}> {x} </{self.label}>")


def compare_elements(x, y):
  return (jnp.equal(x, y)) | (jnp.isnan(x) & jnp.isnan(y))


def find_differences(tree1, tree2):
  flat_tree1, treedef1 = jax.tree_util.tree_flatten(tree1)
  flat_tree2, treedef2 = jax.tree_util.tree_flatten(tree2)

  if treedef1 != treedef2:
    raise ValueError("Input pytrees have different structures.")

  diff_flat_tree = [(compare_elements(x, y), x, y)
                    for x, y in zip(flat_tree1, flat_tree2)]
  diff_tree = jax.tree_util.tree_unflatten(treedef1, diff_flat_tree)

  return diff_tree
