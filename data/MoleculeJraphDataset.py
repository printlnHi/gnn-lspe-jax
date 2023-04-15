import functools
from typing import Any, Tuple, List

from type_aliases import LabelledGraph
import jax.numpy as jnp


class MoleculeJraphDataset:
  def __init__(self, train: List[LabelledGraph], test: List[LabelledGraph],
               val: List[LabelledGraph], num_atom_type: int, num_bond_type: int):
    self.train = train
    self.test = test
    self.val = val
    self.num_atom_type = num_atom_type
    self.num_bond_type = num_bond_type

  def add_PE(self, pe_func, keys):
    def _add_PE(labelledGraph: LabelledGraph) -> LabelledGraph:
      graph, label = labelledGraph
      pe = pe_func(graph)
      nodes = graph.nodes
      for key in keys:
        nodes[key] = pe
      return (graph._replace(nodes=nodes), label)
    self.train = list(map(_add_PE, self.train))
    self.test = list(map(_add_PE, self.test))
    self.val = list(map(_add_PE, self.val))
