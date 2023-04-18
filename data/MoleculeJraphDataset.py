import functools
from typing import Any, Callable, Optional, Tuple, List, Union

from type_aliases import LabelledGraph
import jax.numpy as jnp
import numpy as np


class MoleculeJraphDataset:
  def __init__(self, train: List[LabelledGraph], test: List[LabelledGraph],
               val: List[LabelledGraph], num_atom_type: Optional[int] = None, num_bond_type: Optional[int] = None, num_classes: Optional[int] = None):
    self.train = train
    self.test = test
    self.val = val
    self.num_atom_type = num_atom_type
    self.num_bond_type = num_bond_type
    self.num_classes = num_classes

  def _map_across_graphs(self, fn: Callable[[LabelledGraph], LabelledGraph]):
    self.train = list(map(fn, self.train))
    self.test = list(map(fn, self.test))
    self.val = list(map(fn, self.val))

  def add_PE(self, pe_func, keys):
    def _add_PE(labelledGraph: LabelledGraph) -> LabelledGraph:
      graph, label = labelledGraph
      pe = pe_func(graph)
      nodes = graph.nodes
      for key in keys:
        nodes[key] = pe
      return (graph._replace(nodes=nodes), label)
    self._map_across_graphs(_add_PE)

  def add_norms(self):
    def _add_norm(labelledGraph: LabelledGraph) -> LabelledGraph:
      graph, label = labelledGraph
      nodes = graph.nodes
      n_node = graph.n_node
      sum_n_node = np.sum(n_node)
      graph_indicies = np.repeat(np.arange(n_node.shape[0]), n_node)
      nodes['snorm_n'] = np.sqrt(1 / n_node[graph_indicies])
      return (graph._replace(nodes=nodes), label)
    self._map_across_graphs(_add_norm)
