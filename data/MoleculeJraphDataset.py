from typing import Any, Tuple, List

from type_aliases import LabelledGraph
from utils import add_lapPE
import jax.numpy as jnp


class MoleculeJraphDataset:
  def __init__(self, train: List[LabelledGraph], test: List[LabelledGraph],
               val: List[LabelledGraph], num_atom_type: int, num_bond_type: int):
    self.train = train
    self.test = test
    self.val = val
    self.num_atom_type = num_atom_type
    self.num_bond_type = num_bond_type

  def add_lap_PEs(self, pos_enc_dim):
    self.train = [(add_lapPE(graph, pos_enc_dim), label)
                  for graph, label in self.train]
    self.test = [(add_lapPE(graph, pos_enc_dim), label)
                 for graph, label in self.test]
    self.val = [(add_lapPE(graph, pos_enc_dim), label)
                for graph, label in self.val]
