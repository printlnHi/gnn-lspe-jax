from typing import Any, Tuple, List

from type_aliases import LabelledGraphs


class MoleculeJraphDataset:
  def __init__(self, train: LabelledGraphs, test: LabelledGraphs,
               val: LabelledGraphs, num_atom_type: int, num_bond_type: int):
    self.train = train
    self.test = test
    self.val = val
    self.num_atom_type = num_atom_type
    self.num_bond_type = num_bond_type
