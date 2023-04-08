from typing import Any, Tuple, List

from type_aliases import LabelledGraph


class MoleculeJraphDataset:
  def __init__(self, train: List[LabelledGraph], test: List[LabelledGraph],
               val: List[LabelledGraph], num_atom_type: int, num_bond_type: int):
    self.train = train
    self.test = test
    self.val = val
    self.num_atom_type = num_atom_type
    self.num_bond_type = num_bond_type
