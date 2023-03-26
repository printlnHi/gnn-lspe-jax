from typing import Any, Tuple, List

from jraph import GraphsTuple

DatasetSplit = List[Tuple[GraphsTuple, Any]]


class MoleculeJraphDataset:
  def __init__(self, train: DatasetSplit, test: DatasetSplit,
               val: DatasetSplit, num_atom_type: int, num_bond_type: int):
    self.train = train
    self.test = test
    self.val = val
    self.num_atom_type = num_atom_type
    self.num_bond_type = num_bond_type
