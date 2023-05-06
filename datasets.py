import pickle

from data.molecule_jraph_dataset import MoleculeJraphDataset
from lib.padding import monotonic_power_of_two_padding, pad_labelled_graph


def mutag():
  with open('data/mutag.pickle', 'rb') as f:
    ds = pickle.load(f)
  ds = [(d["input_graph"], d["target"]) for d in ds]
  ds = [pad_labelled_graph(d, monotonic_power_of_two_padding) for d in ds]
  return (ds[:150], ds[150:-10], ds[-10:])


def zinc() -> MoleculeJraphDataset:
  with open('data/zinc_jraph.pickle', 'rb') as f:
    ds = pickle.load(f)
  return ds


def moltox21() -> MoleculeJraphDataset:
  with open('data/moltox21_jraph.pickle', 'rb') as f:
    ds = pickle.load(f)
  return ds
