import pickle
from utils import pad_all
from data.MoleculeJraphDataset import MoleculeJraphDataset


def mutag():
  with open('data/mutag.pickle', 'rb') as f:
    ds = pickle.load(f)
  ds = [(d["input_graph"], d["target"]) for d in ds]
  ds = pad_all(ds)
  return (ds[:150], ds[150:-10], ds[-10:])


def zinc() -> MoleculeJraphDataset:
  with open('data/zinc_jraph.pickle', 'rb') as f:
    ds = pickle.load(f)
  return ds
