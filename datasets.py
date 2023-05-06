"""
The file for dataset conversion, storage and loading
If the file is run as a script, it will convert the datasets to MoleculeJraphDataset instances and save them to disk.
"""


import pickle
from typing import Iterable, List, Tuple, Union

import dgl
import jax.tree_util as tree
import jraph
import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from data.molecules import MoleculeDataset, MoleculeDGL
from data.ogb_mol import OGBMOLDGL, OGBMOLDataset
from jraph_data.molecule_jraph_dataset import MoleculeJraphDataset
from types_and_aliases import LabelledGraph

dataset_names_lowercase = ['mutag', 'zinc', 'moltox21']
local_paths = {
  name: f'jraph_data/jraph_{name}.pickle' for name in dataset_names_lowercase}


def load(name) -> MoleculeJraphDataset:
  """Load a MoleculeJraphDataset from disk"""
  print(f'Loading {name} from {local_paths[name]}')
  with open(local_paths[name], 'rb') as f:
    ds = pickle.load(f)
  return ds


def DGLGraph_to_GraphsTuple(dgl_graph: dgl.DGLGraph) -> jraph.GraphsTuple:
  """Convert a DGLGraph to a jraph.GraphsTuple."""
  senders, receivers = map(lambda x: x.numpy(), dgl_graph.edges())
  nodes = tree.tree_map(lambda x: x.numpy(), dict(dgl_graph.ndata))
  edges = tree.tree_map(lambda x: x.numpy(), dict(dgl_graph.edata))
  return jraph.GraphsTuple(
    nodes=nodes,
    edges=edges,
    receivers=receivers,
    senders=senders,
    globals=None,
    n_node=dgl_graph.batch_num_nodes().numpy(),
    n_edge=dgl_graph.batch_num_edges().numpy())


def convert_labelled_DGLGraphs(ds: Iterable[Tuple[dgl.DGLGraph, torch.Tensor]]) -> List[LabelledGraph]:
  return [(DGLGraph_to_GraphsTuple(graph), label.numpy()) for graph, label in ds]


if __name__ == '__main__':
  print("====Converting and saving datasets====")
  print("==Converting MUTAG==")
  with open('data/molecules/mutag.pickle', 'rb') as f:
    ds = pickle.load(f)
  ds = [(d["input_graph"], d["target"]) for d in ds]
  train, val, test = ds[:150], ds[150:-10], ds[-10:]
  jraphDataset = MoleculeJraphDataset(
    train, val, test, atom_feature_dims=[7], bond_feature_dims=[4])
  with open(local_paths['mutag'], 'wb') as f:
    pickle.dump(jraphDataset, f)

  print("==Converting ZINC==")
  ds = MoleculeDataset('ZINC')
  train, val, test = map(convert_labelled_DGLGraphs,
                         [ds.train, ds.val, ds.test])

  jraphDataset = MoleculeJraphDataset(train, val, test, atom_feature_dims=[
                                      ds.num_atom_type], bond_feature_dims=[ds.num_bond_type])
  with open(local_paths['zinc'], 'wb') as f:
    pickle.dump(jraphDataset, f)

  print("==Converting MOLTOX21==")
  ds = OGBMOLDataset("OGBG-MOLTOX21")
  train, val, test = map(convert_labelled_DGLGraphs,
                         [ds.train, ds.val, ds.test])

  jraphDataset = MoleculeJraphDataset(train, val, test, atom_feature_dims=get_atom_feature_dims(
  ), bond_feature_dims=get_bond_feature_dims(), num_classes=ds.dataset.num_tasks)
  with open(local_paths['moltox21'], 'wb') as f:
    pickle.dump(jraphDataset, f)
