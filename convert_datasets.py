import pickle
from typing import List, Tuple

import dgl
import jax.tree_util as tree
import jraph
import numpy as np

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from data.molecules import MoleculeDataset
from data.ogb_mol import OGBMOLDataset
from data.molecule_jraph_dataset import MoleculeJraphDataset


def convert_to_jraph(dgl_graph: dgl.DGLGraph) -> jraph.GraphsTuple:
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


def convert(ds) -> List[Tuple[jraph.GraphsTuple, np.ndarray]]:
  return [(convert_to_jraph(graph), label.numpy()) for graph, label in ds]


ds = MoleculeDataset('ZINC')
jraphDataset = MoleculeJraphDataset(convert(ds.train), convert(ds.val), convert(
  ds.test), atom_feature_dims=[ds.num_atom_type], bond_feature_dims=[ds.num_bond_type])
with open('data/zinc_jraph.pickle', 'wb') as f:
  pickle.dump(jraphDataset, f)


ds = OGBMOLDataset("OGBG-MOLTOX21")
jraphDataset = MoleculeJraphDataset(
  convert(ds.train), convert(ds.val), convert(ds.test), atom_feature_dims=get_atom_feature_dims(), bond_feature_dims=get_bond_feature_dims(), num_classes=ds.dataset.num_tasks)
with open('data/moltox21_jraph.pickle', 'wb') as f:
  pickle.dump(jraphDataset, f)
