import pickle
from typing import List, Tuple

import dgl
import jax.tree_util as tree
import jraph
import numpy as np
from torch.utils.data import DataLoader

from data.molecules import MoleculeDataset, MoleculeDGL
from data.MoleculeJraphDataset import MoleculeJraphDataset

ds = MoleculeDataset('ZINC')


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


def convert(ds: MoleculeDGL) -> List[Tuple[jraph.GraphsTuple, np.ndarray]]:
  return [(convert_to_jraph(graph), label.numpy()) for graph, label in ds]


jraphDataset = MoleculeJraphDataset(convert(ds.train), convert(ds.val),
                                    convert(ds.test), num_atom_type=ds.num_atom_type, num_bond_type=ds.num_bond_type)

with open('data/zinc_jraph.pickle', 'wb') as f:
  pickle.dump(jraphDataset, f)
