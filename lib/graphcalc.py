import jax.numpy as jnp
import jraph
import numpy as np


def graphLaplacian(graph: jraph.GraphsTuple, np_=np) -> np.ndarray:
  nodes, edges, senders, receivers, globals, n_node, n_edge = graph
  dim = nodes['feat'].shape[0]
  A = np_.zeros((dim, dim))
  if np_ == jnp:
    A = A.at[senders, receivers].set(1)
    in_degrees = np_.bincount(senders, length=dim)
  else:
    A[senders, receivers] = 1
    in_degrees = np_.bincount(senders, minlength=dim)
  in_degrees = np_.clip(in_degrees, 1, None)
  N = np_.diag(in_degrees ** -0.5)
  D = np_.eye(dim)
  L = D - N @ A @ N
  return L


def lapPE(graph: jraph.GraphsTuple, pos_enc_dim: int = 8, np_=np):
  L = graphLaplacian(graph, np_)
  eigValues, eigVectors = np_.linalg.eig(L)
  idx = eigValues.argsort()
  eigValues, eigVectors = eigValues[idx], eigVectors[:, idx]
  # All vectors should be real, should I check this?
  eigVectors = np_.real(eigVectors)
  pe = eigVectors[:, 1:pos_enc_dim + 1]
  return pe


def RWPE(graph: jraph.GraphsTuple, pos_enc_dim: int = 8) -> np.ndarray:
  nodes, edges, senders, receivers, globals, n_node, n_edge = graph
  dim = nodes['feat'].shape[0]
  A = np.zeros((dim, dim))
  A[senders, receivers] = 1
  D = np.diag(np.clip(np.sum(A, axis=1), a_min=1, a_max=None))
  RW = A @ np.linalg.inv(D)
  # PE is diagonals of RW, RW^2, ..., RW^pos_enc_dim
  RW_exp = RW
  diagonals = []
  for i in range(pos_enc_dim):
    diagonals.append(np.diag(RW_exp))
    RW_exp = RW_exp @ RW
  pe = np.stack(diagonals, axis=1)
  return pe
