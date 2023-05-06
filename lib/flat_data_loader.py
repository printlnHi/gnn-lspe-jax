import time
from typing import List

import jraph
import numpy as np

from lib.padding import pad_labelled_graph
from types_and_aliases import LabelledGraph, LoadedData


def flat_data_loader(dataset, batch_size, padding_strategy, rng, print_time=False) -> LoadedData:
  start_time = time.time()
  n = len(dataset)
  length = (n + batch_size - 1) // batch_size
  if rng is not None:
    rng = np.random.default_rng(int(rng[0]))
    rng.shuffle(dataset)
  shuffle_time = time.time()

  graphs, labels = zip(*dataset)
  graphs = [jraph.batch_np(graphs[i * batch_size:(i + 1) * batch_size])
            for i in range(length)]
  if labels[0].shape == (1, ):
    labels = [np.concatenate(
        labels[i * batch_size:(i + 1) * batch_size], axis=0) for i in range(length)]
  else:
    labels = [np.array(
        labels[i * batch_size:(i + 1) * batch_size]) for i in range(length)]
  lengths = [
    batch_size if i < length -
    1 else n -
    i *
      batch_size for i in range(length)]
  unpadded_time = time.time()

  labelled_graphs: List[LabelledGraph] = [
      pad_labelled_graph(
          (graph, label), padding_strategy) for graph, label in zip(
          graphs, labels)]
  batches = list(zip(labelled_graphs, lengths))
  batches_time = time.time()
  if print_time:
    print(f"total time: {batches_time - start_time} = shuffle {shuffle_time-start_time} + unpadded: {unpadded_time - shuffle_time} + batches: {batches_time - unpadded_time}")
  return batches
