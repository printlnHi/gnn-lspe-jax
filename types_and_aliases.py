from typing import Any, Callable, Dict, Iterable, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax

LabelledGraph = Tuple[jraph.GraphsTuple, Any]
LoadedData = Iterable[Tuple[LabelledGraph, int]]

Metrics = Dict[str, float]

MutagEvaluateFn = Callable[[hk.Params, List[LabelledGraph]], Metrics]
MutagTrainResult = Tuple[hk.Params, optax.OptState, Metrics]
MutagTrainFn = Callable[[hk.Params,
                         optax.OptState,
                         optax.TransformUpdateFn,
                         List[LabelledGraph]],
                        MutagTrainResult]

TrainResult = Tuple[hk.Params, hk.State, optax.OptState, Metrics]

GraphClassifierInput: list[type] = [jraph.GraphsTuple, bool]
GraphClassifierOutput = Tuple[jnp.ndarray, jraph.GraphsTuple]

