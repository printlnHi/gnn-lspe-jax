from typing import Any, Callable, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax

LabelledGraph = Tuple[jraph.GraphsTuple, Any]
LabelledGraphs = List[LabelledGraph]

Metrics = Dict[str, float]

MutagEvaluateFn = Callable[[hk.Params, LabelledGraphs], Metrics]
MutagTrainResult = Tuple[hk.Params, optax.OptState, Metrics]
MutagTrainFn = Callable[[hk.Params,
                         optax.OptState,
                         optax.TransformUpdateFn,
                         LabelledGraphs],
                        MutagTrainResult]

EvaluateFn = Callable[[hk.Params, hk.State, LabelledGraphs], Metrics]
TrainResult = Tuple[hk.Params, hk.State, optax.OptState, Metrics]
TrainFn = Callable[[hk.Params,
                   hk.State,
                   jax.random.KeyArray,
                   optax.OptState,
                   optax.TransformUpdateFn,
                   LabelledGraphs],
                   TrainResult]

GraphClassifierFn = Callable[[jraph.GraphsTuple, bool], jnp.ndarray]
