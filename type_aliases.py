from typing import Any, Callable, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax

LabelledGraph = Tuple[jraph.GraphsTuple, Any]

Metrics = Dict[str, float]

MutagEvaluateFn = Callable[[hk.Params, List[LabelledGraph]], Metrics]
MutagTrainResult = Tuple[hk.Params, optax.OptState, Metrics]
MutagTrainFn = Callable[[hk.Params,
                         optax.OptState,
                         optax.TransformUpdateFn,
                         List[LabelledGraph]],
                        MutagTrainResult]

TrainResult = Tuple[hk.Params, hk.State, optax.OptState, Metrics]

GraphClassifierFn = Callable[[jraph.GraphsTuple, bool], jnp.ndarray]

# TODO: Cleanup unused type aliases, mb rename this file to just be types
# if not all are aliases
