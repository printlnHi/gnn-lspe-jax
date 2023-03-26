from typing import Any, Callable, Dict, List, Tuple

import haiku as hk
import jax.numpy as jnp
import jraph
import optax

Metrics = Dict[str, float]

EvaluateStatelessFn = Callable[[
  hk.Params, List[Dict[str, Any]], jraph.GraphsTuple], Metrics]
TrainStatelessResult = Tuple[hk.Params, optax.OptState, Metrics]
TrainStatelessFn = Callable[[hk.Params,
                             optax.OptState,
                             optax.TransformUpdateFn,
                             List[Dict[str,
                                       Any]],
                             jraph.GraphsTuple],
                            TrainStatelessResult]

EvaluateStatefulFn = Callable[[
    hk.Params, hk.State, List[Dict[str, Any]], jraph.GraphsTuple], Tuple[Metrics, hk.State]]
TrainStatefulResult = Tuple[hk.Params, hk.State, optax.OptState, Metrics]
TrainStatefulFn = Callable[[hk.Params,
                            hk.State,
                            optax.OptState,
                            optax.TransformUpdateFn,
                            List[Dict[str,
                                      Any]],
                            jraph.GraphsTuple],
                           TrainStatefulResult]

GraphClassifierFn = Callable[[jraph.GraphsTuple, bool], jnp.ndarray]
