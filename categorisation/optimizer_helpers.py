"""Utility helpers for optimiser wrappers shared across categorisation scripts."""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
import tensorflow as tf

ArrayLike = np.ndarray


def pack_trainable_variables(trainable_vars: Sequence[tf.Variable]) -> Tuple[np.ndarray, Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
    """Flatten TF variables into a single vector along with shape metadata."""
    shapes = tuple(tuple(int(dim) for dim in v.shape) for v in trainable_vars)
    sizes = tuple(int(np.prod(shape)) for shape in shapes)
    flat = np.concatenate([v.numpy().reshape(-1) for v in trainable_vars]).astype(np.float64)
    return flat, shapes, sizes


def assign_trainable_variables(
    trainable_vars: Sequence[tf.Variable],
    vector: ArrayLike,
    shapes: Sequence[Tuple[int, ...]],
    sizes: Sequence[int],
    *,
    step_cap: float | None = None,
) -> None:
    """Assign values from a flat vector back into TF variables.

    If ``step_cap`` is provided, each component update is clipped to
    ``[-step_cap, +step_cap]`` in the flattened space before assignment.
    """
    offset = 0
    for var, shape, size in zip(trainable_vars, shapes, sizes):
        new_flat = np.asarray(vector[offset:offset + size], dtype=np.float64)
        offset += size
        old_flat = var.numpy().reshape(-1)
        if step_cap is not None:
            delta = np.clip(new_flat - old_flat, -step_cap, step_cap)
            new_flat = old_flat + delta
        var.assign(new_flat.reshape(shape).astype(np.float32))


def lbfgs_with_restarts(
    loss_and_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    initial_weights: np.ndarray,
    *,
    options: dict,
    restarts: int = 1,
    assign_initial: Callable[[np.ndarray], None] | None = None,
    status_callback: Callable[[int, object], None] | None = None,
    jitter_sigma: float | None = None,
):
    """Run L-BFGS-B with a few restarts to mitigate line-search failures."""
    current = np.array(initial_weights, dtype=np.float64)
    result = None
    for attempt in range(max(1, restarts)):
        print(f"LBFGS attempt {attempt + 1}/{max(1, restarts)}")
        if assign_initial is not None:
            assign_initial(current)
        result = minimize(
            loss_and_grad,
            current,
            jac=True,
            method="L-BFGS-B",
            options=options,
        )
        if status_callback is not None:
            status_callback(attempt, result)
        if result.status not in (2, 3):
            current = np.array(result.x, dtype=np.float64)
            if jitter_sigma and attempt + 1 < restarts:
                current = current + np.random.normal(scale=jitter_sigma, size=current.shape)
                continue
            break
        current = np.array(result.x, dtype=np.float64)
    return result

