import math
import warnings
from numbers import Real

import numpy as np
from scipy.optimize import minimize_scalar

from .._verify import (
    verify_array,
    verify_number_in_interval,
    verify_number_type,
    verify_type,
)


def binary_entropy(prob: Real | np.ndarray) -> float | np.ndarray:
    """
    Compute the binary entropy, i.e., h(p) = -p * log2(p) - (1-p) * log2(p). By convention, h(0) = h(1) = 0. It works
    both with a single number or with an array.
    """
    verify_type(prob, [Real, np.ndarray])

    if isinstance(prob, Real):
        verify_number_type(prob, Real)
        verify_number_in_interval(prob, 0, 1, "closed")

        if math.isclose(prob, 0) or math.isclose(prob, 1):
            return 0

        return -prob * math.log2(prob) - (1 - prob) * math.log2(1 - prob)

    else:
        verify_array(prob, valid_range=[0, 1, "closed"])
        output = np.zeros_like(prob)
        mask = np.logical_not(
            np.logical_or(np.isclose(prob, 0, atol=1e-14), np.isclose(prob, 1))
        )
        output[mask] = -prob[mask] * np.log2(prob[mask]) - (1 - prob[mask]) * np.log2(
            1 - prob[mask]
        )
        return output


def inverse_binary_entropy(val: Real) -> Real:
    """
    Compute the inverse of the binary entropy (range restricted to closed interval [0, 1/2]).
    """
    # TODO: Perhaps this can also be generalized to accept np.ndarray?
    verify_type(val, Real)
    verify_number_type(val, Real)
    verify_number_in_interval(val, 0, 1, "closed")
    val = float(val)

    if math.isclose(val, 0):
        return 0

    if math.isclose(val, 1):
        return 0.5

    # We don't use binary_entropy() to avoid the type and range checks
    def _h(p):
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    def _inverse(x, p):
        return (_h(x) - p) ** 2

    # Hide RuntimeWarning caused by 0 * np.log2(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize_scalar(
            _inverse,
            args=[val],
            method="Bounded",
            bounds=[0, 0.5],
            options={"xatol": 1e-16},
        )

    if res.success:
        return res.x.item()
    else:
        raise ValueError(
            "Computing the inverse of binary entropy failed. This should have not happened. Please open an issue."
        )
