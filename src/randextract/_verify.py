import math
from numbers import Integral
from pathlib import Path

import numpy as np


def verify_type(var, valid_types):
    """
    Verifies that type(var) is any of the given types in the list valid_types.
    """
    if not isinstance(valid_types, list):
        valid_types = [valid_types]

    valid = False
    for type_ in valid_types:
        if isinstance(var, type_):
            valid = True
            break

    if not valid:
        raise TypeError(
            f"Given variable must of the type(s) {valid_types}, not {type(var)}."
        )


def verify_number_type(number, valid_type):
    """
    Verifies that the number is an instance of the valid_type.
    """
    if not isinstance(number, valid_type):
        raise TypeError(f"Given number must be a(n) {valid_type}, not {type(number)}.")


def verify_number_is_positive(number):
    """
    Verifies that number > 0.
    """
    if not number > 0:
        raise ValueError(f"Given number must be positive, but it is {number} <= 0.")


def verify_number_is_non_negative(number):
    """
    Verifies that number >= 0.
    """
    if not number >= 0:
        raise ValueError(f"Given number must be non-negative, but it is {number} < 0.")


def verify_number_in_interval(number, low, high, interval_type):
    """
    Verifies that number is in any of the intervals (low, high), [low, high], (low, high], or [low, high).
    """
    assert low <= high
    assert interval_type in ["open", "closed", "left-open", "right-open"]

    if interval_type == "open":
        if not low < number < high:
            raise ValueError(f"Given number is not in open interval ({low}, {high}).")
    elif interval_type == "closed":
        if not low <= number <= high:
            raise ValueError(f"Given number is not in closed interval [{low}, {high}].")
    elif interval_type == "left-open":
        if not low < number <= high:
            raise ValueError(
                f"Given number is not in left-open interval ({low}, {high}]."
            )
    elif interval_type == "right-open":
        if not low <= number < high:
            raise ValueError(
                f"Given number is not in right-open interval [{low}, {high})."
            )


def verify_array(
    array, valid_dim=None, valid_size=None, valid_type=None, valid_range=None
):
    """
    Verifies that an array has the correct dimension, size, compatible dtype and/or elements from a given range.
    valid_range must be passed as a list [low, high, interval_type], where interval_type is the same as in
    verify_number_in_interval().
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Given array must be a NumPy array, not {type(array)}.")

    if valid_dim is not None:
        if array.ndim != valid_dim:
            raise ValueError(
                f"Given array must be a {valid_dim}-dim array, but a {array.ndim}-dim array was given."
            )

    if valid_size is not None:
        if array.size != valid_size:
            raise ValueError(
                f"Size of array must be {valid_size}, but it is {array.size}."
            )

    if valid_type is not None:
        if not issubclass(array.dtype.type, valid_type):
            raise TypeError(
                f"Given array must have a dtype compatible with {valid_type}, but it is {array.dtype}."
            )

    if valid_range is not None:
        assert isinstance(valid_range, list)
        assert len(valid_range) == 3
        low = valid_range[0]
        high = valid_range[1]
        interval_type = valid_range[2]
        assert low <= high
        assert interval_type in ["open", "closed", "left-open", "right-open"]

        if interval_type == "open":
            if not low < array.min() <= array.max() < high:
                raise ValueError(
                    f"Given array has values outside open interval ({low}, {high})."
                )
        elif interval_type == "closed":
            if not low <= array.min() <= array.max() <= high:
                raise ValueError(
                    f"Given array has values outside closed interval [{low}, {high}]."
                )
        elif interval_type == "left-open":
            if not low < array.min() <= array.max() <= high:
                raise ValueError(
                    f"Given array has values outside left-open interval ({low}, {high}]."
                )
        elif interval_type == "right-open":
            if not low <= array.min() <= array.max() < high:
                raise ValueError(
                    f"Given array has values outside right-open interval [{low}, {high})."
                )


def verify_kwargs(kwargs: dict, args: list = None, optional_args: list = None):
    """
    Verifies that the user only passed valid arguments to kwargs from the args and optional_args lists.
    It will raise ValueError if an unknown arg is passed, or if a compulsory arg (from the args list) is missing.
    """
    if args is not None:
        for arg in args:
            if arg not in kwargs.keys():
                raise ValueError(f"{arg} is required but it is missing.")

    if optional_args is None:
        return

    for arg in kwargs.keys():
        if args is not None:
            if arg in args:
                continue
        if arg not in optional_args:
            raise ValueError(
                f"{arg} is not a valid kwarg for this method. Check the docstring for more information."
            )
