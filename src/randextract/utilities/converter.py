import math
import sys

import numpy as np
from galois import GF2


def big_endian_binary_vector_to_int(vector: np.ndarray) -> int:
    assert isinstance(vector, np.ndarray), "vector must be a numpy array."
    assert vector.ndim == 1, "vector must be a 1 dimensional array."
    assert vector.size > 0, "vector must have at least one element."
    big_endian_powers_of_2 = 2 ** np.arange(vector.size)[::-1]
    # To avoid a TypeError we convert GF array vector to a np.ndarray
    int_value = np.array(vector).dot(big_endian_powers_of_2)
    return int_value


def error_bound_and_error_bound_per_bit_conversion(
    output_length: int,
    error_bound: float | None = None,
    error_bound_per_bit: float | None = None,
) -> (float, float):
    assert isinstance(output_length, int)
    assert (
        error_bound is not None or error_bound_per_bit is not None
    ), "either error_bound or error_bound_per_bit has to be specified."
    if error_bound is not None:
        # if error_bound is specified, ignore error_bound_per_bit
        assert isinstance(error_bound, float)
        assert 0 < error_bound < 1
        error_bound_per_bit = error_bound / output_length
    else:
        # use error_bound_per_bit
        assert isinstance(error_bound_per_bit, float)
        assert 0 < error_bound_per_bit < 1
        error_bound = error_bound_per_bit * output_length
        assert (
            error_bound < 1
        ), f"error_bound_per_bit={error_bound_per_bit} is too large for desired output_length={output_length}"

    return error_bound, error_bound_per_bit


def entropy_extraction_ratio_and_output_length_conversion(
    relative_source_entropy: float,
    input_length: int,
    entropy_extraction_ratio: float | None = None,
    output_length: int | None = None,
) -> (float, int):
    assert isinstance(relative_source_entropy, float)
    assert isinstance(input_length, int)
    assert (
        entropy_extraction_ratio is not None or output_length is not None
    ), "either entropy_extraction_ratio or output_length has to be specified."

    if entropy_extraction_ratio is not None:
        # if entropy_extraction_ratio is specified, ignore output_length
        assert isinstance(entropy_extraction_ratio, float)
        assert 0 < entropy_extraction_ratio < 1
        output_length = math.floor(entropy_extraction_ratio * relative_source_entropy * input_length)
    else:
        # use output_length
        assert isinstance(output_length, int), "output_length must be an int."
        assert output_length > 0, "output_length must be at least 1."
        assert output_length <= math.floor(
            relative_source_entropy * input_length
        ), "output_length must be at most the available absolute source entropy."
        entropy_extraction_ratio = output_length / (relative_source_entropy * input_length)

    return entropy_extraction_ratio, output_length


def binary_array_to_integer(array: GF2) -> int:
    assert isinstance(array, GF2)
    assert len(array.shape) <= 1, "input must be one dimensional"

    if array.size == 1:
        return array.item()

    try:
        n = int.from_bytes(np.packbits(np.array(array[::-1]), bitorder="little").tobytes(), "little")
    except ValueError as e:
        # Temporarily disabling limiting conversion size due to CVE-2020-10735
        if not hasattr(sys, "set_int_max_str_digits"):
            raise ValueError(e)
        sys.set_int_max_str_digits(0)
        n = int.from_bytes(np.packbits(np.array(array[::-1]), bitorder="little").tobytes(), "little")
        sys.set_int_max_str_digits(sys.int_info.default_max_str_digits)

    return n


def integer_to_binary_array(num: int, pad: int = None) -> GF2:
    assert isinstance(num, int)

    if pad is not None:
        assert isinstance(pad, int)
        assert pad > 0
    else:
        pad = 0

    return GF2(np.frombuffer(f"{num:0{pad}b}".encode(), dtype=np.uint8) - ord("0"))


def binary_array_to_hex_string(array: GF2) -> str:
    assert isinstance(array, GF2)
    assert len(array.shape) <= 1, "input must be one dimensional"

    size_bytes = math.ceil(array.size / 8)
    return int("".join([str(_) for _ in np.array(array).tolist()]), 2).to_bytes(size_bytes).hex()
