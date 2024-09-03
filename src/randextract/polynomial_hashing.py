import math
import warnings
from numbers import Integral, Real

import galois
import gmpy2
import numpy as np
from galois import GF, GF2
from scipy.optimize import root_scalar

from ._verify import (
    verify_array,
    verify_kwargs,
    verify_number_in_interval,
    verify_number_is_positive,
    verify_number_type,
    verify_type,
)
from .randomness_extractor import RandomnessExtractor
from .utilities.converter import binary_array_to_integer, integer_to_binary_array

CTX = gmpy2.get_context()
CTX.precision = 256

ZERO = gmpy2.mpfr("0.0")
ONE = gmpy2.mpfr("1.0")


@RandomnessExtractor.register_subclass("polynomial")
class PolynomialOneBitExtractor(RandomnessExtractor):
    r"""
    Implementation class for the polynomial one-bit randomness extractor.

    This one-bit extractor is actually a concatenation of two hash functions, each of which uses half of the given seed.
    The first hash is a polynomial evaluation and the second hash is the parity of a bitwise multiplication. For a more
    rigorous definition and a detailed example check :ref:`the corresponding section <Polynomial hashing one-bit
    extractor>` in the theory.

    Arguments:
        input_length: Length of the bit string from the weak random source. It must be an integer greater than 1.
        seed_length: Length of the seed. It must be an even integer in the closed interval [2, 2 * input_length].
        irreducible_poly: An irreducible polynomial used for the arithmetic calculation in the Reed Solomon hashing. It
            must be a ``galois.Poly`` object. It must be provided for seed_length larger than 20_001 bits.

    Examples:
        A :obj:`PolynomialOneBitExtractor` object can be created directly calling the constructor of this class or using
        the factory class :obj:`RandomnessExtractor`. Both methods are equivalent.

        .. code-block:: python

            from randextract import RandomnessExtractor, PolynomialOneBitExtractor

            ext1 = RandomnessExtractor.create(
                    extractor_type="polynomial",
                    input_length=2**20,
                    seed_length=100)

            ext2 = XOROneBitExtractor(
                    input_length=2**20,
                    seed_length=100)

            assert ext1.output_length == ext2.output_length == 1
            assert ext1.seed_length == ext2.seed_length == 100

        You can run the example from the documentation

        .. code-block:: python

            from galois import GF2, Poly

            ext = XOROneBitExtractor(
                    input_length=25,
                    seed_length=24,
                    irreducible_poly=Poly.Str("x^12 + x^3 + 1"))

            extractor_input = GF2([1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
            seed = GF2([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1])

            ext.extract(extractor_input, seed)

    See Also:
        Theory: :ref:`Polynomial hashing one-bit extractor`
    """

    def __init__(
        self,
        input_length: Integral,
        seed_length: Integral,
        irreducible_poly: galois.Poly | None = None,
    ):
        verify_number_type(input_length, Integral)
        verify_number_in_interval(input_length, 2, math.inf, "right-open")
        self._input_length = int(input_length)

        verify_number_type(seed_length, Integral)
        verify_number_in_interval(seed_length, 2, 2 * self._input_length, "closed")
        if seed_length % 2 != 0:
            warnings.warn(
                "seed_length must be even. Using seed_length + 1.", UserWarning
            )
            seed_length += 1

        self._seed_length = int(seed_length)
        self._size_block = self._seed_length // 2

        if self._size_block > 10_000 and irreducible_poly is None:
            raise ValueError(
                "An irreducible polynomial must be provided for seed_length > 20_001."
            )

        if self._size_block == 1:
            self._gf = GF2
            if irreducible_poly is not None:
                warnings.warn("Provided irreducible polynomial ignored.", UserWarning)
            self._irreducible_poly = GF2.irreducible_poly
            return

        if irreducible_poly is not None:
            verify_type(irreducible_poly, galois.Poly)
            self._irreducible_poly = irreducible_poly
            self._gf = GF(
                2**self._size_block,
                irreducible_poly=self._irreducible_poly,
            )
        else:
            self._irreducible_poly = galois.irreducible_poly(
                2, self._size_block, terms="min", method="min"
            )
            self._gf = GF(
                2**self._size_block,
                irreducible_poly=self._irreducible_poly,
                verify=False,
            )

    def _reed_solomon_hashing(self, array: GF2, seed: GF2) -> GF2:
        _seed = self._gf(binary_array_to_integer(seed))
        number_blocks = math.ceil(self._input_length / self._size_block)

        x = GF2.Zeros(self._size_block * number_blocks)
        x[: self._input_length] = array
        x = x.reshape(number_blocks, self._size_block)
        output = self._gf(0)

        for i in range(number_blocks):
            xi = self._gf(binary_array_to_integer(x[i]))
            output += xi * _seed ** (number_blocks - i - 1)

        return integer_to_binary_array(output.item(), pad=self._size_block)

    @staticmethod
    def _hadamard_hashing(array: GF2, seed: GF2) -> GF2:
        return array[seed == 1].sum()

    @staticmethod
    def calculate_length(extractor_type: str, input_length: Integral, **kwargs) -> int:
        r"""
        For a given extractor type (i.e., quantum-proof) and a set of parameters, it computes the optimal seed length
        for the polynomial one-bit extractor. This extractor is a combination of two almost two-universal extractors.

        Arguments:
            extractor_type: This determines the type of side information against which the extractor should remain
                secure. Two values are accepted: "quantum" and "classical".
            input_length: The length of the bit string from the weak random source.

        Keyword Arguments:
            relative_source_entropy: Lower bound on the conditional min-entropy of the weak random source, normalized by
                the input length. It must be a real number in the range (0, 1].
            error_bound: Upper bound on the randomness extractor's error, i.e., a measure of how far the output of the
                extractor is from the ideal (uniform) output. It must be a real number in the range (0, 1].

        See Also:
            Theory: :doc:`theory/trevisan`.
        """
        verify_type(extractor_type, str)
        if extractor_type not in ["quantum", "classical"]:
            raise ValueError(
                f'extractor_type should be "quantum" or "classical", but {extractor_type} was passed.'
            )

        verify_number_type(input_length, Integral)
        verify_number_is_positive(input_length)

        verify_kwargs(kwargs, ["relative_source_entropy", "error_bound"])

        relative_source_entropy = kwargs.get("relative_source_entropy")
        verify_number_type(relative_source_entropy, Real)
        verify_number_in_interval(relative_source_entropy, 0, 1, "left-open")

        error_bound = kwargs.get("error_bound")
        verify_number_type(error_bound, Real)
        verify_number_in_interval(error_bound, 0, 1, "left-open")

        # p. 11 arXiv 1212.0520
        def _delta(n, l):
            n = gmpy2.mpz(n)
            l = gmpy2.mpz(l)
            s = gmpy2.mpz(gmpy2.ceil(n / l))
            return gmpy2.mpfr("0.5") + (s - ONE) / CTX.pow(gmpy2.mpz(2), l)

        # Expressions obtained from Eqs. 4 and 10 from https://doi.org/10.1109/TIT.2011.2158473
        def _err(n, l, k):
            # We use gmpy2 because double precision is not good enough to compute this correctly
            n = gmpy2.mpz(n)
            k = gmpy2.mpfr(str(k))
            l = gmpy2.mpz(l)
            if extractor_type == "classical":
                term1 = gmpy2.mpz(2) * _delta(n, l) - gmpy2.mpz(1)
                term2 = CTX.pow(2, 1 - k * n)
                # This can be greater than 1. We could return exactly 1 in all these
                # cases, but we returned the actual value so that the function is smooth
                return gmpy2.mpfr("0.5") * gmpy2.sqrt(term1 + term2)
            if extractor_type == "quantum":
                term1 = (gmpy2.mpz(2) * _delta(n, l) - gmpy2.mpz(1)) ** 2
                term2 = gmpy2.mpz(9) * CTX.pow(2, 1 - k * n)
                # This can be greater than 1. We could return exactly 1 in all these
                # cases, but we returned the actual value so that the function is smooth
                return gmpy2.root(term1 + term2, 4)

        # It is not always possible to extract a bit with the desired error bound.
        # This checks that we are in a valid region.
        if extractor_type == "classical":
            if (
                relative_source_entropy
                <= 2 * math.log2(1 / error_bound) / input_length - 1 / input_length
            ):
                raise ValueError(
                    "It is not possible to find valid number of indices for this input length and error bound. Either "
                    "increase the input length or allow for higher errors."
                )

        else:
            if (
                relative_source_entropy
                <= 1 / input_length
                + 2 * math.log2(3) / input_length
                - 4 * math.log2(error_bound) / input_length
            ):
                raise ValueError(
                    "It is not possible to find valid number of indices for this input length and error bound. Either "
                    "increase the input length or allow for higher errors."
                )

        half_seed_length = 1
        while (
            _err(input_length, half_seed_length, relative_source_entropy) > error_bound
        ):
            half_seed_length += 1

        return 2 * half_seed_length

    def extract(self, extractor_input: np.ndarray | GF2, seed: np.ndarray | GF2) -> GF2:
        verify_array(extractor_input, 1, self._input_length, Integral, [0, 1, "closed"])
        verify_array(seed, 1, self._seed_length, Integral, [0, 1, "closed"])

        extractor_input = GF2(extractor_input)
        seed = GF2(seed)

        return self._hadamard_hashing(
            self._reed_solomon_hashing(extractor_input, seed[: self._seed_length // 2]),
            seed[self._seed_length // 2 :],
        )

    @property
    def input_length(self) -> int:
        return self._input_length

    @property
    def seed_length(self) -> int:
        return self._seed_length

    @property
    def output_length(self) -> int:
        return 1

    @property
    def irreducible_poly(self) -> galois.Poly:
        return self._irreducible_poly
