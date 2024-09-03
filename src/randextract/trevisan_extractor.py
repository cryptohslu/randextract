import itertools
import math
import warnings
from numbers import Integral, Real

import numpy as np
from galois import GF2, FieldArray, is_prime, next_prime

from ._verify import (
    verify_kwargs,
    verify_number_in_interval,
    verify_number_is_positive,
    verify_number_type,
    verify_type,
)
from .randomness_extractor import RandomnessExtractor
from .trevisan.weak_design import WeakDesign


@RandomnessExtractor.register_subclass("trevisan")
class TrevisanExtractor(RandomnessExtractor):
    r"""
    Implementation class for the Trevisan's construction randomness extractor.

    The main idea is that a one-bit extractor can be called multiple times to extract one bit each time. The desired
    guarantee for a randomness extractor is given by a weak design, a family of sets that determines what subset of the
    seed has to be passed for the one-bit extractor in every call. The final output is the concatenation of all these
    extracted bits.

    Different one-bit extractors and weak designs can be used to obtain diverse constructions, which can have distinct
    properties such as required seed or computational complexity.

    For details about the one-bit extractors and weak designs, check their corresponding docstrings. The current
    available one-bit extractor and weak design implementations are:

    One-bit extractors:
        - :obj:`XOROneBitExtractor`
        - :obj:`PolynomialOneBitExtractor`

    Weak designs:
        - :obj:`FiniteFieldPolynomialDesign`
        - :obj:`Block design`

    Arguments:
        input_length: Length of the bit string from the weak random source.
        output_length: Length of the randomness extractor's output.
        weak_design_type: The type weak design of construction. It can be either "finite_field" for the finite field
            polynomial design or "block" for the block weak design.
        one_bit_extractor_type: The type of the one-bit extractor. Use "xor" for the XOR one-bit extractor and
            "polynomial" for the polynomial hashing one-bit extractor.
        one_bit_extractor_seed_length: Length of the seed passed to the one-bit extractor. This, together with the type
            of the weak design, determines the length of the seed of the Trevisan's construction. A value for this
            parameter can be obtained using the ``calculate_length()`` method of the chosen one-bit extractor.
        basic_weak_design_type: (Optional) The basic weak design type when `weak_design_type="block"`. By default, it
            uses "finite_field".
        precomputed_weak_design: (Optional) A family of sets of indices given as a NumPy or Galois array with the
            properties of a weak design. Computing a weak design, specially for large fields, is one of the slowest
            computations when using a Trevisan's extractor. It is recommended to compute it once, save it, and pass it
            later using this argument.

    Examples:
        A :obj:`TrevisanExtractor` object can be created directly calling the constructor of this class or using the
        factory class :obj:`RandomnessExtractor`. Both methods are equivalent. For example, the following code creates
        a Trevisan's construction using the polynomial one-bit extractor and the finite field polynomial weak design.

        .. code-block:: python

            import randextract
            from randextract import RandomnessExtractor, TrevisanExtractor

            ext1 = RandomnessExtractor.create(
                    extractor_type="trevisan",
                    weak_design_type="finite_field",
                    one_bit_extractor_type="polynomial",
                    input_length=2**20,
                    relative_source_entropy=0.8,
                    output_length=2**10,
                    error_bound=1e-3)

            ext2 = TrevisanExtractor(
                    weak_design_type="finite_field",
                    one_bit_extractor_type="polynomial",
                    input_length=2**20,
                    relative_source_entropy=0.8,
                    output_length=2**10,
                    error_bound=1e-3)

            assert ext1.output_length == ext2.output_length
            assert ext1.seed_length == ext2.seed_length

        Trevisan's construction is just an algorithm to create randomness extractor from one-bit extractors and weak
        designs. The required seed length and computational complexity depends on the choice of these two pieces.

        .. code-block:: python

            from galois import GF2

            ext1 = RandomnessExtractor.create(
                    extractor_type="trevisan",
                    weak_design_type="finite_field",
                    one_bit_extractor_type="polynomial",
                    input_length=2**20,
                    relative_source_entropy=0.8,
                    output_length=2**10,
                    error_bound=1e-3)

            ext2 = TrevisanExtractor(
                    weak_design_type="finite_field",
                    one_bit_extractor_type="xor",
                    input_length=2**20,
                    relative_source_entropy=0.8,
                    output_length=2**10,
                    error_bound=1e-3)

            print(f"Trevisan's construction with polynomial one-bit extractor requires a seed of length {ext1.seed_length}")
            print(f"Trevisan's construction with XOR one-bit extractor requires a seed of length {ext2.seed_length}")

            input_array = GF2.Random(ext1.input_length)
            seed1 = GF2.Random(ext1.seed_length)
            seed2 = GF2.Random(ext2.seed_length)

            %timeit (ext1.extract(input_array, seed1))
            %timeit (ext2.extract(input_array, seed2))

    See Also:
        Theory: :doc:`theory/trevisan`.
    """

    def __init__(
        self,
        input_length: Integral,
        output_length: Integral,
        weak_design_type: str,
        one_bit_extractor_type: str,
        one_bit_extractor_seed_length: Integral,
        basic_weak_design_type: str | None = None,
        precomputed_weak_design: np.ndarray | FieldArray | None = None,
    ):
        verify_number_type(input_length, Integral)
        verify_number_is_positive(input_length)
        self._input_length = int(input_length)

        verify_number_type(output_length, Integral)
        verify_number_in_interval(output_length, 1, self._input_length, "closed")
        self._output_length = int(output_length)

        verify_type(weak_design_type, str)
        if weak_design_type not in WeakDesign.subclasses:
            raise ValueError(
                f"{weak_design_type} is not a valid weak design. "
                f"Run randextract.WeakDesign.subclasses.keys() to get the valid types."
            )
        self._weak_design_type = weak_design_type

        verify_type(one_bit_extractor_type, str)
        if one_bit_extractor_type not in ["xor", "polynomial"]:
            raise ValueError(
                f'one_bit_extractor_type should be "xor" or "polynomial", but {weak_design_type} was passed.'
            )
        self._one_bit_extractor_type = one_bit_extractor_type

        verify_number_type(one_bit_extractor_seed_length, Integral)
        # Here we only check that a positive number was passed
        # Further checks are done later by the one-bit extractor and the weak design constructors
        verify_number_is_positive(one_bit_extractor_seed_length)
        self._one_bit_extractor_seed_length = int(one_bit_extractor_seed_length)

        if basic_weak_design_type is not None:
            if self._weak_design_type != "block":
                warnings.warn(
                    'basic_weak_design passed but weak_design_type is not "block". Ignoring...',
                    UserWarning,
                )
                self._basic_weak_design_type = None
            verify_type(basic_weak_design_type, str)
            if basic_weak_design_type not in ["finite_field"]:
                raise ValueError(
                    f'basic_finite_field should be "finite_field", but {weak_design_type} was passed.'
                )
            self._basic_weak_design_type = basic_weak_design_type
        else:
            if self._weak_design_type == "block":
                warnings.warn(
                    'weak_design_type is "block" but basic_weak_design was not passed. Using "finite_field".',
                    UserWarning,
                )
                self._basic_weak_design_type = "finite_field"
            else:
                self._basic_weak_design_type = None

        # Since we only provide weak designs that require that the size of the sets
        # is a prime number, we enforce this here. This should be modified in the
        # future if other weak designs are added without such constraint.
        if is_prime(self._one_bit_extractor_seed_length):
            size_of_sets = self._one_bit_extractor_seed_length
        else:
            size_of_sets = next_prime(self._one_bit_extractor_seed_length)

        self._weak_design = WeakDesign.create(
            weak_design_type=self._weak_design_type,
            basic_weak_design_type=self._basic_weak_design_type,
            number_of_sets=self._output_length,
            size_of_sets=size_of_sets,
            precomputed_weak_design=precomputed_weak_design,
        )
        self._weak_design.compute_design()
        self._seed_length = self._weak_design.range_design

        # It is expected that we do not use optimal seed lengths for the one-bit extractor due to constraints
        # of the weak design. For example, the finite field weak design construction requires that the size of
        # the sets is a prime number, while the XOR one-bit extractor can take any seed lengths
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="seed_length is not a multiple of the bits required per index",
            )
            self._one_bit_extractor = RandomnessExtractor.create(
                extractor_type=self._one_bit_extractor_type,
                input_length=self._input_length,
                seed_length=self._one_bit_extractor_seed_length,
            )

    @staticmethod
    def _calculate_length_specific_construction(
        extractor_type: str,
        input_length: Integral,
        weak_design_type: str,
        one_bit_extractor_type: str,
        relative_source_entropy: Real,
        error_bound: Real,
    ) -> dict[str, int]:
        weak_design_overlap = WeakDesign.get_relative_overlap(weak_design_type)

        max_output_length = math.floor(
            input_length / weak_design_overlap * relative_source_entropy
        )

        # We start from the theoretical maximum bits we can extract based on the bound on the min-entropy and the
        # weak design overlap bound, and we decrease this output length until we find a valid one for a specific one-bit
        # extractor. Note that the error bound of the Trevisan's extractor has to be scaled with the output length, and
        # that there is an entropy loss due to the weak design if the overlap bound is not 1.
        output_length = max_output_length
        while output_length > 0:
            relative_source_entropy_one_bit_extractor = (
                relative_source_entropy
                - weak_design_overlap * output_length / input_length
            )
            error_bound_one_bit_extractor = error_bound / output_length

            try:
                one_bit_extractor_seed_length = RandomnessExtractor.subclasses[
                    one_bit_extractor_type
                ].calculate_length(
                    extractor_type=extractor_type,
                    input_length=input_length,
                    relative_source_entropy=relative_source_entropy_one_bit_extractor,
                    error_bound=error_bound_one_bit_extractor,
                )
                break
            except ValueError:
                output_length -= 1

        if output_length == 0:
            raise ValueError(
                "It was not possible to find the the output length and seed length for given parameters."
            )

        if not is_prime(one_bit_extractor_seed_length):
            one_bit_extractor_seed_length = next_prime(one_bit_extractor_seed_length)

        if weak_design_type == "finite_field":
            seed_length = one_bit_extractor_seed_length**2
        else:  # This should be generalized if any other basic weak design is implemented
            factor = math.ceil(
                (
                    math.log2(output_length - 2 * math.e)
                    - math.log2(one_bit_extractor_seed_length - 2 * math.e)
                )
                / (math.log2(2 * math.e) - math.log2(2 * math.e - 1))
            )
            seed_length = factor * one_bit_extractor_seed_length**2

        return {
            "output_length": output_length,
            "one_bit_extractor_seed_length": one_bit_extractor_seed_length,
            "seed_length": seed_length,
        }

    @staticmethod
    def calculate_length(
        extractor_type: str, input_length: Integral, **kwargs
    ) -> list[dict[str, int]]:
        verify_type(extractor_type, str)
        if extractor_type not in ["quantum", "classical"]:
            raise ValueError(
                f'extractor_type should be "quantum" or "classical", but {extractor_type} was passed.'
            )

        verify_number_type(input_length, Integral)
        verify_number_is_positive(input_length)

        verify_kwargs(
            kwargs,
            ["relative_source_entropy", "error_bound"],  # required
            ["weak_design_type", "one_bit_extractor_type"],  # optional
        )

        relative_source_entropy = kwargs.get("relative_source_entropy")
        verify_number_type(relative_source_entropy, Real)
        verify_number_in_interval(relative_source_entropy, 0, 1, "left-open")

        error_bound = kwargs.get("error_bound")
        verify_number_type(error_bound, Real)
        verify_number_in_interval(error_bound, 0, 1, "left-open")

        if "weak_design_type" not in kwargs:
            weak_designs = ["finite_field", "block"]
        else:
            weak_design_type = kwargs.get("weak_design_type")
            verify_type(weak_design_type, str)
            if weak_design_type not in ["finite_field", "block"]:
                raise ValueError(
                    f"{weak_design_type} is not a valid weak design. "
                    f"Run randextract.WeakDesign.subclasses.keys() to get the valid types."
                )
            weak_designs = [weak_design_type]

        if "one_bit_extractor_type" not in kwargs:
            one_bit_extractors = ["xor", "polynomial"]
        else:
            one_bit_extractor_type = kwargs.get("one_bit_extractor_type")
            verify_type(one_bit_extractor_type, str)
            if one_bit_extractor_type not in ["xor", "polynomial"]:
                raise ValueError(
                    f'one_bit_extractor_type can only be "xor" or "polynomial", but {one_bit_extractor_type} was passed.'
                )
            one_bit_extractors = [one_bit_extractor_type]

        solutions = []
        for ext, weak in itertools.product(one_bit_extractors, weak_designs):
            sol_dict = TrevisanExtractor._calculate_length_specific_construction(
                extractor_type=extractor_type,
                input_length=input_length,
                weak_design_type=weak,
                one_bit_extractor_type=ext,
                relative_source_entropy=relative_source_entropy,
                error_bound=error_bound,
            )
            solutions.append(
                {"weak_design_type": weak, "one_bit_extractor_type": ext, **sol_dict}
            )
        return solutions

    def extract(self, extractor_input: np.ndarray | GF2, seed: np.ndarray | GF2) -> GF2:
        r"""
        For a given input and a uniform seed, the chosen one-bit extractor is called ``output_length`` times using as
        seeds the bits determined by the weak design from the given seed.

        Arguments:
            extractor_input: Binary array from the weak random source.
            seed: Uniform seed used to call the one-bit extractor multiple times.

        Returns:
            GF2: An almost uniform (up to an error ``error_bound``) binary array.
        """
        output = GF2.Zeros(self._output_length)
        for i in range(self._output_length):
            seed_one_bit_extractor = seed[
                self._weak_design.get_set(i)[: self._one_bit_extractor.seed_length]
            ]
            output[i] = self._one_bit_extractor.extract(
                extractor_input=extractor_input,
                seed=seed_one_bit_extractor,
            )
        return output

    @property
    def input_length(self) -> int:
        return self._input_length

    @property
    def seed_length(self) -> int:
        return self._seed_length

    @property
    def output_length(self) -> int:
        return self._output_length

    @property
    def weak_design(self) -> WeakDesign:
        return self._weak_design

    @property
    def one_bit_extractor(self) -> RandomnessExtractor:
        return self._one_bit_extractor
