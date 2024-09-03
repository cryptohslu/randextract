import math
import warnings
from numbers import Integral, Real

import numpy as np
from galois import GF2
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
from .utilities.binary_entropy import binary_entropy, inverse_binary_entropy
from .utilities.converter import binary_array_to_integer


@RandomnessExtractor.register_subclass("xor")
class XOROneBitExtractor(RandomnessExtractor):
    r"""
    Implementation class for the XOR one-bit randomness extractor.

    This one-bit extractor works by computing the parity of a selection of bits from the input. The selection of bits is
    done using a uniform seed. Parity of a binary array refers to whether it contains an odd or even number of 1-bits.
    The array has "odd parity" (1), if it contains odd number of 1s and has "even parity" (0) if it contains even number
    1s. For a more rigorous definition and a detailed example check :ref:`the corresponding section <XOR one-bit
    extractor>` in the theory.

    Arguments:
        input_length: Length of the bit string from the weak random source. It must be greater than 1 and smaller than
            the max integer that can be stored in a np.uint64 (approx. :math:`10^{19}`)
        number_indices: Number of bits from the input that will be XOR together.
            It must be an integer in [1, ``input_length``].
        seed_length: Length of the uniform seed. It must be an integer in [1, ``input_length`` * ``bits_per_index``]

    Examples:
        A :obj:`XOROneBitExtractor` object can be created directly calling the constructor of this class or using
        the factory class :obj:`RandomnessExtractor`. Both methods are equivalent.

        .. code-block:: python

            from randextract import RandomnessExtractor, XOROneBitExtractor

            ext1 = RandomnessExtractor.create(
                extractor_type="xor", input_length=2**20, number_indices=100
            )

            ext2 = XOROneBitExtractor(input_length=2**20, number_indices=100)

            assert ext1.output_length == ext2.output_length == 1
            assert ext1.number_indices == ext2.number_indices == 100
            # 20 bits are needed to store an index in [0, 2^20 - 1]
            assert ext1.seed_length == ext2.seed_length == 20 * 100

        Even though the output is a single bit, to be consistent with the RandomnessExtractor this extractor still
        outputs a 1-dim array. If you want the bit value use the item() function.

        .. code-block:: python

            import numpy as np
            from galois import GF2
            from randextract import XOROneBitExtractor

            ext = XOROneBitExtractor(input_length=100, number_indices=10)

            rng = np.random.default_rng()
            extractor_input = GF2.Random(100, seed=rng)
            seed = rng.integers(low=0, high=100, size=10)

            output = ext.extract(extractor_input, seed).item()

        How many indices or, in other words, how long should the seed be in order to extract securely one bit from a
        weak randomness source can be computed using the class method ``calculate_length()``.

        .. code-block:: python

            from randextract import XOROneBitExtractor

            indices = XOROneBitExtractor.calculate_length(
                extractor_type="quantum",
                return_mode="indices",
                input_length=2**20,
                relative_source_entropy=0.7,
                error_bound=1e-6
            )

            ext = XOROneBitExtractor(2**20, number_indices=indices)
            print(f"input length: {ext.input_length}")
            print(f"seed length: {ext.seed_length}")
            print(f"number indices: {ext.number_indices}")

    Raises:
        TypeError: If the passed arguments have wrong types, or if a compulsory argument is missing
            (e.g. missing ``input_length``).
        ValueError: If the passed arguments have wrong values (e.g. a negative ``seed_length`` value).

    See Also:
        Theory: :ref:`XOR one-bit extractor`
    """

    def __init__(
        self,
        input_length: Integral,
        number_indices: Integral | None = None,
        seed_length: Integral | None = None,
    ):
        verify_number_type(input_length, Integral)
        verify_number_is_positive(input_length)
        self._input_length = int(input_length)
        if self._input_length > np.iinfo(np.uint64).max:
            raise ValueError(
                "Because of how the XOR one-bit extractor is implemented, the max supported input length is ~10^19."
            )

        if self._input_length == 1:
            self._bits_per_index = 1
            if number_indices is not None or seed_length is not None:
                warnings.warn(
                    "No need to provide any seed when input_length is 1.", UserWarning
                )
            self._number_indices = 0
            self._seed_length = 0
            return
        else:
            self._bits_per_index = math.ceil(math.log2(self._input_length))

        if number_indices is None and seed_length is None:
            raise TypeError("You must provide either number_indices or seed_length.")

        if number_indices is not None:
            verify_type(number_indices, Integral)
            verify_number_in_interval(number_indices, 0, self._input_length, "closed")
            self._number_indices = int(number_indices)
            self._seed_length = self._number_indices * self._bits_per_index
            if seed_length is not None:
                warnings.warn(
                    "You have provided both seed_length and number_indices. Using number_indices.",
                    UserWarning,
                )
            return

        verify_type(seed_length, Integral)
        verify_number_in_interval(
            seed_length, 0, self._bits_per_index * self._input_length, "closed"
        )
        if seed_length % self._bits_per_index != 0:
            warnings.warn(
                f"seed_length is not a multiple of the bits required per index, {self._bits_per_index} in this case. "
                f"{seed_length % self._bits_per_index} bits from the seed will be ignored",
                UserWarning,
            )
        self._number_indices = math.floor(seed_length / self._bits_per_index)
        self._seed_length = self._number_indices * self._bits_per_index

    def _convert_seed(self, seed: np.ndarray | GF2) -> np.ndarray:
        """
        Converts a binary seed into an array of indices with dtype np.uint64.
        """

        if seed.size == 0:
            return np.array([], dtype=np.uint64)

        if self._input_length == 1:
            return np.array([0], dtype=np.uint64)

        indices_seed = np.empty(self._number_indices, dtype=np.uint64)
        seed = GF2(
            seed[: self._bits_per_index * self._number_indices].reshape(
                [self._number_indices, self._bits_per_index]
            )
        )

        for i in range(self._number_indices):
            indices_seed[i] = binary_array_to_integer(seed[i, :]) % self._input_length

        return indices_seed

    @staticmethod
    def calculate_length(extractor_type: str, input_length: Integral, **kwargs) -> int:
        r"""
        For a given extractor type (i.e., quantum-proof) and a set of parameters, it computes the optimal seed length
        for the XOR one-bit extractor. The returned value is, by default, in order to be consistent with other
        extractors the number of required bits. Alternatively, the number of indices can be obtained by passing
        ``return_mode="binary"``. Both values can be passed directly to the :obj:`XOROneBitExtractor` constructor to
        select the optimal family of functions.

        Arguments:
            extractor_type: This determines the type of side information against which the extractor should remain
                secure. Two values are accepted: "quantum" and "classical".
            input_length: The length of the bit string from the weak random source.

        Keyword Arguments:
            relative_source_entropy: Lower bound on the conditional min-entropy of the weak random source, normalized by
                the input length. It must be a real number in the range (0, 1].
            error_bound: Upper bound on the randomness extractor's error, i.e., a measure of how far the output of the
                extractor is from the ideal (uniform) output. It must be a real number in the range (0, 1].
            return_mode (optional): Either set with value "indices" to return the optimal number of indices or with
                "binary" (default) to return the optimal number of bits.

        Raises:
            TypeError: If some of the passed arguments or keyword arguments have wrong types.
            ValueError: If some of the passed arguments or keyword arguments have wrong values, if it is not possible to
                find a seed length for given parameters, or if the numerical optimization failed.

        Returns:
            Either the number of indices or the seed length (default) depending on ``return_mode`` required to extract
            securely one bit from the weak randomness source.
        """

        verify_type(extractor_type, str)
        if extractor_type not in ["quantum", "classical"]:
            raise ValueError(
                f'extractor_type should be "quantum" or "classical", but {extractor_type} was passed.'
            )

        verify_number_type(input_length, Integral)
        verify_number_is_positive(input_length)

        verify_kwargs(
            kwargs, ["relative_source_entropy", "error_bound"], ["return_mode"]
        )

        relative_source_entropy = kwargs.get("relative_source_entropy")
        verify_number_type(relative_source_entropy, Real)
        verify_number_in_interval(relative_source_entropy, 0, 1, "left-open")

        error_bound = kwargs.get("error_bound")
        verify_number_type(error_bound, Real)
        verify_number_in_interval(error_bound, 0, 1, "left-open")

        return_mode = kwargs.get("return_mode")
        if return_mode is None:
            return_mode = "binary"
        verify_type(return_mode, str)
        if return_mode not in ["indices", "binary"]:
            raise ValueError(
                f'return_mode should be either "indices" or "binary", but {return_mode} was passed.'
            )

        source_entropy = relative_source_entropy * input_length

        # Edge case uniform input
        if input_length == source_entropy:
            warnings.warn(
                "Input is already uniform, no need to use an extractor.",
                UserWarning,
            )
            return 0

        if extractor_type == "quantum":
            # Lemma B.5 arXiv 1212.0520
            error_bound = (error_bound / (1 + math.sqrt(2))) ** 2

        # It is not always possible to extract a bit with the desired error bound.
        # This checks that we are in a valid region.
        if (
            relative_source_entropy
            < 3 * math.log2(1 / error_bound) / input_length
            + math.log2(4 / 3) / input_length
        ):
            raise ValueError(
                "It is not possible to find valid number of indices for this input length and error bound. Either "
                "increase the input length or allow for higher errors."
            )

        # First paragraph p. 8 arXiv 1212.0520
        # We find g for which the min-entropy requirement holds
        def _find_root(g, n, k, err):
            return k - g - 3 * np.log2(1 / err) / n - np.log2(4 / 3) / n

        gamma = root_scalar(
            _find_root,
            x0=0,
            args=(input_length, relative_source_entropy, error_bound),
            bracket=[0, 1],
        )

        if not gamma.converged:
            raise ValueError(
                "Numerical optimization to find the optimal number of indices failed. "
                "This should not have happened. Please open an issue."
            )

        h_arg = inverse_binary_entropy(gamma.root)
        number_indices = math.ceil(math.log(2) / h_arg * math.log2(2 / error_bound))

        if number_indices > input_length:
            raise ValueError(
                "It is not possible to find valid number of indices for this input length and error bound. Either "
                "increase the input length or allow for higher errors."
            )

        if return_mode == "indices":
            return number_indices
        else:
            return number_indices * math.ceil(math.log2(input_length))

    def extract(
        self, extractor_input: np.ndarray | GF2, seed: np.ndarray | GF2 | None, **kwargs
    ) -> GF2:
        r"""
        Given a bitstring of length ``input_length``, the XOR one-bit extractor returns the parity of the bits selected
        by the ``seed``. This seed can take two different forms: an array of indices or another bitstring. The mode can
        be selected using the keyword argument ``seed_mode``.

        Arguments:
            extractor_input: Binary array from the weak random source.
            seed: An array of indices with values in [0, ``ìnput_length`` - 1] or a binary array.

        Keyword Arguments:
            seed_mode (optional): It can take two values: ``"indices"`` or ``"binary"``. The former assumes that the
                seed is an array with values that can be used to select any of the bits from the ``extractor_input``.
                The latter assumes that each :math:`\lceil\log_2(\text{input\_length})\rceil` bits represent a single
                index.

        Returns:
            The extracted bit returned as a A 1-dim GF2 array of size 1.
        """
        verify_array(extractor_input, 1, self._input_length, Integral, [0, 1, "closed"])
        extractor_input = GF2(extractor_input)

        verify_kwargs(kwargs, [], ["seed_mode"])

        if "seed_mode" not in kwargs:
            seed_mode = "binary"
        else:
            seed_mode = kwargs.get("seed_mode")
            verify_type(seed_mode, str)
            if seed_mode not in ["binary", "indices"]:
                raise ValueError("seed_mode must be 'binary' or 'indices'.")

        # Special case when input is uniform: We return first bit.
        if self.number_indices == 0:
            if seed is not None:
                warnings.warn(
                    "No seed required, but seed was passed. Ignoring it.", UserWarning
                )
            return extractor_input[0]

        if seed_mode == "binary":
            verify_array(seed, 1, self._seed_length, Integral, [0, 1, "closed"])
            seed = self._convert_seed(seed)
        else:
            verify_array(
                seed,
                1,
                self._number_indices,
                Integral,
                [0, self._input_length, "right-open"],
            )

        # Return a 1-dim GF2 array (of size 1) as any other extractor, not a single int.
        return extractor_input[seed].sum()

    @property
    def input_length(self) -> int:
        return self._input_length

    @property
    def seed_length(self) -> int:
        return self._seed_length

    @property
    def number_indices(self) -> int:
        return self._number_indices

    @property
    def output_length(self) -> int:
        return 1
