import math
import warnings
from numbers import Integral, Real

import numpy as np
import scipy as sp
from galois import GF2

from ._verify import (
    verify_array,
    verify_kwargs,
    verify_number_in_interval,
    verify_number_is_positive,
    verify_number_type,
    verify_type,
)
from .randomness_extractor import RandomnessExtractor


@RandomnessExtractor.register_subclass("toeplitz")
class ToeplitzHashing(RandomnessExtractor):
    r"""
    Implementation class for the Toeplitz hashing randomness extractor.

    The constructor expects the input and output lengths. These uniquely define a particular family of Toeplitz hashing
    functions. In order to compute the optimal output length for a given input length and some assumptions, check the
    static method :obj:`calculate_output_length()`.

    The output of this randomness extractor is the result of doing a matrix-vector multiplication, where the matrix is a
    Toeplitz matrix determined by the uniform seed, and the vector is the bit string from the weak random source.

    The :obj:`extract()` method computes the matrix-vector multiplication using the Fast Fourier Transform by
    embedding the Toeplitz matrix into a square circulant matrix. The Toeplitz matrix for a given seed can be obtained
    with the :obj:`to_matrix()` method (only for debug purposes).

    Arguments:
        input_length: Length of the bit string from the weak random source. It must be a positive number.
        output_length: Length of the randomness extractor's output. It must be in the range [1, ``input_length``].
            To compute the optimal output length given some constraints on the tolerated error and the weak random
            source use :obj:`calculate_output_length()`.

    Examples:
        The following creates a randomness extractor corresponding to a family of Toeplitz hashing functions that takes
        as input one million bits and extracts hundred thousand bits. The constructor does not select a particular
        function, this is done when passing a seed to the :obj:`extract()` method.

        .. code-block:: python

            from randextract import ToeplitzHashing

            ext = ToeplitzHashing(input_length=10**6, output_length=10**5)

        The :obj:`extract()` method can be used to extract the randomness from the input, providing a uniform seed of
        the right length. The following example creates a random input and seed and pass them to the extract method.

        .. code-block:: python

            import numpy as np
            from galois import GF2

            input_array = GF2.Random(ext.input_length)
            seed = GF2.Random(ext.seed_length)
            output_ext = ext.extract(input_array, seed)

            assert output_ext.size == ext.output_length

        The actual Toeplitz matrix can be exported using the :obj:`to_matrix()` method.

        .. code-block:: python

            ext = ToeplitzHashing(input_length=20, output_length=8)

            input_array = GF2.Random(ext.input_length)
            seed = GF2.Random(ext.seed_length)

            toeplitz_matrix = ext.to_matrix(seed)
            print(toeplitz_matrix)

            output1 = toeplitz_matrix @ input_array
            output2 = ext.extract(input_array, seed)
            assert np.array_equal(output1, output2)

        The benefits of using the :obj:`extract()` method are very noticeable for large Toeplitz matrices.

        .. code-block:: ipython

            ext = ToeplitzHashing(input_length=2**10, output_length=2**8)

            input_array = GF2.Random(ext.input_length)
            seed = GF2.Random(ext.seed_length)

            %timeit (ext.extract(input_array, seed))
            %timeit (ext.to_matrix(seed) @ input_array)

        By default, the Toeplitz matrix is constructed from the seed vector using the standard mathematical definition,
        i.e., :math:`M[i, j] = \text{seed}[i - j]`. However, it is possible to use a different order using the
        ``seed_mode`` argument. It can take values "default", "col-first", "row-first", or "custom". If "custom" is
        used, you must provide a permutation array and pass it to ``seed_order`` kwarg.

        .. code-block:: ipython

            ext = ToeplitzHashing(input_length=5, output_length=4)

            seed = GF2.Zeros(ext.seed_length)
            seed[0] = 1

            # Default order, first element is the first element of the matrix
            ext.to_matrix(seed)

            # Fill the first column first bottom to top, then first row left to right
            ext.to_matrix(seed, seed_mode="col-first")

            # Fill the first row first left to right, then the first column top to bottom
            ext.to_matrix(seed, seed_mode="row-first")

            # Custom order
            seed_order = np.array([1, 0, 2, 3, 4, 5, 6, 7])
            ext.to_matrix(seed, seed_mode="custom", seed_order=seed_order)

    See Also:
        Theory: :doc:`theory/toeplitz-hashing`.
    """

    def __init__(
        self,
        input_length: Integral,
        output_length: Integral,
    ):
        verify_number_type(input_length, Integral)
        verify_number_is_positive(input_length)
        self._input_length = int(input_length)

        verify_number_type(output_length, Integral)
        verify_number_in_interval(output_length, 1, self._input_length, "closed")
        self._output_length = int(output_length)

        self._number_of_rows = self._output_length
        self._number_of_columns = self._input_length
        self._seed_length = self._number_of_columns + self._number_of_rows - 1

    def _fast_multiplication_with_vector(self, vector: GF2, seed: GF2) -> GF2:
        # pad the input vector with 0s so that it has the total length of the input_seed.
        padded_vector = np.append(vector, GF2.Zeros(seed.size - vector.size))

        output = (
            np.round(
                sp.fft.irfft(
                    sp.fft.rfft(seed) * sp.fft.rfft(padded_vector), padded_vector.size
                )
            ).astype(np.uint8)
            % 2
        )

        # truncate the output vector to the number of rows of the Toeplitz matrix as the rest is not needed.
        return GF2(output[: self._number_of_rows])

    def _reorder_seed(
        self, seed: np.ndarray | GF2, seed_mode: str, seed_order: np.ndarray | None
    ):
        if seed_mode == "default":
            return seed

        if seed_mode == "col-first":
            seed_order = np.concatenate(
                [
                    np.arange(self._number_of_rows - 1, -1, -1),
                    np.arange(
                        self._number_of_rows + self._number_of_columns - 2,
                        self._number_of_rows - 1,
                        -1,
                    ),
                ]
            )
        elif seed_mode == "row-first":
            seed_order = np.concatenate(
                [
                    np.arange(self._number_of_columns - 1, seed.size),
                    np.arange(seed.size - self._number_of_rows),
                ]
            )
        elif seed_mode == "custom":
            assert isinstance(seed_order, np.ndarray)
            assert np.array_equal(np.sort(seed_order), np.arange(seed.size))

            col_first_order = np.concatenate(
                [
                    np.arange(self._number_of_rows - 1, -1, -1),
                    np.arange(
                        self._number_of_rows + self._number_of_columns - 2,
                        self._number_of_rows - 1,
                        -1,
                    ),
                ]
            )
            # Default order would be harder to give a custom permutation, so we first rearrange the seed
            # to be in the "col-first" mode. And later we rearrange with the custom permutation.
            return seed[seed_order][col_first_order]

        return seed[seed_order]

    @staticmethod
    def calculate_length(extractor_type: str, input_length: Integral, **kwargs) -> int:
        r"""
        For a given extractor type (i.e., quantum-proof) and a set of parameters, it computes the optimal output length
        for the (modified) Toeplitz hashing. The returned value can be used to choose the optimal family of extractors.

        Arguments:
            extractor_type: This determines the type of side information against which the extractor should remain
                secure. Two values are accepted: "quantum" and "classical". In this case, there is no difference between
                passing "quantum" or "classical" since the version of the leftover hash lemma with quantum side
                information has exactly the same form as the one with only classical side information.
            input_length: The length of the bit string from the weak random source.

        Keyword Arguments:
            relative_source_entropy: Lower bound on the conditional min-entropy of the weak random source, normalized by
                the input length. It must be a real number in the range (0, 1].
            error_bound: Upper bound on the randomness extractor's error, i.e., a measure of how far the output of the
                extractor is from the ideal (uniform) output. It must be a real number in the range (0, 1].
        """
        verify_type(extractor_type, str)
        if extractor_type not in ["quantum", "classical"]:
            raise ValueError(
                f'extractor_type should be "quantum" or "classical", but {extractor_type} was passed.'
            )

        # The quantum version of the leftover hash lemma recovers the same formula as in the case of the version with
        # classical side information (see e.g., Eq. (11) from arXiv:1002.2436). This means that the code does not change
        # for extractor_type="classical" or extractor_type="quantum". However, the estimation of the min-entropy would
        # probably be different, leading to a different (shorter) output length overall.

        verify_number_type(input_length, Integral)
        verify_number_is_positive(input_length)

        verify_kwargs(kwargs, ["relative_source_entropy", "error_bound"])

        relative_source_entropy = kwargs.get("relative_source_entropy")
        verify_number_type(relative_source_entropy, Real)
        verify_number_in_interval(relative_source_entropy, 0, 1, "left-open")

        error_bound = kwargs.get("error_bound")
        verify_number_type(error_bound, Real)
        verify_number_in_interval(error_bound, 0, 1, "left-open")

        # arXiv:1002.2436 Eq.(1)
        output_length = math.floor(
            relative_source_entropy * input_length - 2 * math.log2(1 / error_bound) + 2
        )

        # Extractors cannot extract more uniform bits than available in the weak random source
        max_output_length = math.floor(input_length * relative_source_entropy)
        if output_length > max_output_length:
            return max_output_length
        elif output_length < 0:
            return 0
        return output_length

    def extract(
        self,
        extractor_input: np.ndarray | GF2,
        seed: np.ndarray | GF2,
        seed_mode: str = "default",
        seed_order: np.ndarray | None = None,
    ) -> GF2:
        r"""
        Given ``input_length`` bits from a weak random source with at least ``relative_source_entropy`` :math:`\times`
        ``input_length`` bits of entropy, it outputs an almost uniform binary array up to an error ``error_bound``.

        Arguments:
            extractor_input: Binary array from the weak random source.
            seed: Uniform seed used to populate the Toeplitz matrix.
            seed_mode: Mode to construct the Toeplitz matrix from the seed vector. Default mode uses the standard
                mathematical definition, i.e., Toeplitz_matrix(i, j) = seed(i - j). Other possible modes are:
                "col-first", "row-first", or "custom". With "custom" a permutation array can be passed to seed_order.
            seed_order: Permutation vector that determines how to construct the Toeplitz matrix from the seed vector. A
                valid seed_order array is a permutation of ``np.arange(seed.size)``. Using ``np.arange(seed.size)`` as
                the permutation is equivalent to seed_mode="col-first". Read the docs for examples of how to use this
                parameter, e.g. :ref:`this unit test <TestToeplitzMatrixAlmostSquare>`.

        Returns:
            GF2: An almost uniform (up to an error ``error_bound``) binary array.
        """
        verify_type(extractor_input, np.ndarray)
        verify_type(seed, np.ndarray)
        verify_array(extractor_input, valid_dim=1, valid_size=self.input_length)
        verify_array(seed, valid_dim=1, valid_size=self.seed_length)
        extractor_input = GF2(extractor_input)
        seed = GF2(seed)

        verify_type(seed_mode, str)
        if seed_mode not in ["default", "col-first", "row-first", "custom"]:
            raise ValueError(
                f'seed_mode must be "default", "col-first", "row-first" or "custom", not {seed_mode}.'
            )

        if seed_mode == "custom" and seed_order is None:
            raise ValueError(f'seed_order must be given when seed_mode="custom".')

        if seed_order is not None:
            verify_array(
                seed_order,
                valid_dim=1,
                valid_size=self.seed_length,
                valid_range=[0, self.seed_length, "right-open"],
            )
            if not np.array_equal(np.sort(seed_order), np.arange(seed.size)):
                raise ValueError(
                    "A valid seed_order must be a permutation of np.arange(seed.size)"
                )
            if seed_mode != "custom":
                warnings.warn(
                    f'seed_order was given but will be ignored because seed_mode is not "custom".',
                    UserWarning,
                )

        seed = self._reorder_seed(seed, seed_mode, seed_order)

        output = self._fast_multiplication_with_vector(
            vector=extractor_input[: self._number_of_columns], seed=seed
        )
        return output

    def to_matrix(
        self,
        seed: np.ndarray | GF2,
        seed_mode: str = "default",
        seed_order: np.ndarray | None = None,
    ) -> GF2:
        r"""
        For a given seed, it outputs the corresponding Toeplitz matrix with dimensions ``output_length`` :math:`\times`
        ``input_length``.

        Arguments:
            seed: Uniform seed used to populate the Toeplitz matrix.
            seed_mode: Mode to construct the Toeplitz matrix from the seed vector. Default mode uses the standard
                mathematical definition, i.e., Toeplitz_matrix(i, j) = seed(i - j). Other possible modes are:
                "col-first", "row-first", or "custom". With "custom" a permutation array can be passed to seed_order.
            seed_order: Permutation vector that determines how to construct the Toeplitz matrix from the seed vector. A
                valid seed_order array is a permutation of ``np.arange(seed.size)``. Using ``np.arange(seed.size)`` as
                the permutation is equivalent to seed_mode="col-first". Read the docs for examples of how to use this
                parameter, e.g. :ref:`this unit test <TestToeplitzMatrixAlmostSquare>`.

        Returns:
            GF2: The corresponding Toeplitz matrix, a 2-dimensional binary array of shape
            (``output_length``, ``input_length``).
        """
        verify_type(seed, np.ndarray)
        verify_array(seed, valid_dim=1, valid_size=self.seed_length)
        seed = GF2(seed)

        verify_type(seed_mode, str)
        if seed_mode not in ["default", "col-first", "row-first", "custom"]:
            raise ValueError(
                f'seed_mode must be "default", "col-first", "row-first" or "custom", not {seed_mode}.'
            )

        if seed_mode == "custom" and seed_order is None:
            raise ValueError(f'seed_order must be given when seed_mode="custom".')

        if seed_order is not None:
            verify_array(
                seed_order,
                valid_dim=1,
                valid_size=self.seed_length,
                valid_range=[0, self.seed_length, "right-open"],
            )
            if not np.array_equal(np.sort(seed_order), np.arange(seed.size)):
                raise ValueError(
                    "A valid seed_order must be a permutation of np.arange(seed.size)"
                )
            if seed_mode != "custom":
                warnings.warn(
                    f'seed_order was given but will be ignored because seed_mode is not "custom".',
                    UserWarning,
                )

        seed = self._reorder_seed(seed, seed_mode, seed_order)

        matrix = GF2.Zeros((self._number_of_rows, self._number_of_columns))
        for i in range(self._number_of_rows):
            for j in range(self._number_of_columns):
                matrix[i, j] = seed[i - j]

        return matrix

    @property
    def input_length(self) -> int:
        return self._input_length

    @property
    def seed_length(self) -> int:
        return self._seed_length

    @property
    def output_length(self) -> int:
        return self._output_length


@RandomnessExtractor.register_subclass("modified_toeplitz")
class ModifiedToeplitzHashing(ToeplitzHashing):
    r"""
    Implementation class for the modified Toeplitz hashing randomness extractor.

    The constructor expects the input and output lengths. These uniquely define a particular family of modified Toeplitz
    hashing functions. In order to compute the optimal output length for a given input length and some assumptions,
    check the static method :obj:`calculate_output_length()`.

    With this information, the output of the randomness extractor is the result of doing a matrix-vector multiplication.
    The matrix is the result of concatenating a Toeplitz matrix, determined by the uniform seed, together with an
    identity matrix of size ``output_length``. The vector is the bit string from the weak random source.

    Because the Toeplitz matrix has dimensions ``output_length`` :math:`\times` ``input_length - output_length``, the
    required seed is smaller than with the Toeplitz hashing.

    The :obj:`extract()` method computes the matrix-vector multiplication using the Fast Fourier Transform by
    embedding the modified Toeplitz matrix into a square circulant matrix. The modified Toeplitz matrix for a given seed
    can be obtained with the :obj:`to_matrix()` method.

    Arguments:
        input_length: Length of the bit string from the weak random source. It must be a positive number.
        output_length: Length of the randomness extractor's output. It must be in the range [1, ``input_length``].
            To compute the optimal output length given some constraints on the tolerated error and the weak random
            source use :obj:`calculate_output_length()`.

    Examples:
        The following creates a randomness extractor corresponding to a family of modified Toeplitz hashing functions
        that takes as input one million bits and extracts hundred thousand bits. The constructor does not select a
        particular function, this is done when passing a seed to the :obj:`extract()` method.

        .. code-block:: python

            from randextract import ModifiedToeplitzHashing

            ext = ModifiedToeplitzHashing(input_length=10**6, output_length=10**5)

        Notice that the required seed is smaller because it does not depend on the output length.

        .. code-block:: python

            from randextract import ToeplitzHashing

            toeplitz = ToeplitzHashing(input_length=10**6, output_length=10**5)

            mod_toeplitz = ModifiedToeplitzHashing(input_length=10**6, output_length=10**5)

            print(f"Toeplitz hashing requires a seed of length {toeplitz.seed_length:_}.")
            print(f"While the modified Toeplitz hashing a seed of length {mod_toeplitz.seed_length:_}.")

        You can check that the new matrix is a concatenation of a Toeplitz matrix with an identity matrix using the
        :obj:`to_matrix()` method.

        .. code-block:: python

            import numpy as np
            from galois import GF2

            ext = ModifiedToeplitzHashing(input_length=20, output_length=8)

            input_array = GF2.Random(ext.input_length)
            seed = GF2.Random(ext.seed_length)

            modified_toeplitz_matrix = ext.to_matrix(seed)
            print(modified_toeplitz_matrix)

            output1 = modified_toeplitz_matrix @ input_array
            output2 = ext.extract(input_array, seed)
            assert np.array_equal(output1, output2)

        This saving in required seed also comes with some performance gain.

        .. code-block:: python

            ext1  = ToeplitzHashing(input_length=2**20, output_length=2**10)
            ext2  = ModifiedToeplitzHashing(input_length=2**20, output_length=2**10)

            input_array = GF2.Random(ext1.input_length)
            seed1 = GF2.Random(ext1.seed_length)
            seed2 = GF2.Random(ext2.seed_length)

            %timeit (ext1.extract(input_array, seed1))
            %timeit (ext2.extract(input_array, seed2))

    See Also:
        Theory: :doc:`theory/toeplitz-hashing`.
    """

    def __init__(
        self,
        input_length: Integral,
        output_length: Integral,
    ):
        verify_number_type(input_length, Integral)
        verify_number_is_positive(input_length)
        self._input_length = int(input_length)

        verify_number_type(output_length, Integral)
        verify_number_in_interval(output_length, 1, self._input_length, "closed")
        self._output_length = int(output_length)

        self._number_of_rows = self._output_length
        self._number_of_columns = self._input_length - self._output_length
        if self._number_of_columns == 0:
            self._seed_length = 0
        else:
            self._seed_length = self._number_of_columns + self._number_of_rows - 1

    @staticmethod
    def calculate_length(extractor_type: str, input_length: Integral, **kwargs) -> int:
        return ToeplitzHashing.calculate_length(extractor_type, input_length, **kwargs)

    def to_matrix(
        self,
        seed: np.ndarray | GF2 | None,
        seed_mode: str = "default",
        seed_order: np.ndarray | None = None,
    ) -> GF2:
        # Special case when input_length = output_length
        if self.seed_length == 0:
            if seed is not None:
                warnings.warn(
                    "No seed required, but seed was passed. Ignoring it.", UserWarning
                )
            return GF2.Identity(self.output_length)

        return np.concatenate(
            (
                super().to_matrix(seed, seed_mode, seed_order),
                GF2.Identity(self._number_of_rows),
            ),
            axis=1,
        )

    def extract(
        self,
        extractor_input: np.ndarray | GF2,
        seed: np.ndarray | GF2 | None,
        seed_mode: str = "default",
        seed_order: np.ndarray | None = None,
    ) -> GF2:
        # Special case when input_length = output_length
        if self.seed_length == 0:
            if seed is not None:
                warnings.warn(
                    "No seed required, but seed was passed. Ignoring it.", UserWarning
                )
            return extractor_input

        return super().extract(extractor_input, seed, seed_mode, seed_order) + GF2(
            extractor_input[self._number_of_columns :]
        )
