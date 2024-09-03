import math
import warnings
from numbers import Integral
from pathlib import Path

import numpy as np
from galois import GF, FieldArray, Poly, is_prime

from .._verify import (
    verify_number_in_interval,
    verify_number_is_positive,
    verify_number_type,
    verify_type,
)
from .weak_design import WeakDesign


@WeakDesign.register_subclass("finite_field")
class FiniteFieldPolynomialDesign(WeakDesign):
    r"""
    Implementation class for the finite field polynomial weak design.

    The weak design is obtained by evaluating polynomials over a finite field of a size matching the desired set size.
    Roughly speaking, as many polynomials as the desired number of sets are computed in a degree increasing order.
    Each polynomial labels one set in the family of sets. Then, for each element in the finite field, the corresponding
    polynomial is evaluated. Finally, the pair (element, polynomial evaluation) is mapped to a larger finite field of
    size matching the length of the extractor's seed. For the mathematical details and a detailed example check
    :ref:`the corresponding section <A basic construction>` in the theory.

    This particular weak design has an upper bound on its relative overlap of :math:`2e`.

    Arguments:
        number_of_sets: Size of the weak design, i.e., the number of sets.
        size_of_sets: The size (cardinality) of all sets. It must be a prime number.
        weak_design: A pre-computed weak design, i.e., a family of ``number_of_sets`` sets of size ``size_of_set``.
            It should be given a FieldArray (or np.ndarray), or a pathlib.Path to a file saved using `save_to_file()`.
        assume_valid: A boolean that triggers computing the overlap of the provided weak design to check that it's a
            valid family with an overlap below the upper bound. Use ``True`` to do the check, use ``False`` (default) to
            skip it.

    Examples:
        A :obj:`FiniteFieldPolynomialDesign` object can be created directly calling the constructor of this class.
        For example, the following code creates a weak design with 1024 sets, each one with 32771 values from the
        finite field :math:`[32771^2] = \{0,\dots,1073938440\}`.

        .. code-block:: python

            import randextract
            from randextract import FiniteFieldPolynomialDesign

            wd = FiniteFieldPolynomialDesign(
                    number_of_sets=2**10,
                    size_of_set=32_771)

            number_of_sets = wd.number_of_sets
            size_of_set = wd.size_of_set
            finite_field_size = wd.weak_design._order

        The following code compute the small design presented as an example in the
        :ref:`theory section <A basic construction>`.

        .. code-block:: python

            wd = FiniteFieldPolynomialDesign(
                    number_of_sets=6,
                    size_of_set=2)

            print(wd.weak_design)

    See Also:
        Theory: :doc:`theory/trevisan`.
    """

    def __init__(
        self,
        number_of_sets: Integral,
        size_of_sets: Integral,
        precomputed_weak_design: np.ndarray | None = None,
        assume_valid: bool | None = False,
    ):
        self._is_computed = False

        verify_number_type(number_of_sets, Integral)
        verify_number_type(size_of_sets, Integral)
        verify_number_is_positive(number_of_sets)

        self._number_of_sets = int(number_of_sets)
        self._size_of_sets = int(size_of_sets)

        if not is_prime(self._size_of_sets):
            raise ValueError("size_of_sets must be a prime number.")

        self._range = self._size_of_sets**2
        self._gf = GF(self._size_of_sets)
        self._GF = GF(self._range)

        self._polynomial_degree = max(
            math.ceil(
                math.log2(self._number_of_sets) / math.log2(self._size_of_sets) - 1
            ),
            0,
        )

        if precomputed_weak_design is not None:
            precomputed_weak_design = WeakDesign.read_from_file(
                precomputed_weak_design,
                self._number_of_sets,
                self._size_of_sets,
                self._range,
            )
            verify_type(assume_valid, bool)

            self._weak_design = precomputed_weak_design

            if not assume_valid:
                if not WeakDesign.is_valid(
                    precomputed_weak_design,
                    self._number_of_sets,
                    self._size_of_sets,
                    self.relative_overlap(),
                ):
                    raise ValueError(
                        "Precomputed weak design is not a valid design, overlap is too large."
                    )

            self._is_computed = True
        else:
            self._weak_design = None

    # TODO: Further refactoring. Everything can be done in a single function avoiding computing
    # many times the same things, e.g. evaluation_points are always the same for a given size_of_sets.
    # However, they are computed every single time. Also, I think all coefficients, polynomials, etc.
    # can be obtained in a single step, and then obtain the evaluation results with a matrix multiplication.

    def _compute_coefficients_polynomial(self, set_index: int) -> FieldArray:
        # TODO: refactor this in one line if galois extends power operation
        # See https://github.com/mhostetter/galois/issues/232
        number_of_coefficients = self._polynomial_degree + 1
        coefficients = set_index // (
            self._size_of_sets ** np.arange(number_of_coefficients)
        )
        return self._gf(coefficients % self._size_of_sets)

    def _compute_set(self, set_index: int) -> np.ndarray:
        coefficients = self._compute_coefficients_polynomial(set_index)
        polynomial = Poly(coefficients, self._gf, order="asc")
        # TODO: clarify this dance between galois field arrays and numpy arrays
        # It is actually to avoid manually choosing the best dtype
        evaluation_points = np.array(self._gf.Range(0, self._size_of_sets))
        # {P(x) : x in GF(size_of_sets)}
        evaluation_results = np.array(polynomial(evaluation_points))
        # In the mapping from [size_of_sets] x [size_of_sets] -> [range] we must change the GF
        _set = self._GF(evaluation_results) * self._GF(self._size_of_sets) + self._GF(
            evaluation_points
        )
        return _set.view(np.ndarray)

    @classmethod
    def relative_overlap(cls) -> float:
        return 2 * math.e

    def compute_design(self):
        if self._is_computed:
            warnings.warn("Weak design is already computed!")
            return

        weak_design = np.zeros((self._number_of_sets, self._size_of_sets), dtype=int)
        for set_index in range(self._number_of_sets):
            weak_design[set_index] = self._compute_set(set_index)
        self._is_computed = True
        self._weak_design = weak_design

    def get_set(self, index: Integral) -> np.ndarray | None:
        if not self._is_computed:
            warnings.warn("Design not computed yet! Use compute_design() method.")
            return

        verify_number_type(index, Integral)
        verify_number_in_interval(index, 0, self._number_of_sets, "right-open")

        return self._weak_design[index]

    @property
    def number_of_sets(self) -> int:
        return self._number_of_sets

    @property
    def size_of_sets(self) -> int:
        return self._size_of_sets

    @property
    def range_design(self) -> int:
        return self._range

    @property
    def weak_design(self) -> np.ndarray | None:
        if not self._is_computed:
            warnings.warn("Design not computed yet! Use compute_design() method.")
            return

        return self._weak_design

    @property
    def is_computed(self) -> bool:
        return self._is_computed
