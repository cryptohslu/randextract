import math
import warnings
from numbers import Integral

import numpy as np
from galois import is_prime

from .._verify import (
    verify_number_in_interval,
    verify_number_is_positive,
    verify_number_type,
    verify_type,
)
from .finite_field_polynomial_design import FiniteFieldPolynomialDesign
from .weak_design import WeakDesign


@WeakDesign.register_subclass("block")
class BlockDesign(WeakDesign):
    def __init__(
        self,
        number_of_sets: int,
        size_of_sets: int,
        precomputed_weak_design: np.ndarray | None = None,
        assume_valid: bool | None = False,
    ):
        self._is_computed = False

        # Modify this in the future if more basic weak designs are implemented
        self._basic_weak_design_type = "finite_field"

        verify_number_type(number_of_sets, Integral)
        verify_number_type(size_of_sets, Integral)

        self._basic_weak_design_overlap = WeakDesign.get_relative_overlap(
            self._basic_weak_design_type
        )

        verify_number_in_interval(
            number_of_sets,
            self._basic_weak_design_overlap,
            math.inf,
            "right-open",
        )

        verify_number_in_interval(
            size_of_sets,
            self._basic_weak_design_overlap,
            math.inf,
            "right-open",
        )

        self._number_of_sets = int(number_of_sets)
        self._size_of_sets = int(size_of_sets)

        if not is_prime(self._size_of_sets):
            raise ValueError("size_of_sets must be a prime number.")

        self._number_basic_weak_designs = self._compute_number_basic_weak_designs()
        self._range = self._number_basic_weak_designs * self._size_of_sets**2

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

        self._weak_design = None

    def _compute_number_basic_weak_designs(self):
        return 1 + max(
            1,
            math.ceil(
                (
                    math.log2(self._number_of_sets - self._basic_weak_design_overlap)
                    - math.log2(self._size_of_sets - self._basic_weak_design_overlap)
                )
                / (
                    math.log2(self._basic_weak_design_overlap)
                    - math.log2(self._basic_weak_design_overlap - 1)
                )
            ),
        )

    def compute_design(self):
        if self._is_computed:
            warnings.warn("Weak design is already computed!")
            return

        weak_design = np.zeros((self._number_of_sets, self._size_of_sets), dtype=int)
        basic_weak_designs = {}

        # Eq.4 arXiv 1212.0520
        n = np.power(
            1 - 1 / self._basic_weak_design_overlap,
            np.arange(self._number_basic_weak_designs - 1),
        ) * (self._number_of_sets / self._basic_weak_design_overlap - 1)
        m = np.zeros(self._number_basic_weak_designs, dtype=int)

        for i in range(self._number_basic_weak_designs - 1):
            m[i] = np.ceil(n[: i + 1].sum()) - m[:i].sum()
        m[-1] = self.number_of_sets - m.sum()

        for i in range(self._number_basic_weak_designs):
            basic_week_design_sets = m[i]
            if basic_week_design_sets not in basic_weak_designs.keys():
                basic_weak_designs[basic_week_design_sets] = WeakDesign.create(
                    weak_design_type=self._basic_weak_design_type,
                    number_of_sets=basic_week_design_sets,
                    size_of_sets=self._size_of_sets,
                )
                basic_weak_designs[basic_week_design_sets].compute_design()
            weak_design[m[:i].sum() : m[:i].sum() + basic_week_design_sets, :] = (
                basic_weak_designs[basic_week_design_sets].weak_design
                + i * self._size_of_sets**2
            )

        self._is_computed = True
        self._weak_design = weak_design

    def get_set(self, index: Integral) -> np.ndarray | None:
        if not self._is_computed:
            warnings.warn("Design not computed yet! Use compute_design() method.")
            return

        verify_number_type(index, Integral)
        verify_number_in_interval(index, 0, self._number_of_sets, "right-open")

        return self._weak_design[index]

    @classmethod
    def relative_overlap(cls) -> float:
        return 1.0

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
