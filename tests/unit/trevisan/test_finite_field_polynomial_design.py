import unittest

import galois
import numpy as np

from randextract.trevisan.finite_field_polynomial_design import (
    FiniteFieldPolynomialDesign,
)
from randextract.trevisan.weak_design import WeakDesign

# TODO:
# - Test internal functions
# - Test compute_design()
# - Test is_computed


class TestInitialization(unittest.TestCase):
    def test_different_integral_types(self):
        with self.subTest(msg="int"):
            FiniteFieldPolynomialDesign(number_of_sets=10, size_of_sets=3)
        with self.subTest(msg="np.uint8"):
            FiniteFieldPolynomialDesign(np.uint8(10), np.uint8(3))
        with self.subTest(msg="np.int64"):
            FiniteFieldPolynomialDesign(np.int64(10), np.int64(3))

    def test_correct_class(self):
        self.assertIsInstance(
            WeakDesign.create("finite_field", 10, 3), FiniteFieldPolynomialDesign
        )

    def test_precomputed_weak_design(self):
        weak_design = np.array(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [0, 6, 12, 18, 24],
                [5, 11, 17, 23, 4],
                [10, 16, 22, 3, 9],
                [15, 21, 2, 8, 14],
                [20, 1, 7, 13, 19],
                [0, 11, 22, 8, 19],
                [5, 16, 2, 13, 24],
                [10, 21, 7, 18, 4],
                [15, 1, 12, 23, 9],
                [20, 6, 17, 3, 14],
                [0, 16, 7, 23, 14],
                [5, 21, 12, 3, 19],
                [10, 1, 17, 8, 24],
                [15, 6, 22, 13, 4],
                [20, 11, 2, 18, 9],
                [0, 21, 17, 13, 9],
                [5, 1, 22, 18, 14],
                [10, 6, 2, 23, 19],
                [15, 11, 7, 3, 24],
                [20, 16, 12, 8, 4],
                [0, 6, 22, 23, 9],
                [5, 11, 2, 3, 14],
                [10, 16, 7, 8, 19],
                [15, 21, 12, 13, 24],
                [20, 1, 17, 18, 4],
                [0, 11, 7, 13, 4],
                [5, 16, 12, 18, 9],
                [10, 21, 17, 23, 14],
                [15, 1, 22, 3, 19],
                [20, 6, 2, 8, 24],
                [0, 16, 17, 3, 24],
                [5, 21, 22, 8, 4],
                [10, 1, 2, 13, 9],
                [15, 6, 7, 18, 14],
                [20, 11, 12, 23, 19],
                [0, 21, 2, 18, 19],
                [5, 1, 7, 23, 24],
                [10, 6, 12, 3, 4],
                [15, 11, 17, 8, 9],
                [20, 16, 22, 13, 14],
                [0, 1, 12, 8, 14],
                [5, 6, 17, 13, 19],
                [10, 11, 22, 18, 24],
                [15, 16, 2, 23, 4],
                [20, 21, 7, 3, 9],
                [0, 11, 17, 18, 14],
                [5, 16, 22, 23, 19],
                [10, 21, 2, 3, 24],
                [15, 1, 7, 8, 4],
                [20, 6, 12, 13, 9],
                [0, 16, 2, 8, 9],
                [5, 21, 7, 13, 14],
                [10, 1, 12, 18, 19],
                [15, 6, 17, 23, 24],
                [20, 11, 22, 3, 4],
            ]
        )

        design = WeakDesign.create(
            "finite_field",
            number_of_sets=60,
            size_of_sets=5,
            precomputed_weak_design=weak_design,
            assume_valid=False,
        )
        self.assertIsInstance(design, FiniteFieldPolynomialDesign)
        self.assertTrue(design.is_computed)


class TestWrongInitialization(unittest.TestCase):
    def test_wrong_weak_design_type(self):
        with self.assertRaises(TypeError):
            WeakDesign.create(5, 3, 2)

    def test_wrong_weak_design(self):
        with self.assertRaises(ValueError):
            WeakDesign.create("apple", 3, 2)

    def test_wrong_number_of_sets_type(self):
        with self.assertRaises(TypeError):
            WeakDesign.create("finite_field", "banana", 2)

    def test_wrong_size_of_sets_type(self):
        with self.assertRaises(TypeError):
            WeakDesign.create("finite_field", 3, "pear")

    def test_too_small_number_of_sets(self):
        with self.assertRaises(ValueError):
            WeakDesign.create("finite_field", 0, 2)

    def test_non_prime_size_of_set(self):
        with self.assertRaises(ValueError):
            WeakDesign.create("finite_field", 3, 4)

    def test_wrong_precomputed_weak_design_type(self):
        with self.subTest(msg="str"):
            # There is no file Path.cwd() / hola
            with self.assertRaises(ValueError):
                WeakDesign.create("finite_field", 3, 2, "hola")
        with self.subTest(msg="int"):
            with self.assertRaises(TypeError):
                WeakDesign.create("finite_field", 3, 2, 4)
        with self.subTest(msg="list"):
            with self.assertRaises(TypeError):
                WeakDesign.create("finite_field", 3, 2, [[0, 1], [2, 3], [0, 3]])
        with self.subTest(msg="wrong shape"):
            with self.assertRaises(ValueError):
                WeakDesign.create("finite_field", 3, 2, np.array([[0, 1], [2, 3]]))
        with self.subTest(msg="wrong values"):
            with self.assertRaises(ValueError):
                WeakDesign.create(
                    "finite_field", 3, 2, np.array([[0, 1], [2, 4], [0, 3]])
                )
        with self.subTest(msg="wrong overlap"):
            with self.assertRaises(ValueError):
                WeakDesign.create("finite_field", 60, 5, np.tile(np.arange(5), (60, 1)))

    def test_precomputed_weak_design_assume_valid(self):
        self.assertIsInstance(
            # Now it doesn't fail because we skip the checking
            WeakDesign.create(
                "finite_field", 60, 5, np.tile(np.arange(5), (60, 1)), True
            ),
            FiniteFieldPolynomialDesign,
        )

    def test_get_set_before_computing_design(self):
        test_design = WeakDesign.create("finite_field", 3, 2)
        with self.assertWarns(UserWarning):
            test_design.get_set(0)

    def test_get_set_wrong_index_type(self):
        test_design = WeakDesign.create("finite_field", 3, 2)
        test_design.compute_design()
        with self.assertRaises(TypeError):
            test_design.get_set("tomato")

    def test_get_set_too_small_index(self):
        test_design = WeakDesign.create("finite_field", 3, 2)
        test_design.compute_design()
        with self.assertRaises(ValueError):
            test_design.get_set(-1)

    def test_get_set_too_big_index(self):
        test_design = WeakDesign.create("finite_field", 3, 2)
        test_design.compute_design()
        with self.assertRaises(ValueError):
            test_design.get_set(4)


class TestSmallestDesign(unittest.TestCase):
    def setUp(self):
        number_of_sets = 1
        size_of_sets = 2
        self.GF = galois.GF(size_of_sets**2)
        self.finite_field_polynomial_design = WeakDesign.create(
            "finite_field", number_of_sets, size_of_sets
        )
        self.finite_field_polynomial_design.compute_design()

    def test_weak_design(self):
        expected_weak_design = self.GF([[0, 1]])
        np.testing.assert_array_equal(
            self.finite_field_polynomial_design.weak_design, expected_weak_design
        )


class TestSmallDesign(unittest.TestCase):
    def setUp(self):
        number_of_sets = 2
        size_of_sets = 2
        self.GF = galois.GF(size_of_sets**2)
        self.finite_field_polynomial_design = WeakDesign.create(
            "finite_field", number_of_sets, size_of_sets
        )
        self.finite_field_polynomial_design.compute_design()

    def test_weak_design(self):
        expected_weak_design = self.GF([[0, 1], [2, 3]])
        np.testing.assert_array_equal(
            self.finite_field_polynomial_design.weak_design, expected_weak_design
        )


class TestExample1(unittest.TestCase):
    def setUp(self):
        number_of_sets = 3
        size_of_sets = 2
        self.GF = galois.GF(size_of_sets**2)
        self.finite_field_polynomial_design = WeakDesign.create(
            "finite_field", number_of_sets, size_of_sets
        )
        self.finite_field_polynomial_design.compute_design()

    def test_weak_design(self):
        expected_weak_design = self.GF([[0, 1], [2, 3], [0, 3]])
        np.testing.assert_array_equal(
            self.finite_field_polynomial_design.weak_design,
            expected_weak_design,
        )


class TestExample2(unittest.TestCase):
    def setUp(self):
        number_of_sets = 60
        size_of_sets = 5
        self.GF = galois.GF(size_of_sets**2)
        self.finite_field_polynomial_design = WeakDesign.create(
            "finite_field", number_of_sets, size_of_sets
        )
        self.finite_field_polynomial_design.compute_design()

    def test_weak_design(self):
        expected_weak_design = self.GF(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [0, 6, 12, 18, 24],
                [5, 11, 17, 23, 4],
                [10, 16, 22, 3, 9],
                [15, 21, 2, 8, 14],
                [20, 1, 7, 13, 19],
                [0, 11, 22, 8, 19],
                [5, 16, 2, 13, 24],
                [10, 21, 7, 18, 4],
                [15, 1, 12, 23, 9],
                [20, 6, 17, 3, 14],
                [0, 16, 7, 23, 14],
                [5, 21, 12, 3, 19],
                [10, 1, 17, 8, 24],
                [15, 6, 22, 13, 4],
                [20, 11, 2, 18, 9],
                [0, 21, 17, 13, 9],
                [5, 1, 22, 18, 14],
                [10, 6, 2, 23, 19],
                [15, 11, 7, 3, 24],
                [20, 16, 12, 8, 4],
                [0, 6, 22, 23, 9],
                [5, 11, 2, 3, 14],
                [10, 16, 7, 8, 19],
                [15, 21, 12, 13, 24],
                [20, 1, 17, 18, 4],
                [0, 11, 7, 13, 4],
                [5, 16, 12, 18, 9],
                [10, 21, 17, 23, 14],
                [15, 1, 22, 3, 19],
                [20, 6, 2, 8, 24],
                [0, 16, 17, 3, 24],
                [5, 21, 22, 8, 4],
                [10, 1, 2, 13, 9],
                [15, 6, 7, 18, 14],
                [20, 11, 12, 23, 19],
                [0, 21, 2, 18, 19],
                [5, 1, 7, 23, 24],
                [10, 6, 12, 3, 4],
                [15, 11, 17, 8, 9],
                [20, 16, 22, 13, 14],
                [0, 1, 12, 8, 14],
                [5, 6, 17, 13, 19],
                [10, 11, 22, 18, 24],
                [15, 16, 2, 23, 4],
                [20, 21, 7, 3, 9],
                [0, 11, 17, 18, 14],
                [5, 16, 22, 23, 19],
                [10, 21, 2, 3, 24],
                [15, 1, 7, 8, 4],
                [20, 6, 12, 13, 9],
                [0, 16, 2, 8, 9],
                [5, 21, 7, 13, 14],
                [10, 1, 12, 18, 19],
                [15, 6, 17, 23, 24],
                [20, 11, 22, 3, 4],
            ]
        )
        np.testing.assert_array_equal(
            self.finite_field_polynomial_design.weak_design,
            expected_weak_design,
        )
