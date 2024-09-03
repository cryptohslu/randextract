import unittest

import galois
import numpy as np
from galois import GF2

from randextract.polynomial_hashing import PolynomialOneBitExtractor

from .._types import INTEGERS, NOT_INTEGERS, NOT_REALS, NOT_STRINGS

RNG = np.random.default_rng()
SIZE_SAMPLE = 100


class TestInitialization(unittest.TestCase):
    def test_wrong_input_length(self):
        for not_integer in NOT_INTEGERS:
            with self.subTest(type=type(not_integer)):
                with self.assertRaises(TypeError):
                    PolynomialOneBitExtractor(input_length=not_integer, seed_length=10)

    def test_wrong_seed_length(self):
        for not_integer in NOT_INTEGERS:
            with self.subTest(type=type(not_integer)):
                with self.assertRaises(TypeError):
                    PolynomialOneBitExtractor(input_length=100, seed_length=not_integer)

    def test_wrong_irreducible_poly(self):
        with self.subTest(type="str"):
            with self.assertRaises(TypeError):
                PolynomialOneBitExtractor(
                    input_length=10, seed_length=10, irreducible_poly="x^5 + x^2 + 1"
                )

        with self.subTest(type="float"):
            with self.assertRaises(TypeError):
                PolynomialOneBitExtractor(
                    input_length=10, seed_length=10, irreducible_poly=1.0
                )

        with self.subTest(type="array"):
            with self.assertRaises(TypeError):
                PolynomialOneBitExtractor(
                    input_length=10,
                    seed_length=10,
                    irreducible_poly=np.array([1, 1, 0, 1]),
                )

    def test_good_values(self):
        ext = PolynomialOneBitExtractor(
            input_length=10,
            seed_length=6,
        )
        self.assertEqual(ext.input_length, 10)
        self.assertEqual(ext.seed_length, 6)
        self.assertEqual(ext.output_length, 1)

        ext = PolynomialOneBitExtractor(
            input_length=10,
            seed_length=6,
            irreducible_poly=galois.Poly.Str("x^3 + x^2 + 1"),
        )
        self.assertEqual(ext.irreducible_poly, galois.Poly.Str("x^3 + x^2 + 1"))

        ext = PolynomialOneBitExtractor(
            input_length=10,
            seed_length=6,
            irreducible_poly=galois.Poly.Str("x^3 + x + 1"),
        )
        self.assertEqual(ext.irreducible_poly, galois.Poly.Str("x^3 + x + 1"))

    def test_min_seed_length(self):
        PolynomialOneBitExtractor(
            input_length=10,
            seed_length=2,
        )

    def test_max_seed_length(self):
        PolynomialOneBitExtractor(
            input_length=10,
            seed_length=10,
        )

        PolynomialOneBitExtractor(
            input_length=11,
            seed_length=10,
        )

    def test_seed_length_too_large(self):
        with self.assertRaises(ValueError):
            PolynomialOneBitExtractor(
                input_length=10,
                seed_length=22,
            )

    def test_seed_length_odd(self):
        with self.assertWarns(UserWarning):
            ext = PolynomialOneBitExtractor(
                input_length=10,
                seed_length=9,
            )
        self.assertEqual(ext.seed_length, 10)

    def test_smallest_case(self):
        PolynomialOneBitExtractor(
            input_length=2,
            seed_length=2,
        )

    def test_given_irreducible_poly_gf2(self):
        with self.assertWarns(UserWarning):
            PolynomialOneBitExtractor(
                input_length=2, seed_length=2, irreducible_poly=galois.Poly.Str("x + 1")
            )

    def test_huge_seed_length_without_irreducible_poly(self):
        with self.assertRaises(ValueError):
            PolynomialOneBitExtractor(
                input_length=10**5,
                seed_length=20_002,
            )

    def test_huge_seed_length_with_irreducible_poly(self):
        pass


class TestCalculateLength(unittest.TestCase):
    def test_wrong_extractor_type_type(self):
        for not_string in NOT_STRINGS:
            with self.assertRaises(TypeError):
                PolynomialOneBitExtractor.calculate_length(
                    extractor_type=not_string,
                    input_length=100,
                    relative_source_entropy=0.9,
                    error_bound=1e-3,
                )

    def test_wrong_extractor_type_value(self):
        for ext_type in ["strong", "weak"]:
            with self.assertRaises(ValueError):
                PolynomialOneBitExtractor.calculate_length(
                    extractor_type=ext_type,
                    input_length=100,
                    relative_source_entropy=0.9,
                    error_bound=1e-3,
                )

    def test_wrong_input_length_type(self):
        for not_integer in NOT_INTEGERS:
            with self.assertRaises(TypeError):
                PolynomialOneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=not_integer,
                    relative_source_entropy=0.9,
                    error_bound=1e-3,
                )

    def test_wrong_input_length_value(self):
        for input_length in [-1, 0]:
            with self.assertRaises(ValueError):
                PolynomialOneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=input_length,
                    relative_source_entropy=0.9,
                    error_bound=1e-3,
                )

    def test_wrong_relative_source_entropy_type(self):
        for not_real in NOT_REALS:
            with self.assertRaises(TypeError):
                PolynomialOneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=100,
                    relative_source_entropy=not_real,
                    error_bound=1e-3,
                )

    def test_wrong_relative_source_entropy_value(self):
        for k in [-0.5, 0, 1.5]:
            with self.assertRaises(ValueError):
                PolynomialOneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=100,
                    relative_source_entropy=k,
                    error_bound=1e-3,
                )

    def test_wrong_error_bound_type(self):
        for not_real in NOT_REALS:
            with self.assertRaises(TypeError):
                PolynomialOneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=100,
                    relative_source_entropy=0.9,
                    error_bound=not_real,
                )

    def test_wrong_error_bound_value(self):
        for eps in [-0.5, 0, 1.5]:
            with self.assertRaises(ValueError):
                PolynomialOneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=100,
                    relative_source_entropy=0.9,
                    error_bound=eps,
                )

    def test_extraction_not_possible(self):
        with self.assertRaises(ValueError):
            PolynomialOneBitExtractor.calculate_length(
                extractor_type="classical",
                input_length=100,
                relative_source_entropy=0.1,
                error_bound=1e-3,
            )

        with self.assertRaises(ValueError):
            PolynomialOneBitExtractor.calculate_length(
                extractor_type="quantum",
                input_length=100,
                relative_source_entropy=0.5,
                error_bound=1e-4,
            )

    def test_n_100(self):
        n = 100
        with self.subTest(k=0.5):
            k = 0.5
            with self.subTest(type="classical"):
                expected_seed_length = [94, 80, 68, 56, 42, 30, 20]
                for i, exp in enumerate(range(-7, 0, 1)):
                    self.assertEqual(
                        PolynomialOneBitExtractor.calculate_length(
                            extractor_type="classical",
                            input_length=n,
                            relative_source_entropy=k,
                            error_bound=10**exp,
                        ),
                        expected_seed_length[i],
                    )
            with self.subTest(type="quantum"):
                expected_seed_length = [46, 34, 22]
                for i, exp in enumerate(range(-3, 0, 1)):
                    self.assertEqual(
                        PolynomialOneBitExtractor.calculate_length(
                            extractor_type="quantum",
                            input_length=n,
                            relative_source_entropy=k,
                            error_bound=10**exp,
                        ),
                        expected_seed_length[i],
                    )

        with self.subTest(k=0.7):
            k = 0.7
            with self.subTest(type="classical"):
                expected_seed_length = [94, 80, 68, 56, 42, 30, 20]
                for i, exp in enumerate(range(-7, 0, 1)):
                    self.assertEqual(
                        PolynomialOneBitExtractor.calculate_length(
                            extractor_type="classical",
                            input_length=n,
                            relative_source_entropy=k,
                            error_bound=10**exp,
                        ),
                        expected_seed_length[i],
                    )
            with self.subTest(type="quantum"):
                expected_seed_length = [60, 46, 34, 22]
                for i, exp in enumerate(range(-4, 0, 1)):
                    self.assertEqual(
                        PolynomialOneBitExtractor.calculate_length(
                            extractor_type="quantum",
                            input_length=n,
                            relative_source_entropy=k,
                            error_bound=10**exp,
                        ),
                        expected_seed_length[i],
                    )

        with self.subTest(k=0.9):
            k = 0.9
            with self.subTest(type="classical"):
                expected_seed_length = [94, 80, 68, 56, 42, 30, 20]
                for i, exp in enumerate(range(-7, 0, 1)):
                    self.assertEqual(
                        PolynomialOneBitExtractor.calculate_length(
                            extractor_type="classical",
                            input_length=n,
                            relative_source_entropy=k,
                            error_bound=10**exp,
                        ),
                        expected_seed_length[i],
                    )
            with self.subTest(type="quantum"):
                expected_seed_length = [84, 72, 60, 46, 34, 22]
                for i, exp in enumerate(range(-6, 0, 1)):
                    self.assertEqual(
                        PolynomialOneBitExtractor.calculate_length(
                            extractor_type="quantum",
                            input_length=n,
                            relative_source_entropy=k,
                            error_bound=10**exp,
                        ),
                        expected_seed_length[i],
                    )

    def test_n_1000(self):
        n = 1000
        # Bound on min-entropy has no impact on required seed length with this input length
        with self.subTest(type="classical"):
            expected_seed_length = [100, 88, 74, 62, 50, 38, 26]
            for i, exp in enumerate(range(-7, 0, 1)):
                for k in [0.5, 0.7, 0.9]:
                    self.assertEqual(
                        PolynomialOneBitExtractor.calculate_length(
                            extractor_type="classical",
                            input_length=n,
                            relative_source_entropy=k,
                            error_bound=10**exp,
                        ),
                        expected_seed_length[i],
                    )
        with self.subTest(type="quantum"):
            expected_seed_length = [104, 92, 78, 66, 54, 40, 28]
            for i, exp in enumerate(range(-7, 0, 1)):
                for k in [0.5, 0.7, 0.9]:
                    self.assertEqual(
                        PolynomialOneBitExtractor.calculate_length(
                            extractor_type="quantum",
                            input_length=n,
                            relative_source_entropy=k,
                            error_bound=10**exp,
                        ),
                        expected_seed_length[i],
                    )


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.ext = PolynomialOneBitExtractor(input_length=20, seed_length=6)

    def test_extractor_input(self):
        seed = GF2.Random(6, seed=RNG)

        with self.subTest(msg="binary numpy array"):
            for i in range(SIZE_SAMPLE):
                extractor_input = RNG.integers(low=0, high=2, size=20)
                self.ext.extract(extractor_input, seed)

        with self.subTest(msg="GF2 array"):
            for i in range(SIZE_SAMPLE):
                extractor_input = GF2.Random(20, seed=RNG)
                self.ext.extract(extractor_input, seed)

    def test_seed(self):
        extractor_input = GF2.Random(20, seed=RNG)

        with self.subTest(msg="binary numpy array"):
            for i in range(SIZE_SAMPLE):
                seed = RNG.integers(low=0, high=2, size=6)
                self.ext.extract(extractor_input, seed)

        with self.subTest(msg="GF2 array"):
            for i in range(SIZE_SAMPLE):
                seed = GF2.Random(6, seed=RNG)
                self.ext.extract(extractor_input, seed)

    def test_wrong_extractor_input_type(self):
        seed = GF2.Random(6, seed=RNG)
        good_input = GF2.Random(20, seed=RNG)
        wrong_inputs = [
            good_input.tolist(),
            tuple(good_input.tolist()),
            1234,
            "input",
            None,
        ]
        for wrong_input in wrong_inputs:
            with self.assertRaises(TypeError):
                self.ext.extract(wrong_input, seed)

    def test_wrong_extractor_input_values(self):
        seed = GF2.Random(6, seed=RNG)
        wrong_inputs = [
            RNG.integers(low=2, high=10, size=20),
            GF2.Random(shape=(1, 20), seed=RNG),
            GF2.Random(19, seed=RNG),
            GF2.Random(21, seed=RNG),
        ]
        for wrong_input in wrong_inputs:
            with self.assertRaises(ValueError):
                self.ext.extract(wrong_input, seed)

    def test_wrong_seed_type(self):
        ext_input = GF2.Random(20, seed=RNG)
        good_seed = GF2.Random(6, seed=RNG)
        wrong_seeds = [
            good_seed.tolist(),
            tuple(good_seed.tolist()),
            1354,
            "seed",
            None,
        ]
        for wrong_seed in wrong_seeds:
            with self.assertRaises(TypeError):
                self.ext.extract(ext_input, wrong_seed)

    def test_wrong_seed_values(self):
        ext_input = GF2.Random(20, seed=RNG)
        wrong_seeds = [
            RNG.integers(low=2, high=10, size=6),
            GF2.Random(shape=(1, 6), seed=RNG),
            GF2.Random(5),
            GF2.Random(7),
        ]
        for wrong_seed in wrong_seeds:
            with self.assertRaises(ValueError):
                self.ext.extract(ext_input, wrong_seed)


class TestReedSolomonHashingL1(unittest.TestCase):
    def setUp(self):
        self.ext = PolynomialOneBitExtractor(
            input_length=2,
            seed_length=2,
        )

    def test_l_1(self):
        with self.subTest(msg="x=[0,0], s=0"):
            input_extractor = GF2([0, 0])
            seed = GF2([0])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([0])
            )
        with self.subTest(msg="x=[0,1], s=0"):
            input_extractor = GF2([0, 1])
            seed = GF2([0])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([1])
            )
        with self.subTest(msg="x=[1,0], s=0"):
            input_extractor = GF2([1, 0])
            seed = GF2([0])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([0])
            )
        with self.subTest(msg="x=[1,1], s=0"):
            input_extractor = GF2([1, 1])
            seed = GF2([0])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([1])
            )
        with self.subTest(msg="x=[0,0], s=1"):
            input_extractor = GF2([0, 0])
            seed = GF2([1])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([0])
            )
        with self.subTest(msg="x=[0,1], s=1"):
            input_extractor = GF2([0, 1])
            seed = GF2([1])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([1])
            )
        with self.subTest(msg="x=[1,0], s=1"):
            input_extractor = GF2([1, 0])
            seed = GF2([1])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([1])
            )
        with self.subTest(msg="x=[1,1], s=1"):
            input_extractor = GF2([1, 1])
            seed = GF2([1])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([0])
            )


class TestReedSolomonHashingL4(unittest.TestCase):
    def setUp(self):
        self.ext = PolynomialOneBitExtractor(
            input_length=8,
            seed_length=8,
        )

    def test_l_4(self):
        with self.subTest(msg="x=[0, 0, 0, 1, 0, 1, 0, 1], s=[0, 1, 1, 1]"):
            input_extractor = GF2([0, 0, 0, 1, 0, 1, 0, 1])
            seed = GF2([0, 1, 1, 1])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([0, 0, 1, 0])
            )
        with self.subTest(msg="x=[1, 0, 1, 0, 0, 1, 1, 1], s=[0, 1, 0, 1]"):
            input_extractor = GF2([1, 0, 1, 0, 0, 1, 1, 1])
            seed = GF2([0, 1, 0, 1])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([0, 0, 1, 1])
            )
        with self.subTest(msg="x=[0, 1, 0, 1, 0, 0, 0, 0], s=[1, 0, 0, 1]"):
            input_extractor = GF2([0, 1, 0, 1, 0, 0, 0, 0])
            seed = GF2([1, 0, 0, 1])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([1, 0, 1, 1])
            )
        with self.subTest(msg="x=[0, 0, 0, 1, 1, 1, 1, 1], s=[1, 1, 1, 0]"):
            input_extractor = GF2([0, 0, 0, 1, 1, 1, 1, 1])
            seed = GF2([1, 1, 1, 0])
            np.testing.assert_array_equal(
                self.ext._reed_solomon_hashing(input_extractor, seed), GF2([0, 0, 0, 1])
            )


class TestReedSolomonHashingExamples(unittest.TestCase):
    def test_correctness_example_1(self):
        ext = PolynomialOneBitExtractor(
            input_length=25,
            seed_length=24,
        )

        input_extractor = GF2(
            [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        )
        seed = GF2([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1])

        np.testing.assert_array_equal(
            ext._reed_solomon_hashing(input_extractor, seed),
            GF2([1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
        )

    def test_correctness_example_1_custom_poly(self):
        ext = PolynomialOneBitExtractor(
            input_length=25, seed_length=24, irreducible_poly=galois.conway_poly(2, 12)
        )

        input_extractor = GF2(
            [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        )
        seed = GF2([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1])

        np.testing.assert_array_equal(
            ext._reed_solomon_hashing(input_extractor, seed),
            GF2([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]),
        )

    def test_correctness_example_2(self):
        ext = PolynomialOneBitExtractor(
            input_length=25,
            seed_length=24,
        )

        input_extractor = GF2(
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        )
        seed = GF2([1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0])

        np.testing.assert_array_equal(
            ext._reed_solomon_hashing(input_extractor, seed),
            GF2([0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0]),
        )

    def test_correctness_example_2_custom_poly(self):
        ext = PolynomialOneBitExtractor(
            input_length=25, seed_length=24, irreducible_poly=galois.conway_poly(2, 12)
        )

        input_extractor = GF2(
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
        )
        seed = GF2([1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0])

        np.testing.assert_array_equal(
            ext._reed_solomon_hashing(input_extractor, seed),
            GF2([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]),
        )

    def test_correctness_example_3(self):
        ext = PolynomialOneBitExtractor(
            input_length=14,
            seed_length=12,
        )

        input_extractor = GF2([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0])
        seed = GF2([1, 1, 0, 0, 0, 1])

        np.testing.assert_array_equal(
            ext._reed_solomon_hashing(input_extractor, seed),
            GF2([1, 1, 0, 1, 0, 0]),
        )

    def test_correctness_example_3_custom_poly(self):
        ext = PolynomialOneBitExtractor(
            input_length=14, seed_length=12, irreducible_poly=galois.conway_poly(2, 6)
        )

        input_extractor = GF2([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0])
        seed = GF2([1, 1, 0, 0, 0, 1])

        np.testing.assert_array_equal(
            ext._reed_solomon_hashing(input_extractor, seed),
            GF2([1, 0, 0, 1, 0, 0]),
        )


class TestHadamardHashingL1(unittest.TestCase):
    def setUp(self):
        self.ext = PolynomialOneBitExtractor(input_length=2, seed_length=2)

    def test_l_1(self):
        with self.subTest(msg="x=0, s=0"):
            self.ext._input_length = 2
            self.ext._seed_length = 2
            input_extractor = GF2(0)
            seed = GF2(0)
            np.testing.assert_array_equal(
                self.ext._hadamard_hashing(input_extractor, seed), GF2(0)
            )
        with self.subTest(msg="x=0, s=1"):
            self.ext._input_length = 2
            self.ext._seed_length = 2
            input_extractor = GF2(0)
            seed = GF2(1)
            np.testing.assert_array_equal(
                self.ext._hadamard_hashing(input_extractor, seed), GF2(0)
            )
        with self.subTest(msg="x=1, s=0"):
            self.ext._input_length = 2
            self.ext._seed_length = 2
            input_extractor = GF2(1)
            seed = GF2(0)
            np.testing.assert_array_equal(
                self.ext._hadamard_hashing(input_extractor, seed), GF2(0)
            )
        with self.subTest(msg="x=1, s=1"):
            self.ext._input_length = 2
            self.ext._seed_length = 2
            input_extractor = GF2(1)
            seed = GF2(1)
            np.testing.assert_array_equal(
                self.ext._hadamard_hashing(input_extractor, seed), GF2(1)
            )


class TestHadamardHashingL4(unittest.TestCase):
    def setUp(self):
        # This is just to access the static method _hadamard_hashing()
        self.ext = PolynomialOneBitExtractor(input_length=2, seed_length=2)

    def test_l_4(self):
        with self.subTest(msg="x=[1, 0, 1, 1], s=[1, 1, 0, 0]"):
            input_extractor = GF2([1, 0, 1, 1])
            seed = GF2([1, 1, 0, 0])
            np.testing.assert_array_equal(
                self.ext._hadamard_hashing(input_extractor, seed), GF2(1)
            )
        with self.subTest(msg="x=[1, 1, 0, 0], s=[0, 0, 1, 1]"):
            input_extractor = GF2([1, 1, 0, 0])
            seed = GF2([0, 0, 1, 1])
            np.testing.assert_array_equal(
                self.ext._hadamard_hashing(input_extractor, seed), GF2(0)
            )
        with self.subTest(msg="x=[1, 1, 1, 1], s=[1, 1, 0, 1]"):
            input_extractor = GF2([1, 1, 1, 1])
            seed = GF2([1, 1, 0, 1])
            np.testing.assert_array_equal(
                self.ext._hadamard_hashing(input_extractor, seed), GF2(1)
            )
        with self.subTest(msg="x=[1, 1, 0, 1], s=[1, 0, 0, 0]"):
            input_extractor = GF2([1, 1, 0, 1])
            seed = GF2([1, 0, 0, 0])
            np.testing.assert_array_equal(
                self.ext._hadamard_hashing(input_extractor, seed), GF2(1)
            )


class TestHadamardHashingExamples(unittest.TestCase):
    def setUp(self):
        # This is just to access the static method _hadamard_hashing()
        self.ext = PolynomialOneBitExtractor(input_length=2, seed_length=2)

    def test_correctness_example_1(self):
        input_extractor = GF2([1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        seed = GF2([1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1])

        np.testing.assert_array_equal(
            self.ext._hadamard_hashing(input_extractor, seed), 0
        )

    def test_correctness_example_1_custom_poly(self):
        input_extractor = GF2([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        seed = GF2([1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1])

        np.testing.assert_array_equal(
            self.ext._hadamard_hashing(input_extractor, seed), 1
        )

    def test_correctness_example_2(self):
        input_extractor = GF2([0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0])
        seed = GF2([0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0])

        np.testing.assert_array_equal(
            self.ext._hadamard_hashing(input_extractor, seed), 1
        )

    def test_correctness_example_2_custom_poly(self):
        input_extractor = GF2([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        seed = GF2([0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0])

        np.testing.assert_array_equal(
            self.ext._hadamard_hashing(input_extractor, seed), 1
        )

    def test_correctness_example_3(self):
        self.ext._input_length = 14
        self.ext._seed_length = 12
        input_extractor = GF2([1, 1, 0, 1, 0, 0])
        seed = GF2([0, 1, 1, 0, 0, 1])

        np.testing.assert_array_equal(
            self.ext._hadamard_hashing(input_extractor, seed), 1
        )

    def test_correctness_example_3_custom_poly(self):
        self.ext._input_length = 14
        self.ext._seed_length = 12
        self.ext._reed_solomon_irreducible_polynomial = galois.conway_poly(2, 6)
        input_extractor = GF2([1, 0, 0, 1, 0, 0])
        seed = GF2([0, 1, 1, 0, 0, 1])

        np.testing.assert_array_equal(
            self.ext._hadamard_hashing(input_extractor, seed), 0
        )


class TestCorrectness(unittest.TestCase):
    def test_correctness_example_1(self):
        ext = PolynomialOneBitExtractor(
            input_length=25,
            seed_length=24,
        )

        # fmt: off
        input_extractor = GF2([1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        seed = GF2([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1])
        # fmt: on

        np.testing.assert_array_equal(ext.extract(input_extractor, seed), GF2([0]))

    def test_correctness_example_1_custom_poly(self):
        ext = PolynomialOneBitExtractor(
            input_length=25, seed_length=24, irreducible_poly=galois.conway_poly(2, 12)
        )

        # fmt: off
        input_extractor = GF2([1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        seed = GF2([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1])
        # fmt: on

        np.testing.assert_array_equal(ext.extract(input_extractor, seed), GF2([1]))

    def test_correctness_example_2(self):
        ext = PolynomialOneBitExtractor(
            input_length=25,
            seed_length=24,
        )

        # fmt: off
        input_extractor = GF2([1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1])
        seed = GF2([1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0])
        # fmt: on

        np.testing.assert_array_equal(ext.extract(input_extractor, seed), GF2([1]))

    def test_correctness_example_2_custom_poly(self):
        ext = PolynomialOneBitExtractor(
            input_length=25, seed_length=24, irreducible_poly=galois.conway_poly(2, 12)
        )

        # fmt: off
        input_extractor = GF2([1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1])
        seed = GF2([1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0])
        # fmt: on

        np.testing.assert_array_equal(ext.extract(input_extractor, seed), GF2([1]))

    def test_correctness_example_3(self):
        ext = PolynomialOneBitExtractor(
            input_length=14,
            seed_length=12,
        )

        input_extractor = GF2([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0])
        seed = GF2([1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1])

        np.testing.assert_array_equal(ext.extract(input_extractor, seed), GF2([1]))

    def test_correctness_example_3_custom_poly(self):
        ext = PolynomialOneBitExtractor(
            input_length=14, seed_length=12, irreducible_poly=galois.conway_poly(2, 6)
        )

        input_extractor = GF2([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0])
        seed = GF2([1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1])

        np.testing.assert_array_equal(ext.extract(input_extractor, seed), GF2([0]))
