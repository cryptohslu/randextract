import itertools
import math
import unittest

import numpy as np
from galois import GF, GF2

from randextract.toeplitz_hashing import ModifiedToeplitzHashing

from .._types import INTEGERS, NOT_INTEGERS, NOT_REALS, NOT_STRINGS

RNG = np.random.default_rng()
SIZE_SAMPLE = 100


class TestInitialization(unittest.TestCase):
    def test_wrong_input_length(self):
        for not_integer in NOT_INTEGERS:
            with self.subTest(type=type(not_integer)):
                with self.assertRaises(TypeError):
                    ModifiedToeplitzHashing(input_length=not_integer, seed_length=2)

    def test_wrong_output_length(self):
        for not_integer in NOT_INTEGERS:
            with self.subTest(type=type(not_integer)):
                with self.assertRaises(TypeError):
                    ModifiedToeplitzHashing(input_length=10**3, seed_length=not_integer)

    def test_good_input_length(self):
        for integer in INTEGERS:
            integer += type(integer)(10)
            with self.subTest(type=type(integer)):
                ModifiedToeplitzHashing(input_length=integer, output_length=2)

    def test_good_output_length(self):
        for integer in INTEGERS:
            integer += type(integer)(10)
            with self.subTest(type=type(integer)):
                ModifiedToeplitzHashing(input_length=10**3, output_length=integer)

    def test_smallest_case(self):
        ext = ModifiedToeplitzHashing(input_length=1, output_length=1)
        self.assertEqual(ext.input_length, 1)
        self.assertEqual(ext.output_length, 1)
        self.assertEqual(ext.seed_length, 0)

    def test_smallest_useful_case(self):
        ext = ModifiedToeplitzHashing(input_length=2, output_length=1)
        self.assertEqual(ext.input_length, 2)
        self.assertEqual(ext.output_length, 1)
        self.assertEqual(ext.seed_length, 1)

    def test_same_input_output_lengths(self):
        ext = ModifiedToeplitzHashing(input_length=100, output_length=100)
        self.assertEqual(ext.input_length, 100)
        self.assertEqual(ext.output_length, 100)
        self.assertEqual(ext.seed_length, 0)

    def test_input_length_too_small(self):
        for i in [-1, 0]:
            with self.assertRaises(ValueError):
                ModifiedToeplitzHashing(input_length=i, output_length=1)

    def test_output_length_too_small(self):
        with self.assertRaises(ValueError):
            ModifiedToeplitzHashing(input_length=2, output_length=0)
        with self.assertRaises(ValueError):
            ModifiedToeplitzHashing(input_length=2, output_length=-1)

    def test_output_length_too_large(self):
        with self.assertRaises(ValueError):
            ModifiedToeplitzHashing(input_length=2, output_length=3)

    def test_examples_docstring(self):
        with self.subTest("example1"):
            ext = ModifiedToeplitzHashing(input_length=10**6, output_length=10**5)
            self.assertEqual(ext.input_length, 1_000_000)
            self.assertEqual(ext.output_length, 100_000)

        with self.subTest("example2"):
            ext = ModifiedToeplitzHashing(input_length=20, output_length=8)
            input_array = GF2.Random(ext.input_length, seed=RNG)
            seed = GF2.Random(ext.seed_length, seed=RNG)
            toeplitz_matrix = ext.to_matrix(seed)
            out1 = toeplitz_matrix @ input_array
            out2 = ext.extract(input_array, seed)
            np.testing.assert_array_equal(out1, out2)

    def test_random_correct_input_output(self):
        for _ in range(SIZE_SAMPLE):
            input_length = RNG.integers(2, 10**8)
            output_length = RNG.integers(1, input_length)
            ext = ModifiedToeplitzHashing(input_length, output_length)
            self.assertEqual(ext.input_length, input_length)
            self.assertEqual(ext.output_length, output_length)
            self.assertEqual(ext.seed_length, max(input_length - 1, 0))


class TestCalculateLength(unittest.TestCase):
    def test_wrong_extractor_type_type(self):
        for not_string in NOT_STRINGS:
            with self.assertRaises(TypeError):
                ModifiedToeplitzHashing.calculate_length(
                    extractor_type=not_string,
                    input_length=10,
                    relative_source_entropy=0.9,
                    error_bound=1e-6,
                )

    def test_wrong_extractor_type_value(self):
        for ext_type in ["strong", "weak"]:
            with self.assertRaises(ValueError):
                ModifiedToeplitzHashing.calculate_length(
                    extractor_type=ext_type,
                    input_length=10,
                    relative_source_entropy=0.9,
                    error_bound=1e-6,
                )

    def test_wrong_input_length_type(self):
        for not_integer in NOT_INTEGERS:
            with self.assertRaises(TypeError):
                ModifiedToeplitzHashing.calculate_length(
                    extractor_type="quantum",
                    input_length=not_integer,
                    relative_source_entropy=0.9,
                    error_bound=1e-6,
                )

    def test_wrong_input_length_value(self):
        for input_length in [-1, 0]:
            with self.assertRaises(ValueError):
                ModifiedToeplitzHashing.calculate_length(
                    extractor_type="quantum",
                    input_length=input_length,
                    relative_source_entropy=0.9,
                    error_bound=1e-6,
                )

    def test_wrong_relative_source_entropy_type(self):
        for not_real in NOT_REALS:
            with self.assertRaises(TypeError):
                ModifiedToeplitzHashing.calculate_length(
                    extractor_type="quantum",
                    input_length=10,
                    relative_source_entropy=not_real,
                    error_bound=1e-6,
                )

    def test_wrong_relative_source_entropy_value(self):
        for k in [-0.5, 0, 1.5]:
            with self.assertRaises(ValueError):
                ModifiedToeplitzHashing.calculate_length(
                    extractor_type="quantum",
                    input_length=10,
                    relative_source_entropy=k,
                    error_bound=1e-6,
                )

    def test_wrong_error_bound_type(self):
        for not_real in NOT_REALS:
            with self.assertRaises(TypeError):
                ModifiedToeplitzHashing.calculate_length(
                    extractor_type="quantum",
                    input_length=10,
                    relative_source_entropy=0.9,
                    error_bound=not_real,
                )

    def test_wrong_error_bound_value(self):
        for eps in [-0.5, 0, 1.5]:
            with self.assertRaises(ValueError):
                ModifiedToeplitzHashing.calculate_length(
                    extractor_type="quantum",
                    input_length=10,
                    relative_source_entropy=0.9,
                    error_bound=eps,
                )

    def test_output_zero(self):
        for k, eps in itertools.product([0.5, 0.7, 0.9], [1e-9, 1e-6, 1e-3]):
            self.assertEqual(
                ModifiedToeplitzHashing.calculate_length(
                    extractor_type="quantum",
                    input_length=10,
                    relative_source_entropy=k,
                    error_bound=eps,
                ),
                0,
            )

    def test_n_1000(self):
        input_length = 1000
        with self.subTest("k=0.5"):
            k = 0.5
            expected_output_length = [442, 462, 482, 500]
            for _, exp in enumerate(range(-9, 1, 3)):
                self.assertEqual(
                    ModifiedToeplitzHashing.calculate_length(
                        extractor_type="classical",
                        input_length=input_length,
                        relative_source_entropy=k,
                        error_bound=10**exp,
                    ),
                    expected_output_length[_],
                )

        with self.subTest("k=0.7"):
            k = 0.7
            expected_output_length = [642, 662, 682, 700]
            for _, exp in enumerate(range(-9, 1, 3)):
                self.assertEqual(
                    ModifiedToeplitzHashing.calculate_length(
                        extractor_type="quantum",
                        input_length=input_length,
                        relative_source_entropy=k,
                        error_bound=10**exp,
                    ),
                    expected_output_length[_],
                )

        with self.subTest("k=0.9"):
            k = 0.9
            expected_output_length = [842, 862, 882, 900]
            for _, exp in enumerate(range(-9, 1, 3)):
                self.assertEqual(
                    ModifiedToeplitzHashing.calculate_length(
                        extractor_type="quantum",
                        input_length=input_length,
                        relative_source_entropy=k,
                        error_bound=10**exp,
                    ),
                    expected_output_length[_],
                )

    def test_max_error_bound(self):
        for n in [100, 1000, 2000]:
            self.assertEqual(
                ModifiedToeplitzHashing.calculate_length(
                    extractor_type="quantum",
                    input_length=n,
                    relative_source_entropy=0.6,
                    error_bound=1,
                ),
                math.floor(0.6 * n),
            )

    def test_same_input_output_length(self):
        for n in [100, 1000, 2000]:
            self.assertEqual(
                ModifiedToeplitzHashing.calculate_length(
                    extractor_type="quantum",
                    input_length=n,
                    relative_source_entropy=1,
                    error_bound=1,
                ),
                n,
            )


class TestToMatrix(unittest.TestCase):
    def setUp(self):
        self.ext = ModifiedToeplitzHashing(input_length=3, output_length=1)

    def test_wrong_seed_type(self):
        for wrong_seed in [[1, 0], (1, 0), 2, "seed"]:
            with self.assertRaises(TypeError):
                self.ext.to_matrix(wrong_seed)

    def test_wrong_seed_value(self):
        for wrong_seed in [
            np.array([2, 3]),  # Wrong range
            GF(5)([1, 2]),  # Wrong range
            GF2([[1, 0]]),  # Wrong dimension
            GF2([0, 1, 0]),  # Wrong size
        ]:
            with self.assertRaises(ValueError):
                self.ext.to_matrix(wrong_seed)

    def test_wrong_seed_mode_type(self):
        for not_string in NOT_STRINGS:
            with self.assertRaises(TypeError):
                self.ext.to_matrix(GF2([1, 0]), seed_mode=not_string)

    def test_wrong_seed_mode_value(self):
        for seed_mode in ["standard", "reverse"]:
            with self.assertRaises(ValueError):
                self.ext.to_matrix(GF2([1, 0]), seed_mode=seed_mode)

    def test_missing_seed_order(self):
        with self.assertRaises(ValueError):
            self.ext.to_matrix(GF2([1, 0]), seed_mode="custom")

    def test_wrong_seed_order_type(self):
        for wrong_seed_order in [[1, 0], (1, 0), 2, "seed_order"]:
            with self.assertRaises(TypeError):
                self.ext.to_matrix(
                    GF2([1, 0]), seed_mode="custom", seed_order=wrong_seed_order
                )

    def test_wrong_seed_order_value(self):
        for wrong_seed_order in [
            np.array([2, 3]),  # Wrong range
            GF(5)([1, 2]),  # Wrong range
            GF2([[1, 0]]),  # Wrong dimension
            GF2([0, 1, 0]),  # Wrong size
            GF2([1, 1]),  # Not all bits are used
        ]:
            with self.assertRaises(ValueError):
                self.ext.to_matrix(
                    GF2([1, 0]), seed_mode="custom", seed_order=wrong_seed_order
                )

    def test_seed_order_without_seed_mode(self):
        with self.assertWarns(UserWarning):
            self.ext.to_matrix(GF2([1, 0]), seed_order=np.array([0, 1]))


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.ext = ModifiedToeplitzHashing(input_length=3, output_length=1)

    def test_wrong_extractor_input_type(self):
        for wrong_input in [[1, 0, 1], (1, 0, 1), 2, "input"]:
            with self.assertRaises(TypeError):
                self.ext.extract(wrong_input, seed=GF2([1, 1]))

    def test_wrong_extractor_input_value(self):
        for wrong_input in [
            np.array([2, 3, 1]),  # Wrong range
            GF(5)([1, 2, 3]),  # Wrong range
            GF2([[1, 0, 1]]),  # Wrong dimension
            GF2([0, 1]),  # Wrong size
        ]:
            with self.assertRaises(ValueError):
                self.ext.extract(wrong_input, seed=GF2([1, 1]))

    def test_wrong_seed_type(self):
        for wrong_seed in [[1, 0], (1, 0), 2, "seed"]:
            with self.assertRaises(TypeError):
                self.ext.extract(GF2([1, 1]), seed=wrong_seed)

    def test_wrong_seed_value(self):
        for wrong_seed in [
            np.array([2, 3]),  # Wrong range
            GF(5)([1, 2]),  # Wrong range
            GF2([[1, 0]]),  # Wrong dimension
            GF2([0, 1, 0]),  # Wrong size
        ]:
            with self.assertRaises(ValueError):
                self.ext.extract(GF2([1, 1]), seed=wrong_seed)

    def test_wrong_seed_mode_type(self):
        for not_string in NOT_STRINGS:
            with self.assertRaises(TypeError):
                self.ext.extract(GF2([1, 1, 1]), seed=GF2([1, 0]), seed_mode=not_string)

    def test_wrong_seed_mode_value(self):
        for seed_mode in ["standard", "reverse"]:
            with self.assertRaises(ValueError):
                self.ext.extract(GF2([1, 1, 1]), seed=GF2([1, 0]), seed_mode=seed_mode)

    def test_missing_seed_order(self):
        with self.assertRaises(ValueError):
            self.ext.extract(GF2([1, 1, 1]), seed=GF2([1, 0]), seed_mode="custom")

    def test_wrong_seed_order_type(self):
        for wrong_seed_order in [[1, 0], (1, 0), 2, "seed_order"]:
            with self.assertRaises(TypeError):
                self.ext.extract(
                    GF2([1, 1, 1]),
                    seed=GF2([1, 0]),
                    seed_mode="custom",
                    seed_order=wrong_seed_order,
                )

    def test_wrong_seed_order_value(self):
        for wrong_seed_order in [
            np.array([2, 3]),  # Wrong range
            GF(5)([1, 2]),  # Wrong range
            GF2([[1, 0]]),  # Wrong dimension
            GF2([0, 1, 0]),  # Wrong size
            GF2([1, 1]),  # Not all bits are used
        ]:
            with self.assertRaises(ValueError):
                self.ext.extract(
                    GF2([1, 1, 1]),
                    seed=GF2([1, 0]),
                    seed_mode="custom",
                    seed_order=wrong_seed_order,
                )

    def test_seed_order_without_seed_mode(self):
        with self.assertWarns(UserWarning):
            self.ext.extract(GF2([1, 1, 1]), GF2([1, 0]), seed_order=np.array([0, 1]))


class TestCorrectnessInput1Output1(unittest.TestCase):
    def setUp(self):
        self.ext = ModifiedToeplitzHashing(input_length=1, output_length=1)

    def test_input_0(self):
        extractor_input = GF2([0])
        seed = GF2([])
        expected_result = GF2([0])
        with self.assertWarns(UserWarning):
            obtained_result = self.ext.extract(extractor_input, seed)
            np.testing.assert_array_equal(expected_result, obtained_result)
        obtained_result = self.ext.extract(extractor_input, None)
        np.testing.assert_array_equal(expected_result, obtained_result)

    def test_input_1(self):
        extractor_input = GF2([1])
        seed = GF2([0])
        expected_result = GF2([1])
        with self.assertWarns(UserWarning):
            obtained_result = self.ext.extract(extractor_input, seed)
            np.testing.assert_array_equal(expected_result, obtained_result)
        obtained_result = self.ext.extract(extractor_input, None)
        np.testing.assert_array_equal(expected_result, obtained_result)


class TestCorrectnessInput10Output10(unittest.TestCase):
    def setUp(self):
        self.ext = ModifiedToeplitzHashing(input_length=10, output_length=10)

    def test_to_matrix(self):
        with self.subTest(msg="With seed"):
            with self.assertWarns(UserWarning):
                matrix = self.ext.to_matrix(seed=GF2([]))
            np.testing.assert_array_equal(matrix, np.eye(10))
        with self.subTest(msg="Without seed"):
            matrix = self.ext.to_matrix(seed=None)
            np.testing.assert_array_equal(matrix, np.eye(10))

    def test_extract_random_input(self):
        for _ in range(SIZE_SAMPLE):
            ext_input = GF2.Random(10, seed=RNG)
            ext_seed = None
            np.testing.assert_array_equal(
                self.ext.extract(ext_input, ext_seed), ext_input
            )


class TestCorrectnessInput2Output1(unittest.TestCase):
    def setUp(self):
        self.ext = ModifiedToeplitzHashing(input_length=2, output_length=1)

    def test_input_00(self):
        extractor_input = GF2([0, 0])
        with self.subTest(msg="seed 0"):
            seed = GF2([0])
            expected_result = GF2([0])
            obtained_result = self.ext.extract(extractor_input, seed)
            np.testing.assert_array_equal(expected_result, obtained_result)
        with self.subTest(msg="seed 1"):
            seed = GF2([1])
            expected_result = GF2([0])
            obtained_result = self.ext.extract(extractor_input, seed)
            np.testing.assert_array_equal(expected_result, obtained_result)

    def test_input_01(self):
        extractor_input = GF2([0, 1])
        with self.subTest(msg="seed 0"):
            seed = GF2([0])
            expected_result = GF2([1])
            obtained_result = self.ext.extract(extractor_input, seed)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(msg="seed 1"):
            seed = GF2([1])
            expected_result = GF2([1])
            obtained_result = self.ext.extract(extractor_input, seed)
            np.testing.assert_array_equal(expected_result, obtained_result)

    def test_input_10(self):
        extractor_input = GF2([1, 0])
        with self.subTest(msg="seed 0"):
            seed = GF2([0])
            expected_result = GF2([0])
            obtained_result = self.ext.extract(extractor_input, seed)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(msg="seed 1"):
            seed = GF2([1])
            expected_result = GF2([1])
            obtained_result = self.ext.extract(extractor_input, seed)
            np.testing.assert_array_equal(expected_result, obtained_result)

    def test_input_11(self):
        extractor_input = GF2([1, 1])
        with self.subTest(msg="seed 0"):
            seed = GF2([0])
            expected_result = GF2([1])
            obtained_result = self.ext.extract(extractor_input, seed)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(msg="seed 1"):
            seed = GF2([1])
            expected_result = GF2([0])
            obtained_result = self.ext.extract(extractor_input, seed)
            np.testing.assert_array_equal(expected_result, obtained_result)


class TestCorrectnessInput9Output4(unittest.TestCase):
    def setUp(self):
        self.ext = ModifiedToeplitzHashing(input_length=9, output_length=4)

    def test_to_matrix(self):
        seed = GF2([1, 0, 0, 0, 0, 0, 0, 0])

        with self.subTest(seed_mode="default"):
            # fmt: off
            expected_matrix = GF2([
                [1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(seed=seed)
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="col-first"):
            # fmt: off
            expected_matrix = GF2([
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(seed=seed, seed_mode="col-first")
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="row-first"):
            # fmt: off
            expected_matrix = GF2([
                [0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(seed=seed, seed_mode="row-first")
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (1)"):
            seed_order = np.array([1, 0, 2, 3, 4, 5, 6, 7])
            # fmt: off
            expected_matrix = GF2([
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (2)"):
            seed_order = np.array([1, 2, 0, 3, 4, 5, 6, 7])
            # fmt: off
            expected_matrix = GF2([
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (3)"):
            seed_order = np.array([1, 2, 3, 4, 0, 5, 6, 7])
            # fmt: off
            expected_matrix = GF2([
                [0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

    def test_extract(self):
        ext_input = GF2([1, 0, 0, 1, 0, 1, 1, 1, 0])
        seed = GF2([0, 1, 1, 0, 1, 1, 1, 1])

        with self.subTest(seed_mode="default"):
            matrix = self.ext.to_matrix(seed=seed)
            expected_result = GF2([0, 1, 1, 0])
            obtained_result = self.ext.extract(ext_input, seed)
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="col-first"):
            matrix = self.ext.to_matrix(seed=seed, seed_mode="col-first")
            expected_result = GF2([0, 1, 1, 0])
            obtained_result = self.ext.extract(ext_input, seed, seed_mode="col-first")
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="row-first"):
            matrix = self.ext.to_matrix(seed=seed, seed_mode="row-first")
            expected_result = GF2([1, 1, 0, 0])
            obtained_result = self.ext.extract(ext_input, seed, seed_mode="row-first")
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="custom (random)"):
            seed_order = RNG.permutation(np.arange(seed.size))
            matrix = self.ext.to_matrix(seed, seed_mode="custom", seed_order=seed_order)
            obtained_result = self.ext.extract(
                ext_input,
                seed,
                seed_mode="custom",
                seed_order=seed_order,
            )
            np.testing.assert_array_equal(matrix @ ext_input, obtained_result)


class TestCorrectnessInput11Output8(unittest.TestCase):
    def setUp(self):
        self.ext = ModifiedToeplitzHashing(input_length=11, output_length=8)

    def test_to_matrix(self):
        seed = GF2([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

        with self.subTest(seed_mode="default"):
            # fmt: off
            expected_matrix = GF2([
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(seed=seed)
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="col-first"):
            # fmt: off
            expected_matrix = GF2([
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(seed=seed, seed_mode="col-first")
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="row-first"):
            # fmt: off
            expected_matrix = GF2([
                [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(seed=seed, seed_mode="row-first")
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (1)"):
            # fmt: off
            seed_order = np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])
            expected_matrix = GF2([
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (2)"):
            # fmt: off
            seed_order = np.array([1, 2, 0, 3, 4, 5, 6, 7, 8, 9])
            expected_matrix = GF2([
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (3)"):
            # fmt: off
            seed_order = np.array([1, 2, 3, 4, 0, 5, 6, 7, 8, 9])
            expected_matrix = GF2([
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

    def test_extract(self):
        ext_input = GF2([1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1])
        seed = GF2([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

        with self.subTest(seed_mode="default"):
            matrix = self.ext.to_matrix(seed=seed)
            expected_result = GF2([1, 1, 1, 1, 1, 1, 1, 1])
            obtained_result = self.ext.extract(ext_input, seed)
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="col-first"):
            matrix = self.ext.to_matrix(seed=seed, seed_mode="col-first")
            expected_result = GF2([0, 0, 0, 0, 1, 0, 0, 0])
            obtained_result = self.ext.extract(ext_input, seed, seed_mode="col-first")
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="row-first"):
            matrix = self.ext.to_matrix(seed=seed, seed_mode="row-first")
            expected_result = GF2([0, 1, 1, 0, 0, 1, 0, 1])
            obtained_result = self.ext.extract(ext_input, seed, seed_mode="row-first")
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="custom (random)"):
            seed_order = RNG.permutation(np.arange(seed.size))
            matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            obtained_result = self.ext.extract(
                ext_input,
                seed,
                seed_mode="custom",
                seed_order=seed_order,
            )
            np.testing.assert_array_equal(matrix @ ext_input, obtained_result)


class TestCorrectnessInput11Output3(unittest.TestCase):
    def setUp(self):
        self.ext = ModifiedToeplitzHashing(input_length=11, output_length=3)

    def test_to_matrix(self):
        seed = GF2([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

        with self.subTest(seed_mode="default"):
            # fmt: off
            expected_matrix = GF2([
                [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
                [1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
                [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(seed=seed)
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="col-first"):
            # fmt: off
            expected_matrix = GF2([
                [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(seed=seed, seed_mode="col-first")
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="row-first"):
            # fmt: off
            expected_matrix = GF2([
                [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
                [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(seed=seed, seed_mode="row-first")
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (1)"):
            # fmt: off
            seed_order = np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])
            expected_matrix = GF2([
                [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (2)"):
            # fmt: off
            seed_order = np.array([1, 2, 0, 3, 4, 5, 6, 7, 8, 9])
            expected_matrix = GF2([
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
                [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (3)"):
            # fmt: off
            seed_order = np.array([1, 2, 3, 4, 0, 5, 6, 7, 8, 9])
            expected_matrix = GF2([
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
                [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

    def test_extract(self):
        ext_input = GF2([1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1])
        seed = GF2([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

        with self.subTest(seed_mode="default"):
            matrix = self.ext.to_matrix(seed=seed)
            expected_result = GF2([0, 1, 0])
            obtained_result = self.ext.extract(ext_input, seed)
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="col-first"):
            matrix = self.ext.to_matrix(seed=seed, seed_mode="col-first")
            expected_result = GF2([0, 1, 1])
            obtained_result = self.ext.extract(ext_input, seed, seed_mode="col-first")
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="row-first"):
            matrix = self.ext.to_matrix(seed=seed, seed_mode="row-first")
            expected_result = GF2([0, 0, 0])
            obtained_result = self.ext.extract(ext_input, seed, seed_mode="row-first")
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="custom (random)"):
            seed_order = RNG.permutation(np.arange(seed.size))
            matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            obtained_result = self.ext.extract(
                ext_input,
                seed,
                seed_mode="custom",
                seed_order=seed_order,
            )
            np.testing.assert_array_equal(matrix @ ext_input, obtained_result)


class TestCorrectnessInput11Output10(unittest.TestCase):
    def setUp(self):
        self.ext = ModifiedToeplitzHashing(input_length=11, output_length=10)

    def test_to_matrix(self):
        seed = GF2([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

        with self.subTest(seed_mode="default"):
            # fmt: off
            expected_matrix = GF2([
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(seed=seed)
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="col-first"):
            # fmt: off
            expected_matrix = GF2([
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(seed=seed, seed_mode="col-first")
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (1)"):
            seed_order = np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])
            # fmt: off
            expected_matrix = GF2([
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (2)"):
            seed_order = np.array([1, 2, 0, 3, 4, 5, 6, 7, 8, 9])
            # fmt: off
            expected_matrix = GF2([
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (3)"):
            seed_order = np.array([1, 2, 3, 4, 0, 5, 6, 7, 8, 9])
            # fmt: off
            expected_matrix = GF2([
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ])
            # fmt: on
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

    def test_extract(self):
        ext_input = GF2([1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0])
        seed = GF2([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

        with self.subTest(seed_mode="default"):
            matrix = self.ext.to_matrix(seed=seed)
            expected_result = GF2([1, 1, 0, 1, 1, 0, 1, 0, 1, 1])
            obtained_result = self.ext.extract(ext_input, seed)
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="col-first"):
            matrix = self.ext.to_matrix(seed=seed, seed_mode="col-first")
            expected_result = GF2([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
            obtained_result = self.ext.extract(ext_input, seed, seed_mode="col-first")
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="row-first"):
            matrix = self.ext.to_matrix(seed=seed, seed_mode="row-first")
            expected_result = GF2([1, 1, 0, 1, 1, 0, 1, 0, 1, 1])
            obtained_result = self.ext.extract(ext_input, seed, seed_mode="row-first")
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="custom (random)"):
            seed_order = RNG.permutation(np.arange(seed.size))
            matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            obtained_result = self.ext.extract(
                ext_input,
                seed,
                seed_mode="custom",
                seed_order=seed_order,
            )
            np.testing.assert_array_equal(matrix @ ext_input, obtained_result)


class TestCorrectnessInput11Output1(unittest.TestCase):
    def setUp(self):
        self.ext = ModifiedToeplitzHashing(input_length=11, output_length=1)

    def test_to_matrix(self):
        seed = GF2([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

        with self.subTest(seed_mode="default"):
            expected_matrix = GF2([[0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]])
            obtained_matrix = self.ext.to_matrix(seed=seed)
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="col-first"):
            expected_matrix = GF2([[0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]])
            obtained_matrix = self.ext.to_matrix(seed=seed, seed_mode="col-first")
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="row-first"):
            expected_matrix = GF2([[1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1]])
            obtained_matrix = self.ext.to_matrix(seed=seed, seed_mode="row-first")
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (1)"):
            seed_order = np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])
            expected_matrix = GF2([[1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1]])
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (2)"):
            seed_order = np.array([1, 2, 0, 3, 4, 5, 6, 7, 8, 9])
            expected_matrix = GF2([[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

        with self.subTest(seed_mode="custom (3)"):
            seed_order = np.array([1, 2, 3, 4, 0, 5, 6, 7, 8, 9])
            expected_matrix = GF2([[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
            obtained_matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            np.testing.assert_array_equal(expected_matrix, obtained_matrix)

    def test_extract(self):
        ext_input = GF2([0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0])
        seed = GF2([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

        with self.subTest(seed_mode="default"):
            matrix = self.ext.to_matrix(seed=seed)
            expected_result = GF2([0])
            obtained_result = self.ext.extract(ext_input, seed)
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="col-first"):
            matrix = self.ext.to_matrix(seed=seed, seed_mode="col-first")
            expected_result = GF2([0])
            obtained_result = self.ext.extract(ext_input, seed, seed_mode="col-first")
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="row-first"):
            matrix = self.ext.to_matrix(seed=seed, seed_mode="row-first")
            expected_result = GF2([0])
            obtained_result = self.ext.extract(ext_input, seed, seed_mode="row-first")
            np.testing.assert_array_equal(matrix @ ext_input, expected_result)
            np.testing.assert_array_equal(expected_result, obtained_result)

        with self.subTest(seed_mode="custom (random)"):
            seed_order = RNG.permutation(np.arange(seed.size))
            matrix = self.ext.to_matrix(
                seed=seed, seed_mode="custom", seed_order=seed_order
            )
            obtained_result = self.ext.extract(
                ext_input,
                seed,
                seed_mode="custom",
                seed_order=seed_order,
            )
            np.testing.assert_array_equal(matrix @ ext_input, obtained_result)
