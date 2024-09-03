import itertools
import unittest

import numpy as np
from galois import GF, GF2

from randextract.xor_one_bit_extractor import XOROneBitExtractor

from .._types import (
    INTEGERS,
    NOT_INTEGERS,
    NOT_INTEGERS_WITHOUT_NONE,
    NOT_REALS,
    NOT_STRINGS,
)

RNG = np.random.default_rng()
SIZE_SAMPLE = 100


class TestInitialization(unittest.TestCase):
    def test_wrong_input_length(self):
        for not_integer in NOT_INTEGERS:
            with self.subTest(type=type(not_integer)):
                with self.assertRaises(TypeError):
                    XOROneBitExtractor(input_length=not_integer, seed_length=3)

    def test_wrong_number_indices(self):
        for not_integer in NOT_INTEGERS:
            with self.subTest(type=type(not_integer)):
                with self.assertRaises(TypeError):
                    XOROneBitExtractor(
                        input_length=not_integer, number_indices=not_integer
                    )

    def test_wrong_seed_length(self):
        for not_integer in NOT_INTEGERS_WITHOUT_NONE:
            with self.subTest(type=type(not_integer)):
                with self.assertRaises(TypeError):
                    XOROneBitExtractor(input_length=10, seed_length=not_integer)

    def test_good_input_length(self):
        for integer in INTEGERS:
            integer += type(integer)(10)
            with self.subTest(type=type(integer)):
                XOROneBitExtractor(input_length=integer, number_indices=1)

    def test_input_length_too_large(self):
        with self.assertRaises(ValueError):
            XOROneBitExtractor(input_length=10**20, number_indices=10)

    def test_good_number_indices(self):
        for integer in INTEGERS:
            integer += type(integer)(5)
            with self.subTest(type=type(integer)):
                XOROneBitExtractor(input_length=10, number_indices=integer)

    def test_good_seed_length(self):
        for integer in INTEGERS:
            integer = type(integer)(8)
            with self.subTest(type=type(integer)):
                XOROneBitExtractor(input_length=10, seed_length=integer)

    def test_missing_seed(self):
        with self.assertRaises(TypeError):
            XOROneBitExtractor(input_length=10)

    def test_both_seed_length_and_indices(self):
        with self.assertWarns(UserWarning):
            XOROneBitExtractor(input_length=10, number_indices=3, seed_length=12)

    def test_smallest_case(self):
        with self.assertWarns(UserWarning):
            XOROneBitExtractor(input_length=1, number_indices=0)
        with self.assertWarns(UserWarning):
            XOROneBitExtractor(input_length=1, seed_length=0)
        XOROneBitExtractor(input_length=1)

    def test_min_seed_length(self):
        with self.subTest(n=32):
            XOROneBitExtractor(input_length=32, seed_length=5)

        with self.subTest(n=33):
            XOROneBitExtractor(input_length=33, seed_length=6)

    def test_seed_length_too_small(self):
        with self.assertRaises(ValueError):
            XOROneBitExtractor(input_length=32, seed_length=-1)

    def test_max_seed_length(self):
        with self.subTest(n=32):
            XOROneBitExtractor(input_length=32, seed_length=5 * 32)

        with self.subTest(n=33):
            XOROneBitExtractor(input_length=33, seed_length=6 * 33)

    def test_seed_length_too_large(self):
        with self.subTest(n=32):
            with self.assertRaises(ValueError):
                XOROneBitExtractor(input_length=32, seed_length=5 * 32 + 1)

        with self.subTest(n=33):
            with self.assertRaises(ValueError):
                XOROneBitExtractor(input_length=33, seed_length=6 * 33 + 1)

    def test_zero_input(self):
        with self.assertRaises(ValueError):
            XOROneBitExtractor(input_length=0, number_indices=0)

    def test_bits_per_index(self):
        with self.subTest(input_length=1):
            ext = XOROneBitExtractor(input_length=1)
            self.assertEqual(ext._bits_per_index, 1)

        with self.subTest(input_length=2):
            ext = XOROneBitExtractor(input_length=2, number_indices=1)
            self.assertEqual(ext._bits_per_index, 1)

        with self.subTest(input_length=32):
            ext = XOROneBitExtractor(32, number_indices=1)
            self.assertEqual(ext._bits_per_index, 5)

        with self.subTest(input_length=33):
            ext = XOROneBitExtractor(33, number_indices=1)
            self.assertEqual(ext._bits_per_index, 6)

        with self.subTest(input_length=100):
            ext = XOROneBitExtractor(100, number_indices=1)
            self.assertEqual(ext._bits_per_index, 7)

        with self.subTest(input_length=1000):
            ext = XOROneBitExtractor(1000, number_indices=1)
            self.assertEqual(ext._bits_per_index, 10)

    def test_seed_length_to_number_indices(self):
        with self.subTest(input_length=1):
            with self.assertWarns(UserWarning):
                ext = XOROneBitExtractor(input_length=1, seed_length=0)
            self.assertEqual(ext.number_indices, 0)

            with self.assertWarns(UserWarning):
                ext = XOROneBitExtractor(input_length=1, seed_length=1)
            self.assertEqual(ext.number_indices, 0)

        with self.subTest(input_length=2):
            ext = XOROneBitExtractor(input_length=2, seed_length=1)
            self.assertEqual(ext.number_indices, 1)

            ext = XOROneBitExtractor(input_length=2, seed_length=2)
            self.assertEqual(ext.number_indices, 2)

        with self.subTest(input_length=32):
            with self.assertWarns(UserWarning):
                ext = XOROneBitExtractor(input_length=32, seed_length=7)
            self.assertEqual(ext.number_indices, 1)

            ext = XOROneBitExtractor(input_length=32, seed_length=10)
            self.assertEqual(ext.number_indices, 2)

        with self.subTest(input_length=100):
            with self.assertWarns(UserWarning):
                ext = XOROneBitExtractor(input_length=100, seed_length=100)
            self.assertEqual(ext.number_indices, 14)

            ext = XOROneBitExtractor(input_length=100, seed_length=175)
            self.assertEqual(ext.number_indices, 25)

    def test_number_indices_to_seed_length(self):
        with self.subTest(input_length=1):
            ext = XOROneBitExtractor(input_length=1)
            self.assertEqual(ext.seed_length, 0)

            with self.assertWarns(UserWarning):
                ext = XOROneBitExtractor(input_length=1, number_indices=0)
            self.assertEqual(ext.seed_length, 0)

            with self.assertWarns(UserWarning):
                ext = XOROneBitExtractor(input_length=1, number_indices=1)
            self.assertEqual(ext.seed_length, 0)

        with self.subTest(input_length=2):
            ext = XOROneBitExtractor(input_length=2, number_indices=0)
            self.assertEqual(ext.seed_length, 0)
            ext = XOROneBitExtractor(input_length=2, number_indices=1)
            self.assertEqual(ext.seed_length, 1)
            ext = XOROneBitExtractor(input_length=2, number_indices=2)
            self.assertEqual(ext.seed_length, 2)

        with self.subTest(input_length=32):
            ext = XOROneBitExtractor(input_length=32, number_indices=1)
            self.assertEqual(ext.seed_length, 5)
            ext = XOROneBitExtractor(input_length=32, number_indices=2)
            self.assertEqual(ext.seed_length, 10)

        with self.subTest(input_length=100):
            ext = XOROneBitExtractor(input_length=100, number_indices=1)
            self.assertEqual(ext.seed_length, 7)
            ext = XOROneBitExtractor(input_length=100, number_indices=25)
            self.assertEqual(ext.seed_length, 175)

        with self.subTest(input_length=1000):
            ext = XOROneBitExtractor(input_length=1000, number_indices=1)
            self.assertEqual(ext.seed_length, 10)
            ext = XOROneBitExtractor(input_length=1000, number_indices=10)
            self.assertEqual(ext.seed_length, 100)


class TestCalculateLength(unittest.TestCase):
    def test_wrong_extractor_type_type(self):
        for not_string in NOT_STRINGS:
            with self.assertRaises(TypeError):
                XOROneBitExtractor.calculate_length(
                    extractor_type=not_string,
                    input_length=10,
                    relative_source_entropy=0.9,
                    error_bound=1e-6,
                )

    def test_wrong_extractor_type_value(self):
        for ext_type in ["strong", "weak"]:
            with self.assertRaises(ValueError):
                XOROneBitExtractor.calculate_length(
                    extractor_type=ext_type,
                    input_length=10,
                    relative_source_entropy=0.9,
                    error_bound=1e-6,
                )

    def test_wrong_input_length_type(self):
        for not_integer in NOT_INTEGERS:
            with self.assertRaises(TypeError):
                XOROneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=not_integer,
                    relative_source_entropy=0.9,
                    error_bound=1e-6,
                )

    def test_wrong_input_length_value(self):
        for input_length in [-1, 0]:
            with self.assertRaises(ValueError):
                XOROneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=input_length,
                    relative_source_entropy=0.9,
                    error_bound=1e-6,
                )

    def test_wrong_relative_source_entropy_type(self):
        for not_real in NOT_REALS:
            with self.assertRaises(TypeError):
                XOROneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=10,
                    relative_source_entropy=not_real,
                    error_bound=1e-6,
                )

    def test_wrong_relative_source_entropy_value(self):
        for k in [-0.5, 0, 1.5]:
            with self.assertRaises(ValueError):
                XOROneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=10,
                    relative_source_entropy=k,
                    error_bound=1e-6,
                )

    def test_wrong_error_bound_type(self):
        for not_real in NOT_REALS:
            with self.assertRaises(TypeError):
                XOROneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=10,
                    relative_source_entropy=0.9,
                    error_bound=not_real,
                )

    def test_wrong_error_bound_value(self):
        for eps in [-0.5, 0, 1.5]:
            with self.assertRaises(ValueError):
                XOROneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=10,
                    relative_source_entropy=0.9,
                    error_bound=eps,
                )

    def test_wrong_return_mode_type(self):
        for not_string in NOT_STRINGS:
            with self.assertRaises(TypeError):
                XOROneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=10,
                    relative_source_entropy=0.9,
                    error_bound=1e-3,
                    return_mode=not_string,
                )

    def test_wrong_return_mode_value(self):
        for mode in ["default", "index", "bitstring"]:
            with self.assertRaises(ValueError):
                XOROneBitExtractor.calculate_length(
                    extractor_type="quantum",
                    input_length=10,
                    relative_source_entropy=0.9,
                    error_bound=1e-3,
                    return_mode=mode,
                )

    def test_uniform_input(self):
        for _ in range(SIZE_SAMPLE):
            input_length = RNG.integers(1, 10**8)
            eps = RNG.random()
            for ext_type in ["classical", "quantum"]:
                with self.assertWarns(UserWarning):
                    result = XOROneBitExtractor.calculate_length(
                        ext_type,
                        input_length,
                        relative_source_entropy=1,
                        error_bound=eps,
                    )
                self.assertEqual(result, 0)
                with self.assertWarns(UserWarning):
                    result = XOROneBitExtractor.calculate_length(
                        ext_type,
                        input_length,
                        relative_source_entropy=1,
                        error_bound=eps,
                        return_mode="binary",
                    )
                self.assertEqual(result, 0)

    def test_input_length_100(self):
        n = 100
        with self.subTest(extractor_type="classical"):
            ext_type = "classical"

            for eps in [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
                with self.assertRaises(ValueError):
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    )

            with self.subTest(error_bound=1e-4):
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    49,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    29,
                )
            with self.subTest(error_bound=1e-3):
                eps = 1e-3
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    97,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    53,
                )
            with self.subTest(error_bound=1e-2):
                eps = 1e-2
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    49,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    29,
                )
            with self.subTest(error_bound=1e-1):
                eps = 1e-1
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    39,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    21,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    13,
                )
            with self.subTest(error_bound=1):
                eps = 1
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    7,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    4,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    3,
                )

            with self.subTest(extractor_type="quantum"):
                ext_type = "quantum"

                for eps in [1e-8, 1e-6, 1e-5, 1e-2, 1e-1]:
                    with self.assertRaises(ValueError):
                        XOROneBitExtractor.calculate_length(
                            ext_type,
                            n,
                            relative_source_entropy=0.5,
                            error_bound=eps,
                            return_mode="indices",
                        )

                with self.subTest(error_bound=1e-2):
                    eps = 1e-2
                    for k in [0.7, 0.9]:
                        with self.assertRaises(ValueError):
                            XOROneBitExtractor.calculate_length(
                                ext_type,
                                n,
                                relative_source_entropy=k,
                                error_bound=eps,
                                return_mode="indices",
                            )
                with self.subTest(error_bound=1e-1):
                    eps = 1e-1
                    self.assertEqual(
                        XOROneBitExtractor.calculate_length(
                            ext_type,
                            n,
                            relative_source_entropy=0.7,
                            error_bound=eps,
                            return_mode="indices",
                        ),
                        83,
                    )
                    self.assertEqual(
                        XOROneBitExtractor.calculate_length(
                            ext_type,
                            n,
                            relative_source_entropy=0.9,
                            error_bound=eps,
                            return_mode="indices",
                        ),
                        46,
                    )
                with self.subTest(error_bound=1):
                    eps = 1
                    self.assertEqual(
                        XOROneBitExtractor.calculate_length(
                            ext_type,
                            n,
                            relative_source_entropy=0.5,
                            error_bound=eps,
                            return_mode="indices",
                        ),
                        29,
                    )
                    self.assertEqual(
                        XOROneBitExtractor.calculate_length(
                            ext_type,
                            n,
                            relative_source_entropy=0.7,
                            error_bound=eps,
                            return_mode="indices",
                        ),
                        16,
                    )
                    self.assertEqual(
                        XOROneBitExtractor.calculate_length(
                            ext_type,
                            n,
                            relative_source_entropy=0.9,
                            error_bound=eps,
                            return_mode="indices",
                        ),
                        10,
                    )

    def test_input_length_1000(self):
        n = 1000
        with self.subTest(extractor_type="classical"):
            ext_type = "classical"
            with self.subTest(error_bound=1e-4):
                eps = 1e-4
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    102,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    58,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    35,
                )
            with self.subTest(error_bound=1e-3):
                eps = 1e-3
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    76,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    44,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    27,
                )
            with self.subTest(error_bound=1e-2):
                eps = 1e-2
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    52,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    30,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    18,
                )
            with self.subTest(error_bound=1e-1):
                eps = 1e-1
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    29,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    17,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    10,
                )
            with self.subTest(error_bound=1):
                eps = 1
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    7,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    4,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    3,
                )

        with self.subTest(extractor_type="quantum"):
            ext_type = "quantum"
            with self.subTest(error_bound=1e-4):
                eps = 1e-4
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    252,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    139,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    84,
                )
            with self.subTest(error_bound=1e-3):
                eps = 1e-3
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    184,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    103,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    62,
                )
            with self.subTest(error_bound=1e-2):
                eps = 1e-2
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    124,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    70,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    43,
                )
            with self.subTest(error_bound=1e-1):
                eps = 1e-1
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    70,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    41,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    25,
                )
            with self.subTest(error_bound=1):
                eps = 1
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.5,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    23,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.7,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    14,
                )
                self.assertEqual(
                    XOROneBitExtractor.calculate_length(
                        ext_type,
                        n,
                        relative_source_entropy=0.9,
                        error_bound=eps,
                        return_mode="indices",
                    ),
                    8,
                )

    def test_return_mode_binary(self):
        self.assertEqual(
            XOROneBitExtractor.calculate_length(
                "classical",
                100,
                relative_source_entropy=0.9,
                error_bound=1e-2,
                return_mode="binary",
            ),
            29 * 7,
        )
        self.assertEqual(
            XOROneBitExtractor.calculate_length(
                "quantum",
                100,
                relative_source_entropy=0.7,
                error_bound=1e-1,
                return_mode="binary",
            ),
            83 * 7,
        )
        self.assertEqual(
            XOROneBitExtractor.calculate_length(
                "classical",
                1000,
                relative_source_entropy=0.5,
                error_bound=1e-3,
                return_mode="binary",
            ),
            76 * 10,
        )
        self.assertEqual(
            XOROneBitExtractor.calculate_length(
                "quantum",
                1000,
                relative_source_entropy=0.7,
                error_bound=1e-4,
                return_mode="binary",
            ),
            139 * 10,
        )


class TestConvertSeed(unittest.TestCase):
    def test_seed_length_0(self):
        ext = XOROneBitExtractor(input_length=10, seed_length=0)
        seed = GF2([])
        np.testing.assert_array_equal(ext._convert_seed(seed), np.array([]))

    def test_n_1(self):
        with self.subTest(seed_length=0):
            with self.assertWarns(UserWarning):
                ext = XOROneBitExtractor(input_length=1, seed_length=0)
            seed = GF2([])
            np.testing.assert_array_equal(ext._convert_seed(seed), np.array([]))

        with self.subTest(seed_length=1):
            with self.assertWarns(UserWarning):
                ext = XOROneBitExtractor(input_length=1, seed_length=1)

            with self.subTest(seed=[0]):
                seed = GF2([0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0]))

            with self.subTest(seed=[1]):
                seed = GF2([1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0]))

    def test_n_2(self):
        with self.subTest(seed_length=1):
            ext = XOROneBitExtractor(input_length=2, seed_length=1)

            with self.subTest(seed=[0]):
                seed = GF2([0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0]))

            with self.subTest(seed=[1]):
                seed = GF2([1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([1]))

        with self.subTest(seed_length=2):
            ext = XOROneBitExtractor(input_length=2, seed_length=2)

            with self.subTest(seed=[0, 0]):
                seed = GF2([0, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0, 0]))

            with self.subTest(seed=[0, 1]):
                seed = GF2([0, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0, 1]))

            with self.subTest(seed=[1, 0]):
                seed = GF2([1, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([1, 0]))

            with self.subTest(seed=[1, 1]):
                seed = GF2([1, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([1, 1]))

    def test_n_3(self):
        # This test is interesting because it is the first time we can get indices outside
        # the valid interval [0, input_length) if not handle properly
        with self.subTest(number_indices=2):
            ext = XOROneBitExtractor(input_length=3, number_indices=2)

            with self.subTest(seed=[0, 0, 0, 0]):
                seed = GF2([0, 0, 0, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0, 0]))

            with self.subTest(seed=[0, 0, 0, 1]):
                seed = GF2([0, 0, 0, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0, 1]))

            with self.subTest(seed=[0, 0, 1, 0]):
                seed = GF2([0, 0, 1, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0, 2]))

            with self.subTest(seed=[0, 0, 1, 1]):
                seed = GF2([0, 0, 1, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0, 0]))

            with self.subTest(seed=[0, 1, 0, 0]):
                seed = GF2([0, 1, 0, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([1, 0]))

            with self.subTest(seed=[0, 1, 0, 1]):
                seed = GF2([0, 1, 0, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([1, 1]))

            with self.subTest(seed=[0, 1, 1, 0]):
                seed = GF2([0, 1, 1, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([1, 2]))

            with self.subTest(seed=[0, 1, 1, 1]):
                seed = GF2([0, 1, 1, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([1, 0]))

            with self.subTest(seed=[1, 0, 0, 0]):
                seed = GF2([1, 0, 0, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([2, 0]))

            with self.subTest(seed=[1, 0, 0, 1]):
                seed = GF2([1, 0, 0, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([2, 1]))

            with self.subTest(seed=[1, 0, 1, 0]):
                seed = GF2([1, 0, 1, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([2, 2]))

            with self.subTest(seed=[1, 0, 1, 1]):
                seed = GF2([1, 0, 1, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([2, 0]))

            with self.subTest(seed=[1, 1, 0, 0]):
                seed = GF2([1, 1, 0, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0, 0]))

            with self.subTest(seed=[1, 1, 0, 1]):
                seed = GF2([1, 1, 0, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0, 1]))

            with self.subTest(seed=[1, 1, 1, 0]):
                seed = GF2([1, 1, 1, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0, 2]))

            with self.subTest(seed=[1, 1, 1, 1]):
                seed = GF2([1, 1, 1, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0, 0]))

    def test_n_32(self):
        with self.subTest(seed_length=5):
            ext = XOROneBitExtractor(input_length=32, seed_length=5)

            with self.subTest(seed=GF2.Zeros(5)):
                seed = GF2.Zeros(5)
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0]))

            with self.subTest(seed=[0, 0, 1, 1, 0]):
                seed = GF2([0, 0, 1, 1, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([6]))

            with self.subTest(seed=[1, 1, 0, 0, 1]):
                seed = GF2([1, 1, 0, 0, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([25]))

            with self.subTest(seed=GF2.Ones(5)):
                seed = GF2.Ones(5)
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([31]))

        with self.subTest(seed_length=7):
            with self.assertWarns(UserWarning):
                ext = XOROneBitExtractor(input_length=32, seed_length=7)

            with self.subTest(seed=GF2.Zeros(7)):
                seed = GF2.Zeros(7)
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0]))

            with self.subTest(seed=[1, 0, 0, 1, 1, 1, 0]):
                seed = GF2([1, 0, 0, 1, 1, 1, 0])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([19]))

            with self.subTest(seed=[1, 1, 0, 1, 0, 1, 1]):
                seed = GF2([1, 1, 0, 1, 0, 1, 1])
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([26]))

            with self.subTest(seed=GF2.Ones(7)):
                seed = GF2.Ones(7)
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([31]))

        with self.subTest(seed_length=10):
            ext = XOROneBitExtractor(input_length=32, seed_length=10)

            with self.subTest(seed=GF2.Zeros(10)):
                seed = GF2.Zeros(10)
                np.testing.assert_array_equal(ext._convert_seed(seed), np.array([0, 0]))

            with self.subTest(seed=[0, 0, 0, 1, 1, 0, 1, 1, 0, 0]):
                seed = GF2([0, 0, 0, 1, 1, 0, 1, 1, 0, 0])
                np.testing.assert_array_equal(
                    ext._convert_seed(seed), np.array([3, 12])
                )

            with self.subTest(seed=[1, 1, 1, 0, 1, 0, 0, 1, 0, 0]):
                seed = GF2([1, 1, 1, 0, 1, 0, 0, 1, 0, 0])
                np.testing.assert_array_equal(
                    ext._convert_seed(seed), np.array([29, 4])
                )

            with self.subTest(seed=GF2.Ones(10)):
                seed = GF2.Ones(10)
                np.testing.assert_array_equal(
                    ext._convert_seed(seed), np.array([31, 31])
                )


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.ext = XOROneBitExtractor(input_length=3, number_indices=2)

    def test_wrong_extractor_input_type(self):
        for wrong_input in [[1, 0, 1], (1, 0, 1), 4, "input"]:
            with self.assertRaises(TypeError):
                self.ext.extract(wrong_input, seed=GF2([1, 0]))

    def test_wrong_extractor_input_value(self):
        for wrong_input in [
            np.array([2, 3, 1]),  # Wrong range
            GF(5)([1, 2, 4]),  # Wrong range
            GF2([[1, 0, 1]]),  # Wrong dimension
            GF2([0, 1, 0, 0]),  # Wrong size
        ]:
            with self.assertRaises(ValueError):
                self.ext.extract(wrong_input, seed=GF2([1, 0]))

    def test_wrong_seed_type(self):
        for wrong_seed in [[1, 0], (1, 0), 2, "seed"]:
            with self.assertRaises(TypeError):
                self.ext.extract(GF2([1, 1, 0]), seed=wrong_seed, seed_mode="binary")
                self.ext.extract(GF2([1, 1, 0]), seed=wrong_seed, seed_mode="indices")

    def test_wrong_seed_value(self):
        with self.subTest(seed_mode="binary"):
            for wrong_seed in [
                np.array([2, 3]),  # Wrong range
                GF(5)([1, 2]),  # Wrong range
                GF2([[1, 0]]),  # Wrong dimension
                GF2([0, 1, 0]),  # Wrong size
            ]:
                with self.assertRaises(ValueError):
                    self.ext.extract(
                        GF2([1, 1, 0]), seed=wrong_seed, seed_mode="binary"
                    )

        with self.subTest(seed_mode="indices"):
            for wrong_seed in [
                np.array([2, 3]),  # Wrong range
                GF(5)([2, 3]),  # Wrong range
                GF2([[1, 0]]),  # Wrong dimension
                GF2([0, 1, 0]),  # Wrong size
            ]:
                with self.assertRaises(ValueError):
                    self.ext.extract(
                        GF2([1, 1, 0]), seed=wrong_seed, seed_mode="indices"
                    )

    def test_wrong_seed_mode_type(self):
        for not_string in NOT_STRINGS:
            with self.assertRaises(TypeError):
                self.ext.extract(GF2([1, 1, 0]), seed=GF2([1, 0]), seed_mode=not_string)

    def test_wrong_seed_mode_value(self):
        for seed_mode in ["bitstring", "index"]:
            with self.assertRaises(ValueError):
                self.ext.extract(GF2([1, 1, 1]), seed=GF2([1, 0]), seed_mode=seed_mode)

    def test_extractor_input(self):
        seed = RNG.integers(low=0, high=3, size=2)

        with self.subTest(msg="binary numpy array"):
            for i in range(SIZE_SAMPLE):
                extractor_input = RNG.integers(low=0, high=2, size=3)
                self.ext.extract(extractor_input, seed, seed_mode="indices")

        with self.subTest(msg="GF2 array"):
            for i in range(SIZE_SAMPLE):
                extractor_input = GF2.Random(3, seed=RNG)
                self.ext.extract(extractor_input, seed, seed_mode="indices")

    def test_seed_mode_indices(self):
        extractor_input = GF2.Random(3, seed=RNG)
        for t in np.typecodes["AllInteger"]:
            with self.subTest(msg=f"{np.dtype(t)}"):
                seed = RNG.integers(low=0, high=3, size=2, dtype=t)
                self.ext.extract(extractor_input, seed, seed_mode="indices")

    def test_seed_mode_binary(self):
        extractor_input = GF2.Random(3, seed=RNG)
        for t in np.typecodes["AllInteger"]:
            with self.subTest(msg=f"{np.dtype(t)}"):
                seed = RNG.integers(low=0, high=2, size=4, dtype=t)
                self.ext.extract(extractor_input, seed, seed_mode="binary")


class TestCorrectnessSmallestCase(unittest.TestCase):
    def setUp(self):
        self.ext = XOROneBitExtractor(input_length=1)

    def test_attributes(self):
        self.assertEqual(self.ext.input_length, 1)
        self.assertEqual(self.ext.number_indices, 0)
        self.assertEqual(self.ext.seed_length, 0)

    def test_input_0(self):
        self.assertEqual(self.ext.extract(GF2([0]), None), 0)
        with self.assertWarns(UserWarning):
            self.assertEqual(self.ext.extract(GF2([0]), np.array([0])), 0)

    def test_input_1(self):
        self.assertEqual(self.ext.extract(GF2([1]), None), 1)
        with self.assertWarns(UserWarning):
            self.assertEqual(self.ext.extract(GF2([0]), np.array([0])), 0)


class TestCorrectness(unittest.TestCase):
    def test_docs_example(self):
        ext = XOROneBitExtractor(input_length=20, number_indices=7)
        self.assertEqual(ext.input_length, 20)
        self.assertEqual(ext.number_indices, 7)
        self.assertEqual(ext.seed_length, 7 * 5)
        self.assertEqual(ext.output_length, 1)
        extractor_input = GF2(
            [0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        )
        seed = np.array([0, 10, 13, 15, 17, 14, 5])
        expected_output = GF2([0])
        np.testing.assert_array_equal(
            ext.extract(extractor_input, seed, seed_mode="indices"), expected_output
        )

    def test_smallest_case(self):
        with self.assertWarns(UserWarning):
            ext = XOROneBitExtractor(input_length=1, number_indices=1)
        self.assertEqual(ext.input_length, 1)
        self.assertEqual(ext.number_indices, 0)
        self.assertEqual(ext.seed_length, 0)
        self.assertEqual(ext.output_length, 1)

        # fmt: off
        np.testing.assert_array_equal(
            ext.extract(GF2([0]), None),
            GF2([0])
        )
        np.testing.assert_array_equal(
            ext.extract(GF2([1]), None),
            GF2([1])
        )
        # fmt: on

    def test_number_indices_one(self):
        ext = XOROneBitExtractor(input_length=5, number_indices=1)
        self.assertEqual(ext.input_length, 5)
        self.assertEqual(ext.number_indices, 1)
        self.assertEqual(ext.seed_length, 3)
        self.assertEqual(ext.output_length, 1)

        extractor_input = GF2([0, 1, 0, 0, 1])

        # fmt: off
        np.testing.assert_array_equal(
            ext.extract(extractor_input, np.array([0]), seed_mode="indices"),
            GF2([0])
        )
        np.testing.assert_array_equal(
            ext.extract(extractor_input, np.array([1]), seed_mode="indices"),
            GF2([1])
        )
        np.testing.assert_array_equal(
            ext.extract(extractor_input, np.array([2]), seed_mode="indices"),
            GF2([0])
        )
        np.testing.assert_array_equal(
            ext.extract(extractor_input, np.array([3]), seed_mode="indices"),
            GF2([0])
        )
        np.testing.assert_array_equal(
            ext.extract(extractor_input, np.array([4]), seed_mode="indices"),
            GF2([1])
        )
        # fmt: on

    def test_seed_length_two(self):
        ext = XOROneBitExtractor(input_length=2, number_indices=2)
        self.assertEqual(ext.input_length, 2)
        self.assertEqual(ext.number_indices, 2)
        self.assertEqual(ext.seed_length, 2 * 1)
        self.assertEqual(ext.output_length, 1)

        extractor_input = GF2([0, 1])

        # fmt: off
        np.testing.assert_array_equal(
            ext.extract(extractor_input, np.array([0, 0])),
            GF2([0])
        )
        np.testing.assert_array_equal(
            ext.extract(extractor_input, np.array([0, 1])),
            GF2([1])
        )
        np.testing.assert_array_equal(
            ext.extract(extractor_input, np.array([1, 0])),
            GF2([1])
        )
        np.testing.assert_array_equal(
            ext.extract(extractor_input, np.array([1, 1])),
            GF2([0])
        )
        # fmt: on

    def test_small_case(self):
        ext = XOROneBitExtractor(input_length=30, number_indices=5)
        # fmt: off
        extractor_input = GF2(
            [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1]
        )
        # fmt: on
        self.assertEqual(ext.input_length, 30)
        self.assertEqual(ext.number_indices, 5)
        self.assertEqual(ext.seed_length, 5 * 5)
        self.assertEqual(ext.output_length, 1)

        with self.subTest(msg="seed = [11, 14, 10, 14, 11]"):
            seed = np.array([11, 14, 10, 14, 11])
            # fmt: off
            np.testing.assert_array_equal(
                ext.extract(extractor_input, seed, seed_mode="indices"),
                GF2([0])
            )
            # fmt: on

        with self.subTest(msg="seed = [26,  2, 22,  2, 29]"):
            seed = np.array([26, 2, 22, 2, 29])
            # fmt: off
            np.testing.assert_array_equal(
                ext.extract(extractor_input, seed, seed_mode="indices"),
                GF2([0])
            )
            # fmt: on

        with self.subTest(msg="seed = [28, 21,  0,  3, 10]"):
            seed = np.array([28, 21, 0, 3, 10])
            # fmt: off
            np.testing.assert_array_equal(
                ext.extract(extractor_input, seed, seed_mode="indices"),
                GF2([1])
            )
            # fmt: on

        with self.subTest(msg="seed = [18, 17, 11, 20, 22]"):
            seed = np.array([18, 17, 11, 20, 22])
            # fmt: off
            np.testing.assert_array_equal(
                ext.extract(extractor_input, seed, seed_mode="indices"),
                GF2([0])
            )
            # fmt: on

    def test_all_zeros(self):
        ext = XOROneBitExtractor(input_length=100, number_indices=10)
        self.assertEqual(ext.input_length, 100)
        self.assertEqual(ext.number_indices, 10)
        self.assertEqual(ext.seed_length, 10 * 7)
        self.assertEqual(ext.output_length, 1)
        extractor_input = GF2.Zeros(100)

        for i in range(SIZE_SAMPLE):
            np.testing.assert_array_equal(
                ext.extract(
                    extractor_input,
                    RNG.integers(low=0, high=100, size=10),
                    seed_mode="indices",
                ),
                GF2([0]),
            )

    def test_all_ones(self):
        extractor_input = GF2.Ones(100)

        with self.subTest(msg="seed_length even"):
            ext = XOROneBitExtractor(input_length=100, number_indices=10)
            self.assertEqual(ext.input_length, 100)
            self.assertEqual(ext.number_indices, 10)
            self.assertEqual(ext.seed_length, 10 * 7)
            self.assertEqual(ext.output_length, 1)

            for i in range(SIZE_SAMPLE):
                np.testing.assert_array_equal(
                    ext.extract(
                        extractor_input,
                        RNG.integers(low=0, high=100, size=10),
                        seed_mode="indices",
                    ),
                    GF2([0]),
                )

        with self.subTest(msg="seed_length odd"):
            ext = XOROneBitExtractor(input_length=100, number_indices=11)
            self.assertEqual(ext.input_length, 100)
            self.assertEqual(ext.number_indices, 11)
            self.assertEqual(ext.seed_length, 11 * 7)
            self.assertEqual(ext.output_length, 1)

            for i in range(SIZE_SAMPLE):
                np.testing.assert_array_equal(
                    ext.extract(
                        extractor_input,
                        RNG.integers(low=0, high=100, size=11),
                        seed_mode="indices",
                    ),
                    GF2([1]),
                )
