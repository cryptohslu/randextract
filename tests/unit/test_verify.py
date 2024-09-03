import math
import unittest
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from galois import GF, GF2

from randextract._verify import (
    verify_array,
    verify_kwargs,
    verify_number_in_interval,
    verify_number_is_non_negative,
    verify_number_is_positive,
    verify_number_type,
    verify_type,
)

from .._types import (
    INTEGERS,
    NOT_INTEGERS,
    NOT_REALS,
    NUMPY_INTEGRAL_TYPES,
    NUMPY_REAL_TYPES,
    REALS,
)

RNG = np.random.default_rng()


class TestVerifyType(unittest.TestCase):
    def test_single_valid_type(self):
        verify_type("test", str)
        verify_type(1, int)
        verify_type([1], list)

    def test_single_wrong_type(self):
        with self.subTest(msg="test"):
            with self.assertRaises(TypeError):
                verify_type("test", int)
        with self.subTest(msg=1):
            with self.assertRaises(TypeError):
                verify_type(1, str)
        with self.subTest(msg=[1]):
            with self.assertRaises(TypeError):
                verify_type([1], tuple)

    def test_list_valid_types(self):
        valid_types = [str, Path, np.ndarray]

        with self.subTest(msg="file"):
            verify_type("file", valid_types)

        with self.subTest(msg="Path.cwd()"):
            verify_type(Path.cwd(), valid_types)

        with self.subTest(msg="np.ndarray"):
            verify_type(np.array([1]), valid_types)

    def test_list_wrong_types(self):
        valid_types = [str, Path, np.ndarray]

        with self.subTest(msg="int"):
            with self.assertRaises(TypeError):
                verify_type(10, valid_types)

        with self.subTest(msg="print"):
            with self.assertRaises(TypeError):
                verify_type(print, valid_types)

        with self.subTest(msg="list"):
            with self.assertRaises(TypeError):
                verify_type([np.array([1])], valid_types)


class TestVerifyNumberType(unittest.TestCase):
    def test_valid_types_integral(self):
        for integer in INTEGERS:
            with self.subTest(type=f"{type(integer)}"):
                self.assertIsNone(verify_number_type(integer, Integral))

    def test_wrong_types_integral(self):
        for not_integer in NOT_INTEGERS:
            with self.subTest(type=f"{type(not_integer)}"):
                with self.assertRaises(TypeError):
                    verify_number_type(not_integer, Integral)

    def test_valid_types_real(self):
        for real in REALS:
            with self.subTest(type=f"{type(real)}"):
                self.assertIsNone(verify_number_type(real, Real))

    def test_wrong_types_real(self):
        for not_real in NOT_REALS:
            with self.subTest(type=f"{type(not_real)}"):
                with self.assertRaises(TypeError):
                    verify_number_type(not_real, Real)


class TestVerifyNumberIsPositive(unittest.TestCase):
    def test_positive_number(self):
        for number in REALS:
            with self.subTest(type=f"{type(number)}"):
                self.assertIsNone(verify_number_is_positive(number))

    def test_wrong_positive_numbers(self):
        test_cases = [0, -1, -1e1, False]
        for number in test_cases:
            with self.subTest(type=f"{type(number)}"):
                with self.assertRaises(ValueError):
                    verify_number_is_positive(number)


class TestVerifyNumberIsNonNegative(unittest.TestCase):
    def test_non_negative_number(self):
        for number in REALS + [0, 0.0, False]:
            with self.subTest(type=f"{type(number)}"):
                self.assertIsNone(verify_number_is_non_negative(number))

    def test_wrong_non_negative_numbers(self):
        test_cases = [-1, -1e1, -1e-9]
        for number in test_cases:
            with self.subTest(type=f"{type(number)}"):
                with self.assertRaises(ValueError):
                    verify_number_is_non_negative(number)


class TestVerifyNumberInInterval(unittest.TestCase):
    def test_open_interval(self):
        with self.subTest(msg="0 < 1.5 < 2"):
            self.assertIsNone(verify_number_in_interval(1.5, 0, 2, "open"))

        with self.subTest(msg="-10 < 1.5 < 10"):
            self.assertIsNone(verify_number_in_interval(1.5, -10, 2, "open"))

        with self.subTest(msg="-inf < 1.5 < inf"):
            self.assertIsNone(
                verify_number_in_interval(1.5, -math.inf, math.inf, "open")
            )

        with self.subTest(msg="0 < 1 < 1"):
            with self.assertRaises(ValueError):
                verify_number_in_interval(1, 0, 1, "open")

        with self.subTest(msg="0 < 0 < 1"):
            with self.assertRaises(ValueError):
                verify_number_in_interval(0, 0, 1, "open")

    def test_closed_interval(self):
        with self.subTest(msg="0 <= 1.5 <= 2"):
            self.assertIsNone(verify_number_in_interval(1.5, 0, 2, "closed"))

        with self.subTest(msg="-10 <= 1.5 <= 10"):
            self.assertIsNone(verify_number_in_interval(1.5, -10, 2, "closed"))

        with self.subTest(msg="-inf <= 1.5 <= inf"):
            self.assertIsNone(
                verify_number_in_interval(1.5, -math.inf, math.inf, "closed")
            )

        with self.subTest(msg="0 <= 1 <= 1"):
            self.assertIsNone(verify_number_in_interval(1, 0, 1, "closed"))

        with self.subTest(msg="0 <= 0 <= 1"):
            self.assertIsNone(verify_number_in_interval(0, 0, 1, "closed"))

        with self.subTest(msg="-pi <= 3.5 <= pi"):
            with self.assertRaises(ValueError):
                verify_number_in_interval(3.5, -math.pi, math.pi, "closed")

    def test_left_open_interval(self):
        with self.subTest(msg="0 < 1.5 <= 2"):
            self.assertIsNone(verify_number_in_interval(1.5, 0, 2, "left-open"))

        with self.subTest(msg="-10 < 1.5 <= 10"):
            self.assertIsNone(verify_number_in_interval(1.5, -10, 2, "left-open"))

        with self.subTest(msg="-inf < 1.5 <= inf"):
            self.assertIsNone(
                verify_number_in_interval(1.5, -math.inf, math.inf, "left-open")
            )

        with self.subTest(msg="0 < 1 <= 1"):
            self.assertIsNone(verify_number_in_interval(1, 0, 1, "left-open"))

        with self.subTest(msg="0 < 0 <= 1"):
            with self.assertRaises(ValueError):
                verify_number_in_interval(0, 0, 1, "left-open")

        with self.subTest(msg="-pi < -pi <= pi"):
            with self.assertRaises(ValueError):
                verify_number_in_interval(-math.pi, -math.pi, math.pi, "left-open")

        with self.subTest(msg="-pi < pi <= pi"):
            self.assertIsNone(
                verify_number_in_interval(math.pi, -math.pi, math.pi, "left-open")
            )

    def test_right_open_interval(self):
        with self.subTest(msg="0 <= 1.5 < 2"):
            self.assertIsNone(verify_number_in_interval(1.5, 0, 2, "right-open"))

        with self.subTest(msg="-10 <= 1.5 < 10"):
            self.assertIsNone(verify_number_in_interval(1.5, -10, 2, "right-open"))

        with self.subTest(msg="-inf <= 1.5 < inf"):
            self.assertIsNone(
                verify_number_in_interval(1.5, -math.inf, math.inf, "right-open")
            )

        with self.subTest(msg="0 <= 1 < 1"):
            with self.assertRaises(ValueError):
                verify_number_in_interval(1, 0, 1, "right-open")

        with self.subTest(msg="0 <= 0 < 1"):
            self.assertIsNone(verify_number_in_interval(0, 0, 1, "right-open"))

        with self.subTest(msg="-pi <= -pi < pi"):
            self.assertIsNone(
                verify_number_in_interval(-math.pi, -math.pi, math.pi, "right-open")
            )

        with self.subTest(msg="-pi <= pi < pi"):
            with self.assertRaises(ValueError):
                verify_number_in_interval(math.pi, -math.pi, math.pi, "right-open")


class TestVerifyArray(unittest.TestCase):
    def test_is_array(self):
        self.assertIsNone(verify_array(np.array([1])))
        self.assertIsNone(verify_array(GF2([1])))

    def test_is_not_array(self):
        test_cases = ["array", None, print, [0, 1], (0, 1), {"0": 1}]
        for test in test_cases:
            with self.assertRaises(TypeError):
                verify_array(test)

    def test_array_correct_dim(self):
        with self.subTest(msg="1-dim"):
            self.assertIsNone(verify_array(GF2([0, 1, 1]), valid_dim=1))

        with self.subTest(msg="2-dim"):
            self.assertIsNone(verify_array(GF2([[0, 1, 1]]), valid_dim=2))

        with self.subTest(msg="3-dim"):
            self.assertIsNone(verify_array(GF2([[[0, 1, 1]]]), valid_dim=3))

    def test_array_wrong_dim(self):
        with self.subTest(msg="1-dim"):
            with self.assertRaises(ValueError):
                verify_array(GF2([[0, 1, 1]]), valid_dim=1)

        with self.subTest(msg="2-dim"):
            with self.assertRaises(ValueError):
                verify_array(GF2([0, 1, 1]), valid_dim=2)

    def test_array_correct_type(self):
        with self.subTest(msg="valid_type=Integral"):
            for t in NUMPY_INTEGRAL_TYPES:
                array = RNG.integers(low=0, high=127, size=10, dtype=t)
                self.assertIsNone(verify_array(array, valid_type=Integral))

        with self.subTest(msg="valid_type=Real"):
            for t in NUMPY_REAL_TYPES + NUMPY_INTEGRAL_TYPES:
                # We use astype() because random.Generator.random() only supports np.float32 and np.float64
                array = RNG.random(10).astype(t)
                self.assertIsNone(verify_array(array, valid_type=Real))

    def test_array_wrong_type(self):
        for t in NUMPY_REAL_TYPES:
            with self.assertRaises(TypeError):
                # We use astype() because random.Generator.random() only supports np.float32 and np.float64
                array = RNG.random(10).astype(t)
                verify_array(array, valid_type=Integral)

    def test_wrong_range(self):
        array = GF2.Random(10, seed=RNG)

        with self.subTest(msg="low > high"):
            with self.assertRaises(AssertionError):
                verify_array(array, valid_range=[1, 0, "closed"])

        with self.subTest(msg="Missing interval type"):
            with self.assertRaises(AssertionError):
                verify_array(array, valid_range=[0, 1])

        with self.subTest(msg="Wrong interval type"):
            with self.assertRaises(AssertionError):
                verify_array(array, valid_range=[0, 1, "right-closed"])

        with self.subTest(msg="Values outside given range"):
            for interval_type in ["open", "closed", "left-open", "right-open"]:
                with self.assertRaises(ValueError):
                    verify_array(
                        np.array(array) - 20, valid_range=[0, 1, interval_type]
                    )
                with self.assertRaises(ValueError):
                    verify_array(
                        np.array(array) + 20, valid_range=[0, 1, interval_type]
                    )


class TestVerifyArrayPracticalCases(unittest.TestCase):
    def test_good_binary_1_dim_array_given_size(self):
        size = 10**3
        with self.subTest(msg="numpy array"):
            array = RNG.integers(low=0, high=2, size=size)
            self.assertIsNone(
                verify_array(
                    array,
                    valid_dim=1,
                    valid_size=size,
                    valid_type=Integral,
                    valid_range=[0, 1, "closed"],
                )
            )

        with self.subTest(msg="GF2 array"):
            array = GF2.Random(size, seed=RNG)
            self.assertIsNone(
                verify_array(
                    array,
                    valid_dim=1,
                    valid_size=size,
                    valid_type=Integral,
                    valid_range=[0, 1, "closed"],
                )
            )

    def test_wrong_binary_1_dim_array_given_size(self):
        size = 10**3
        array = GF2.Random(size, seed=RNG).view(np.ndarray)
        array[-1] = 2
        with self.assertRaises(ValueError):
            verify_array(
                array,
                valid_dim=1,
                valid_size=size,
                valid_type=Integral,
                valid_range=[0, 1, "closed"],
            )

    def test_good_2_dim_array_in_range(self):
        shape = (10**3, 10**3)
        array = RNG.integers(low=-10, high=10, endpoint=False, size=shape)
        self.assertIsNone(
            verify_array(
                array,
                valid_dim=2,
                valid_type=Integral,
                valid_range=[-10, 10, "right-open"],
            )
        )

    def test_good_1_dim_array_finite_field(self):
        size = 10**3
        array = GF(2**16).Random(size, seed=RNG)
        self.assertIsNone(
            verify_array(
                array,
                valid_dim=1,
                valid_type=Integral,
                valid_range=[0, 2**16, "right-open"],
            )
        )


class TestVerifyKwargs(unittest.TestCase):
    def test_empty_kwargs(self):
        verify_kwargs({}, [])

    def test_correct_only_required_args(self):
        verify_kwargs({"r1": 1, "r2": 2}, ["r1", "r2"])

    def test_wrong_only_required_args(self):
        with self.assertRaises(ValueError):
            verify_kwargs({"r1": 1}, ["r1", "r2"])

    def test_correct_only_optional_args(self):
        with self.subTest("all optional args passed"):
            verify_kwargs({"o1": 1, "o2": 2}, optional_args=["o1", "o2"])
        with self.subTest("subset optional args passed"):
            verify_kwargs({"o1": 1}, optional_args=["o1", "o2"])
        with self.subTest("none optional args passed"):
            verify_kwargs({}, optional_args=["o1", "o2"])

    def test_correct_only_subset_optional_args(self):
        verify_kwargs({"o1": 1}, optional_args=["o1", "o2"])

    def test_wrong_only_optional_args(self):
        with self.assertRaises(ValueError):
            verify_kwargs({"o1": 1, "o2": 2}, optional_args=["o1"])

    def test_correct_both_required_and_optional_args(self):
        with self.subTest("all optional args passed"):
            verify_kwargs(
                {"r1": 1, "r2": 2, "o1": 1, "o2": 2}, ["r1", "r2"], ["o1", "o2"]
            )
        with self.subTest("subset optional args passed"):
            verify_kwargs({"r1": 1, "r2": 2, "o1": 1}, ["r1", "r2"], ["o1", "o2"])
        with self.subTest("none optional args passed"):
            verify_kwargs({"r1": 1, "r2": 2}, ["r1", "r2"], ["o1", "o2"])

    def test_wrong_both_required_and_optional_args(self):
        with self.assertRaises(ValueError):
            verify_kwargs({"r1": 1, "r2": 2, "o1": 1, "o2": 2}, ["r1", "r2"], ["o1"])
