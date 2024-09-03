import unittest

import numpy as np
from galois import GF2

from randextract.utilities.converter import (
    big_endian_binary_vector_to_int,
    binary_array_to_integer,
    entropy_extraction_ratio_and_output_length_conversion,
    error_bound_and_error_bound_per_bit_conversion,
    integer_to_binary_array,
)


class TestBigEndianBinaryVectorToInt(unittest.TestCase):
    def test_big_endian_binary_vector_to_int_wrong_vector_type(self):
        with self.assertRaises(AssertionError):
            big_endian_binary_vector_to_int(vector="grape")

    def test_big_endian_binary_vector_to_int_wrong_vector_dimensions(self):
        with self.assertRaises(AssertionError):
            big_endian_binary_vector_to_int(vector=np.array([[1, 0, 0, 1, 1, 0, 1]]))

    def test_big_endian_binary_vector_to_int_empty_vector(self):
        with self.assertRaises(AssertionError):
            big_endian_binary_vector_to_int(vector=np.array([]))

    def test_big_endian_binary_vector_to_int_small(self):
        self.assertEqual(big_endian_binary_vector_to_int(vector=np.array([1])), 1)

    def test_big_endian_binary_vector_to_int_all_0(self):
        self.assertEqual(
            big_endian_binary_vector_to_int(vector=np.array([0, 0, 0, 0, 0, 0, 0])), 0
        )

    def test_big_endian_binary_vector_to_int_all_1(self):
        self.assertEqual(
            big_endian_binary_vector_to_int(vector=np.array([1, 1, 1, 1, 1, 1, 1])), 127
        )

    def test_big_endian_binary_vector_to_int_random(self):
        self.assertEqual(
            big_endian_binary_vector_to_int(vector=np.array([1, 0, 0, 1, 1, 0, 1])), 77
        )


class TestErrorBoundAndErrorBoundPerBitConversion(unittest.TestCase):
    def test_output_length_wrong_type(self):
        with self.assertRaises(AssertionError):
            error_bound_and_error_bound_per_bit_conversion(
                output_length="tomato", error_bound=1e-10
            )

    def test_error_bound_wrong_type(self):
        with self.assertRaises(AssertionError):
            error_bound_and_error_bound_per_bit_conversion(
                output_length=1000, error_bound="litchi"
            )

    def test_error_bound_per_bit_wrong_type(self):
        with self.assertRaises(AssertionError):
            error_bound_and_error_bound_per_bit_conversion(
                output_length=1000, error_bound_per_bit="banana"
            )

    def test_error_bound_and_error_bound_per_bit_none(self):
        with self.assertRaises(AssertionError):
            error_bound_and_error_bound_per_bit_conversion(
                output_length=1000, error_bound=None, error_bound_per_bit=None
            )

    def test_error_bound_too_small(self):
        with self.assertRaises(AssertionError):
            error_bound_and_error_bound_per_bit_conversion(
                output_length=1000, error_bound=0.0
            )

    def test_error_bound_too_big(self):
        with self.assertRaises(AssertionError):
            error_bound_and_error_bound_per_bit_conversion(
                output_length=1000, error_bound=1.0
            )

    def test_error_bound_per_bit_too_small(self):
        with self.assertRaises(AssertionError):
            error_bound_and_error_bound_per_bit_conversion(
                output_length=1000, error_bound_per_bit=0.0
            )

    def test_error_bound_per_bit_too_big(self):
        with self.assertRaises(AssertionError):
            error_bound_and_error_bound_per_bit_conversion(
                output_length=1000, error_bound_per_bit=1.0
            )

    def test_indirect_error_bound_too_big(self):
        with self.assertRaises(AssertionError):
            error_bound_and_error_bound_per_bit_conversion(
                output_length=1000, error_bound_per_bit=0.001
            )

    def test_correct_error_bound(self):
        (
            error_bound,
            error_bound_per_bit,
        ) = error_bound_and_error_bound_per_bit_conversion(
            output_length=1000, error_bound=0.5
        )
        self.assertEqual(error_bound, 0.5)
        self.assertEqual(error_bound_per_bit, 0.0005)

    def test_correct_error_bound_per_bit(self):
        (
            error_bound,
            error_bound_per_bit,
        ) = error_bound_and_error_bound_per_bit_conversion(
            output_length=1000, error_bound_per_bit=0.0005
        )
        self.assertEqual(error_bound, 0.5)
        self.assertEqual(error_bound_per_bit, 0.0005)


class TestEntropyExtractionRatioAndOutputLengthConversion(unittest.TestCase):
    def test_relative_source_entropy_wrong_type(self):
        with self.assertRaises(AssertionError):
            entropy_extraction_ratio_and_output_length_conversion(
                relative_source_entropy="peach",
                input_length=2**20,
                entropy_extraction_ratio=0.01,
            )

    def test_input_length_wrong_type(self):
        with self.assertRaises(AssertionError):
            entropy_extraction_ratio_and_output_length_conversion(
                relative_source_entropy=0.9,
                input_length=0.2,
                entropy_extraction_ratio=0.01,
            )

    def test_entropy_extraction_ratio_wrong_type(self):
        with self.assertRaises(AssertionError):
            entropy_extraction_ratio_and_output_length_conversion(
                relative_source_entropy=0.9,
                input_length=2**20,
                entropy_extraction_ratio="chili",
            )

    def test_output_length_wrong_type(self):
        with self.assertRaises(AssertionError):
            entropy_extraction_ratio_and_output_length_conversion(
                relative_source_entropy=0.9, input_length=2**20, output_length=0.01
            )

    def test_entropy_extraction_ratio_and_output_length_none(self):
        with self.assertRaises(AssertionError):
            entropy_extraction_ratio_and_output_length_conversion(
                relative_source_entropy=0.9,
                input_length=2**20,
                entropy_extraction_ratio=None,
                output_length=None,
            )

    def test_entropy_extraction_ratio_too_small(self):
        with self.assertRaises(AssertionError):
            entropy_extraction_ratio_and_output_length_conversion(
                relative_source_entropy=0.9,
                input_length=2**20,
                entropy_extraction_ratio=0.0,
            )

    def test_entropy_extraction_ratio_too_big(self):
        with self.assertRaises(AssertionError):
            entropy_extraction_ratio_and_output_length_conversion(
                relative_source_entropy=0.9,
                input_length=2**20,
                entropy_extraction_ratio=1.0,
            )

    def test_output_length_too_small(self):
        with self.assertRaises(AssertionError):
            entropy_extraction_ratio_and_output_length_conversion(
                relative_source_entropy=0.9, input_length=2**20, output_length=0
            )

    def test_output_length_too_big(self):
        with self.assertRaises(AssertionError):
            entropy_extraction_ratio_and_output_length_conversion(
                relative_source_entropy=0.9, input_length=2**20, output_length=2**20
            )

    def test_correct_entropy_extraction_ratio(self):
        (
            entropy_extraction_ratio,
            output_length,
        ) = entropy_extraction_ratio_and_output_length_conversion(
            relative_source_entropy=0.9, input_length=1000, entropy_extraction_ratio=0.1
        )
        self.assertEqual(entropy_extraction_ratio, 0.1)
        self.assertEqual(output_length, 90)

    def test_correct_output_length(self):
        (
            entropy_extraction_ratio,
            output_length,
        ) = entropy_extraction_ratio_and_output_length_conversion(
            relative_source_entropy=0.9, input_length=1000, output_length=90
        )
        self.assertEqual(output_length, 90)
        self.assertEqual(entropy_extraction_ratio, 0.1)


class TestBinaryArrayToInteger(unittest.TestCase):
    def test_wrong_type(self):
        with self.subTest(msg="ndarray"):
            with self.assertRaises(AssertionError):
                binary_array_to_integer(np.array([1, 1, 0, 0]))

        with self.subTest(msg="list"):
            with self.assertRaises(AssertionError):
                binary_array_to_integer([1, 1, 0, 0])

        with self.subTest(msg="tuple"):
            with self.assertRaises(AssertionError):
                binary_array_to_integer((1, 1, 0, 0))

    def test_wrong_dimension(self):
        with self.assertRaises(AssertionError):
            binary_array_to_integer(GF2.Ones((3, 3)))

    def test_right_output(self):
        with self.subTest(msg="0"):
            self.assertEqual(binary_array_to_integer(GF2(0)), 0)

        with self.subTest(msg="1"):
            self.assertEqual(binary_array_to_integer(GF2(1)), 1)

        with self.subTest(msg="000...0"):
            self.assertEqual(binary_array_to_integer(GF2.Zeros(100)), 0)

        with self.subTest(msg="111...1"):
            self.assertEqual(binary_array_to_integer(GF2.Ones(100)), 2**100 - 1)

        with self.subTest(msg="1010011010"):
            self.assertEqual(
                binary_array_to_integer(GF2([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])), 666
            )


class TestIntegerToBinaryArray(unittest.TestCase):
    def test_wrong_num_type(self):
        with self.subTest(msg="str"):
            with self.assertRaises(AssertionError):
                integer_to_binary_array("array")

        with self.subTest(msg="list"):
            with self.assertRaises(AssertionError):
                integer_to_binary_array([1, 2, 3])

        with self.subTest(msg="GF2"):
            with self.assertRaises(AssertionError):
                integer_to_binary_array(GF2(1))

    def test_wrong_pad_type(self):
        with self.subTest(msg="str"):
            with self.assertRaises(AssertionError):
                integer_to_binary_array(5, pad="array")

        with self.subTest(msg="list"):
            with self.assertRaises(AssertionError):
                integer_to_binary_array(5, pad=[1, 2, 3])

        with self.subTest(msg="GF2"):
            with self.assertRaises(AssertionError):
                integer_to_binary_array(5, pad=GF2(1))

    def test_output_no_pad(self):
        with self.subTest(msg="0"):
            np.testing.assert_array_equal(integer_to_binary_array(0), GF2([0]))

        with self.subTest(msg="1"):
            np.testing.assert_array_equal(integer_to_binary_array(1), GF2([1]))

        with self.subTest(msg="2^100-1"):
            np.testing.assert_array_equal(
                integer_to_binary_array(2**100 - 1), GF2.Ones(100)
            )

        with self.subTest(msg="666"):
            np.testing.assert_array_equal(
                integer_to_binary_array(666), GF2([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
            )

    def test_output_with_pad(self):
        with self.subTest(msg="0"):
            np.testing.assert_array_equal(
                integer_to_binary_array(0, pad=100), GF2.Zeros(100)
            )

        with self.subTest(msg="1"):
            expected = GF2.Zeros(100)
            expected[-1] = 1
            np.testing.assert_array_equal(integer_to_binary_array(1, pad=100), expected)

        with self.subTest(msg="666"):
            np.testing.assert_array_equal(
                integer_to_binary_array(666, pad=15),
                GF2([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]),
            )
