import unittest

import numpy as np

from randextract.utilities.binary_entropy import binary_entropy, inverse_binary_entropy

from ..._types import NOT_REALS, REALS


class TestBinaryEntropy(unittest.TestCase):
    def test_wrong_types(self):
        for not_real in NOT_REALS:
            if isinstance(not_real, np.ndarray):
                continue
            with self.assertRaises(TypeError):
                binary_entropy(not_real)

    def test_good_types(self):
        for real in REALS:
            binary_entropy(real)

    def test_negative(self):
        with self.assertRaises(ValueError):
            binary_entropy(-0.01)
        with self.assertRaises(ValueError):
            binary_entropy(np.array([-0.01]))

    def test_too_big(self):
        with self.assertRaises(ValueError):
            binary_entropy(1.51)
        with self.assertRaises(ValueError):
            binary_entropy(np.array([1.51]))

    def test_zero(self):
        self.assertEqual(binary_entropy(0.0), 0.0)
        np.testing.assert_array_equal(binary_entropy(np.array([0.0])), np.array([0.0]))

    def test_one_half(self):
        self.assertEqual(binary_entropy(0.5), 1.0)
        np.testing.assert_array_equal(binary_entropy(np.array([0.5])), np.array([1.0]))

    def test_one(self):
        self.assertEqual(binary_entropy(1.0), 0.0)
        np.testing.assert_array_equal(binary_entropy(np.array([1.0])), np.array([0.0]))

    def test_symmetry(self):
        probs = [
            0.0,
            0.14285714,
            0.28571429,
            0.42857143,
            0.57142857,
            0.71428571,
            0.85714286,
            1.0,
        ]
        for p in probs:
            self.assertAlmostEqual(binary_entropy(p), binary_entropy(1 - p))
        np.testing.assert_array_almost_equal(
            binary_entropy(np.array(probs)), binary_entropy(1 - np.array(probs))
        )


class TestInverseBinaryEntropy(unittest.TestCase):
    def test_wrong_types(self):
        for not_real in NOT_REALS:
            with self.assertRaises(TypeError):
                inverse_binary_entropy(not_real)

    def test_good_types(self):
        for real in REALS:
            inverse_binary_entropy(real)

    def test_negative(self):
        with self.assertRaises(ValueError):
            inverse_binary_entropy(-0.01)

    def test_too_big(self):
        with self.assertRaises(ValueError):
            inverse_binary_entropy(1.51)

    def test_zero(self):
        self.assertEqual(inverse_binary_entropy(0.0), 0.0)

    def test_one(self):
        self.assertEqual(inverse_binary_entropy(1.0), 0.5)

    def test_correctness(self):
        probs = [
            0.0,
            0.14285714,
            0.28571429,
            0.42857143,
            0.57142857,
            0.71428571,
            0.85714286,
            1.0,
        ]
        h = [binary_entropy(_) for _ in probs]
        for i in range(len(h)):
            if probs[i] < 0.5:
                self.assertAlmostEqual(probs[i], inverse_binary_entropy(h[i]))
            else:
                self.assertAlmostEqual(1 - probs[i], inverse_binary_entropy(h[i]))
