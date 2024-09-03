import unittest

import numpy as np

from randextract.trevisan.block_design import BlockDesign


class TestInitialization(unittest.TestCase):
    pass


class TestNumberBasicWeakDesigns(unittest.TestCase):
    # This is just to have access to the object methods,
    # actual values are manually changed later.
    def setUp(self):
        self.block_design = BlockDesign(20, 23)

    def test_number_of_basic_weak_designs(self):
        with self.subTest(msg="m=10, t=10"):
            self.block_design._number_of_sets = 10
            self.block_design._size_of_sets = 10
            expected = 2
            actual = self.block_design._compute_number_basic_weak_designs()
            self.assertEqual(expected, actual)

        with self.subTest(msg="m=512, t=59"):
            self.block_design._number_of_sets = 512
            self.block_design._size_of_sets = 59
            expected = 13
            actual = self.block_design._compute_number_basic_weak_designs()
            self.assertEqual(expected, actual)


class TestSmallestBlockDesign(unittest.TestCase):
    def setUp(self):
        self.block_design = BlockDesign(6, 7)

    def test_number_of_basic_weak_designs(self):
        self.assertEqual(self.block_design._number_basic_weak_designs, 2)

    def test_range(self):
        self.assertEqual(self.block_design.range_design, 98)

    def test_weak_design(self):
        self.block_design.compute_design()

        np.testing.assert_array_equal(
            self.block_design.weak_design,
            # fmt: off
            np.array([
                [0, 1, 2, 3, 4, 5, 6],
                [49, 50, 51, 52, 53, 54, 55],
                [56, 57, 58, 59, 60, 61, 62],
                [63, 64, 65, 66, 67, 68, 69],
                [70, 71, 72, 73, 74, 75, 76],
                [77, 78, 79, 80, 81, 82, 83],
            ])
            # fmt: on
        )


class TestThreeBlocks(unittest.TestCase):
    def setUp(self):
        self.block_design = BlockDesign(13, 11)

    def test_number_of_basic_weak_designs(self):
        self.assertEqual(self.block_design._number_basic_weak_designs, 3)

    def test_range(self):
        self.assertEqual(self.block_design.range_design, 363)

    def test_weak_design(self):
        self.block_design.compute_design()
        np.testing.assert_array_equal(
            self.block_design.weak_design,
            # fmt: off
            np.vstack([
                np.arange(11), np.arange(11, 22),  # W0
                np.arange(121, 132),  # W1
                [np.arange(242 + 11 * i, 242 + 11 * (i + 1)) for i in range(10)]  # W3
            ])
            # fmt: on
        )
