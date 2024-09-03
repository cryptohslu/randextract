import unittest

from randextract import RandomnessExtractor

from .._types import NOT_STRINGS


class TestCreate(unittest.TestCase):
    def test_valid_extractor_type(self):
        with self.subTest(extractor_type="toeplitz"):
            RandomnessExtractor.create(
                extractor_type="toeplitz", input_length=1000, output_length=100
            )
        with self.subTest(extractor_type="modified-toeplitz"):
            RandomnessExtractor.create(
                extractor_type="modified_toeplitz", input_length=1000, output_length=100
            )
        with self.subTest(extractor_type="xor"):
            RandomnessExtractor.create(
                extractor_type="xor", input_length=1000, seed_length=20
            )
        with self.subTest(extractor_type="polynomial"):
            RandomnessExtractor.create(
                extractor_type="polynomial", input_length=1000, seed_length=10
            )
        with self.subTest(extractor_type="trevisan"):
            RandomnessExtractor.create(
                extractor_type="trevisan",
                input_length=1000,
                output_length=100,
                weak_design_type="finite_field",
                one_bit_extractor_type="xor",
                one_bit_extractor_seed_length=13,
            )

    def test_extractor_type_wrong_type(self):
        for ext_type in NOT_STRINGS:
            with self.assertRaises(TypeError):
                RandomnessExtractor.create(extractor_type=ext_type)

    def test_extractor_type_wrong_value(self):
        for ext_type in ["good", "fast"]:
            with self.assertRaises(ValueError):
                RandomnessExtractor.create(extractor_type=ext_type)
