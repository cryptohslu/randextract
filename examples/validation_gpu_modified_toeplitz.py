"""
This example tests the high performance GPU implementation from https://github.com/nicoboss/PrivacyAmplification
against randextract using the Validator class with input_method='custom' and a custom_class.
"""

import math
import subprocess
from pathlib import Path

import numpy as np
from galois import GF2
from randextract import ModifiedToeplitzHashing, Validator, ValidatorCustomClassAbs

# How many tests are performed for each fixed input and output length
SAMPLE_SIZE = 10

# Tested input lengths go from 2^10 to 2^INPUT_LENGTH_MAX_POWER (both included)
INPUT_LENGTH_MAX_POWER = 27

# Output lengths must be multiple of 128 and less or equal than 2^INPUT_LENGTH_MAX_POWER
# OUTPUT_LENGTH_SAMPLE different random choices are used to run the tests
OUTPUT_LENGTH_SAMPLE = 5

# (P)RNG used to get the inputs, seeds, output length, etc.
RNG = np.random.default_rng(1337)


class CustomValidator(ValidatorCustomClassAbs):
    def __init__(self, extractor):
        self.ext = extractor
        self.count = 0

    @staticmethod
    def gf2_from_file(filename, mode):
        if mode == "little":
            arr = np.fromfile(filename, dtype=np.uint32)
            return GF2(
                ((arr[:, None] & (0x80000000 >> np.arange(32))) > 0)
                .astype(np.uint8)
                .flatten()
            )
        elif mode == "big":
            return GF2(np.unpackbits(np.fromfile(filename, dtype=np.uint8)))

    @staticmethod
    def gf2_to_file(arr, filename, mode):
        if mode == "big":
            np.packbits(np.array(arr)).tofile(filename)
        elif mode == "little":
            arr_ = np.zeros(2 ** math.ceil(math.log2(arr.size)), dtype=np.uint8)
            arr_[: arr.size] = arr
            np.packbits(np.array(arr_).reshape((-1, 32))).view(">u4").byteswap().tofile(
                filename
            )

    def generate_config_file(self, path):
        path = Path(path)
        config = f"""
factor_exp: {self.ext.input_length.bit_length() - 1}
reuse_seed_amount: 0
vertical_len: {self.ext.output_length}
do_xor_key_rest: true
do_compress: true
reduction_exp: 11
pre_mul_reduction_exp: 5
gpu_device_id_to_use: 0
input_blocks_to_cache: 16
output_blocks_to_cache: 16
show_ampout: -1
show_zeromq_status: false
use_matrix_seed_server: false
toeplitz_seed_path: 'ext_seed.bin'
use_key_server: false
keyfile_path: 'ext_input.bin'
host_ampout_server: false
store_first_ampouts_in_file: 1
verify_ampout: false
"""
        print(config, file=open(path, "w"))

    def get_extractor_inputs(self):
        while self.count < SAMPLE_SIZE:
            ext_input = GF2.Random(self.ext.input_length - 1, seed=RNG)
            # If patch https://github.com/nicoboss/PrivacyAmplification/pull/2 is not applied
            # ext_input[-1] = 0
            self.gf2_to_file(ext_input, "ext_input.bin", mode="little")
            yield ext_input
            self.count += 1
        return

    def get_extractor_seeds(self):
        while self.count < SAMPLE_SIZE:
            ext_seed = GF2.Random(self.ext.seed_length, seed=RNG)
            self.gf2_to_file(ext_seed, "ext_seed.bin", mode="little")
            yield ext_seed
        return

    def get_extractor_output(self, ext_input, ext_seed):
        self.generate_config_file("config.yaml")

        try:
            subprocess.run(
                args=["./PrivacyAmplificationCuda"],
                capture_output=False,
                timeout=0.7,
                stdout=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            pass

        ext_out = self.gf2_from_file("ampout.bin", mode="big")
        return ext_out


def main():
    print("\nThis script tests a high-performance GPU implementation of the modified")
    print(
        "Toeplitz hashing against the class ModifiedToeplitzHashing from randextract."
    )
    print("\nDiferent families of hash functions are used from the set of supported")
    print("input and output lengths by the GPU implementation. The exact lengths used")
    print("are printed by the script.\n")
    for input_length in (2**x + 1 for x in range(10, INPUT_LENGTH_MAX_POWER)):
        n_valid_output_lengths = input_length // 128
        for output_length in RNG.choice(
            [128 * x for x in range(1, n_valid_output_lengths + 1)],
            size=min(n_valid_output_lengths, OUTPUT_LENGTH_SAMPLE),
        ):
            print(f"input_length = {input_length}")
            print(f"output_length = {output_length}")
            ext = ModifiedToeplitzHashing(input_length, output_length)
            val = Validator(ext)
            custom_class = CustomValidator(ext)
            val.add_implementation(
                "GPU-CUDA", input_method="custom", custom_class=custom_class
            )
            val.validate()
            print(val)
            if not val.all_passed:
                return


if __name__ == "__main__":
    main()
