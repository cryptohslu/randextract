"""
This example tests the high performance GPU implementation from https://github.com/nicoboss/PrivacyAmplification
against randextract using the Validator class with input_method='custom' and a custom_class which uses ZeroMQ to
communicate with PrivacyAmplificationCuda, instead of binary files as it is done in
validation_gpu_implementation_toeplitz.py.

The application PrivacyAmplificationCuda is expected to be running already. The following config.yaml file was used
for the example described in :ref:`GPU modified Toeplitz hashing`.

factor_exp: 27
reuse_seed_amount: 0
vertical_len: 50331648
do_xor_key_rest: true
do_compress: true
reduction_exp: 11
pre_mul_reduction_exp: 5
gpu_device_id_to_use: 0
input_blocks_to_cache: 16
output_blocks_to_cache: 16
show_ampout: 8
show_zeromq_status: true
use_matrix_seed_server: true
address_seed_in: 'tcp://127.0.0.1:45555'
use_key_server: true
address_key_in: 'tcp://127.0.0.1:47777'
host_ampout_server: true
address_amp_out: 'tcp://127.0.0.1:48888'
store_first_ampouts_in_file: 0
verify_ampout: false
verify_ampout_threads: 8

If you modify some of the options, remember to modify the script as well.
"""

import datetime
import math
import time

import numpy as np
import zmq
from galois import GF2
from randextract import ModifiedToeplitzHashing, Validator, ValidatorCustomClassAbs

SAMPLE_SIZE = 1000
INPUT_LENGTH = 2**27 + 1
OUTPUT_LENGTH = 50331648

RNG = np.random.default_rng()

# ZeroMQ sockets
ADDRESS_SEED = "tcp://*:45555"
ADDRESS_INPUT = "tcp://*:47777"
ADDRESS_OUTPUT = "tcp://147.88.195.53:48888"


def gf2_to_bytes(arr, mode):
    if mode == "big":
        return np.packbits(np.array(arr)).tobytes()
    elif mode == "little":
        arr_ = np.zeros(2 ** math.ceil(math.log2(arr.size)), dtype=np.uint8)
        arr_[: arr.size] = arr
        return (
            np.packbits(np.array(arr_).reshape((-1, 32)))
            .view(">u4")
            .byteswap()
            .tobytes()
        )


def bytes_to_gf2(bytes_, mode):
    if mode == "little":
        arr = np.frombuffer(bytes_, dtype=np.uint32)
        return GF2(
            ((arr[:, None] & (0x80000000 >> np.arange(32))) > 0)
            .astype(np.uint8)
            .flatten()
        )
    elif mode == "big":
        return GF2(np.unpackbits(np.frombuffer(bytes_, dtype=np.uint8)))


class CustomValidatorGPU(ValidatorCustomClassAbs):
    def __init__(self, extractor):
        self.ext = extractor
        self.count = 0

    def get_extractor_inputs(self):
        with zmq.Context() as ctx:
            with ctx.socket(zmq.PUSH) as s:
                s.set_hwm(1)
                s.bind(ADDRESS_INPUT)
                while self.count < SAMPLE_SIZE:
                    ext_input = GF2.Random(self.ext.input_length, seed=RNG)
                    # If patch https://github.com/nicoboss/PrivacyAmplification/pull/2 is not applied
                    # ext_input[-1] = 0
                    input_bytes = gf2_to_bytes(ext_input, "little")[
                        : math.ceil(INPUT_LENGTH / 32) * 4
                    ]
                    # Flags required by the Cuda implementation to apply the correct
                    # Modified Toeplitz hashing. See SendKeysExample for more details
                    s.send(b"\x01", zmq.SNDMORE)
                    s.send(b"\x01", zmq.SNDMORE)
                    s.send(np.uint32(OUTPUT_LENGTH // 32).tobytes(), zmq.SNDMORE)
                    s.send(input_bytes, 0)
                    yield ext_input
                    self.count += 1
                return

    def get_extractor_seeds(self):
        with zmq.Context() as ctx:
            with ctx.socket(zmq.PUSH) as s:
                s.set_hwm(1)
                s.bind(ADDRESS_SEED)
                while self.count < SAMPLE_SIZE:
                    ext_seed = GF2.Random(self.ext.seed_length, seed=RNG)
                    seed_bytes = gf2_to_bytes(ext_seed, "little")[:INPUT_LENGTH]
                    # Flags required by the Cuda implementation to apply the correct
                    # Modified Toeplitz hashing. See SendKeysExample for more details
                    s.send(b"\x00\x00\x00\x00", zmq.SNDMORE)
                    s.send(seed_bytes, 0)
                    yield ext_seed
                return

    def get_extractor_output(self, ext_input, ext_seed):
        with zmq.Context() as ctx:
            with ctx.socket(zmq.PULL) as s:
                s.set_hwm(1)
                s.connect(ADDRESS_OUTPUT)
                recv_output = s.recv()
                gpu_output = bytes_to_gf2(recv_output, mode="big")
                return gpu_output


def main():
    print("\nThis script tests a high-performance GPU implementation of the modified")
    print(
        "Toeplitz hashing against the class ModifiedToeplitzHashing from randextract."
    )
    print("\nThe extractor is defined by the input and output lengths:\n")
    print(f"input_length = {INPUT_LENGTH}")
    print(f"output_length = {OUTPUT_LENGTH}\n")
    ext = ModifiedToeplitzHashing(INPUT_LENGTH, OUTPUT_LENGTH)
    # TODO: uncomment this when __str__() from RandomnessExtractor is implemented
    # print(ext)

    val = Validator(ext)
    custom_class = CustomValidatorGPU(ext)
    val.add_implementation("GPU-CUDA", input_method="custom", custom_class=custom_class)
    print(val)

    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Starting validation using {SAMPLE_SIZE} random samples...\n")
    start = time.time()
    val.validate()
    stop = time.time()

    print(val)
    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Validation finished in {round(stop - start)} seconds\n")


if __name__ == "__main__":
    main()
