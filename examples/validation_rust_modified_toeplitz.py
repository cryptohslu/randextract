"""
This example tests the high-performance Rust implementation from https://github.com/cryptohslu/toeplitz-rust
against randextract using the Validator class with input_method='stdio'.

This example assumes that the binary modified_toeplitz is in the same folder as the working directory from where
you execute the script. For example, you can generate such a file running the following:

git clone https://github.com/cryptohslu/toeplitz-rust.git
cd toeplitz-rust
cargo build --release
cp target/release/modified_toeplitz <randExtract working directory>
"""

import datetime
import time
import secrets

import numpy as np
from galois import GF2
from randextract import ModifiedToeplitzHashing, Validator

INPUT_LENGTH_MIN = 2
INPUT_LENGTH_MAX = 10**5
RANDOM_TESTING_ROUNDS = 100
SAMPLES_PER_ROUND = 100


def gf2_to_str(gf2_arr):
    return (np.array(gf2_arr) + ord("0")).tobytes().decode()


def edge_cases_testing():
    print("\nFirst we test some edge cases")

    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Starting to test edge-cases...\n")
    start = time.perf_counter()

    # Smallest valid modified Toeplitz extractor
    ext = ModifiedToeplitzHashing(2, 1)
    val = Validator(ext)
    val.add_implementation(
        label="Rust-stdio-simple",
        input_method="stdio",
        command="./modified_toeplitz simple run-args $SEED$ $INPUT$ $OUTPUT_LENGTH$",
        format_dict={"$SEED$": gf2_to_str, "$INPUT$": gf2_to_str},
    )
    val.add_implementation(
        label="Rust-stdio-fft",
        input_method="stdio",
        command="./modified_toeplitz fft run-args $SEED$ $INPUT$ $OUTPUT_LENGTH$",
        format_dict={"$SEED$": gf2_to_str, "$INPUT$": gf2_to_str},
    )
    val.add_implementation(
        label="Rust-stdio-realfft",
        input_method="stdio",
        command="./modified_toeplitz realfft run-args $SEED$ $INPUT$ $OUTPUT_LENGTH$",
        format_dict={"$SEED$": gf2_to_str, "$INPUT$": gf2_to_str},
    )
    val.validate(mode="brute-force", max_attempts="all")

    # Modified Toeplitz matrix is a row
    ext = ModifiedToeplitzHashing(3, 1)
    val = Validator(ext)
    val.add_implementation(
        label="Rust-stdio-simple",
        input_method="stdio",
        command="./modified_toeplitz simple run-args $SEED$ $INPUT$ $OUTPUT_LENGTH$",
        format_dict={"$SEED$": gf2_to_str, "$INPUT$": gf2_to_str},
    )
    val.add_implementation(
        label="Rust-stdio-fft",
        input_method="stdio",
        command="./modified_toeplitz fft run-args $SEED$ $INPUT$ $OUTPUT_LENGTH$",
        format_dict={"$SEED$": gf2_to_str, "$INPUT$": gf2_to_str},
    )
    val.add_implementation(
        label="Rust-stdio-realfft",
        input_method="stdio",
        command="./modified_toeplitz realfft run-args $SEED$ $INPUT$ $OUTPUT_LENGTH$",
        format_dict={"$SEED$": gf2_to_str, "$INPUT$": gf2_to_str},
    )
    val.validate(mode="brute-force", max_attempts="all")

    # Toeplitz part is a column, rest is identity matrix
    ext = ModifiedToeplitzHashing(3, 2)
    val = Validator(ext)
    val.add_implementation(
        label="Rust-stdio-simple",
        input_method="stdio",
        command="./modified_toeplitz simple run-args $SEED$ $INPUT$ $OUTPUT_LENGTH$",
        format_dict={"$SEED$": gf2_to_str, "$INPUT$": gf2_to_str},
    )
    val.add_implementation(
        label="Rust-stdio-fft",
        input_method="stdio",
        command="./modified_toeplitz fft run-args $SEED$ $INPUT$ $OUTPUT_LENGTH$",
        format_dict={"$SEED$": gf2_to_str, "$INPUT$": gf2_to_str},
    )
    val.add_implementation(
        label="Rust-stdio-realfft",
        input_method="stdio",
        command="./modified_toeplitz realfft run-args $SEED$ $INPUT$ $OUTPUT_LENGTH$",
        format_dict={"$SEED$": gf2_to_str, "$INPUT$": gf2_to_str},
    )
    val.validate(mode="brute-force", max_attempts="all")

    stop = time.perf_counter()
    print(val)
    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Edge-case testing finished in {round(stop - start)} seconds\n")


def random_testing():
    print("\nNow we do some random testing for arbitrary input and output lengths")

    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Starting random testing...\n")
    start = time.perf_counter()

    for _ in range(RANDOM_TESTING_ROUNDS):
        print(f"Random testing ({_ + 1}/{RANDOM_TESTING_ROUNDS})")
        random_input_length = 0
        while random_input_length < 2:
            random_input_length = secrets.randbelow(INPUT_LENGTH_MAX)
        random_output_length = secrets.randbelow(random_input_length - 1)
        print(f"\nTesting {SAMPLES_PER_ROUND} samples with:")
        print(f"   input_length  = {random_input_length}")
        print(f"   output_length = {random_output_length}")
        ext = ModifiedToeplitzHashing(random_input_length, random_output_length)
        val = Validator(ext)
        val.add_implementation(
            label="Rust-stdio-fft",
            input_method="stdio",
            command="./modified_toeplitz fft run-args $SEED$ $INPUT$ $OUTPUT_LENGTH$",
            format_dict={"$SEED$": gf2_to_str, "$INPUT$": gf2_to_str},
        )
        val.add_implementation(
            label="Rust-stdio-realfft",
            input_method="stdio",
            command="./modified_toeplitz realfft run-args $SEED$ $INPUT$ $OUTPUT_LENGTH$",
            format_dict={"$SEED$": gf2_to_str, "$INPUT$": gf2_to_str},
        )
        val.validate(mode="random", sample_size=SAMPLES_PER_ROUND)
        print(val)
        if not val.all_passed:
            break

    stop = time.perf_counter()
    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Random testing finished in {round(stop - start)} seconds\n")


def main():
    print("\nThis script tests a high-performance Rust implementation of the modified")
    print("Toeplitz hashing against the class ModifiedToeplitzHashing.")
    edge_cases_testing()
    random_testing()


if __name__ == "__main__":
    main()
