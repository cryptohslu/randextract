from math import floor
from pathlib import Path

from randextract import ModifiedToeplitzHashing, ToeplitzHashing, Validator


def main():
    input_size = 10**6
    ratios = [2, 3, 4]

    # Toeplitz hashing
    for r in ratios:
        file = Path(__file__).parent.parent / "test_vectors" / f"toeplitz_hashing_testvec_1e6_cr_1_{r}.rsp"
        ref_ext = ToeplitzHashing(input_size, floor(input_size / r))
        val = Validator(ref_ext)
        val.generate_test_vector(file, 8, "rsp", overwrite=True)

    # Modified Toeplitz hashing
    for r in ratios:
        file = Path(__file__).parent.parent / "test_vectors" / f"modified_toeplitz_hashing_testvec_1e6_cr_1_{r}.rsp"
        ref_ext = ModifiedToeplitzHashing(input_size, floor(input_size / r))
        val = Validator(ref_ext)
        val.generate_test_vector(file, 8, "rsp", overwrite=True)


if __name__ == "__main__":
    main()
