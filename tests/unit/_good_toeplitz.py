import argparse
import sys

from galois import GF2


def toeplitz(input_length, output_length, seed, extractor_input):
    toeplitz_matrix = GF2.Zeros((output_length, input_length))
    seed = GF2([int(_) for _ in seed[1:-1].replace(" ", "").replace(",", "")])
    extractor_input = GF2(
        [int(_) for _ in extractor_input[1:-1].replace(" ", "").replace(",", "")]
    )
    for i in range(output_length):
        for j in range(input_length):
            toeplitz_matrix[i, j] = seed[i - j]

    output = toeplitz_matrix @ extractor_input
    print(output.tolist())


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parsed_args = get_args(args)
    toeplitz(
        parsed_args.input_length,
        parsed_args.output_length,
        parsed_args.seed,
        parsed_args.input,
    )


def get_args(args):
    return get_parser().parse_args(args)


def get_parser():
    parser = argparse.ArgumentParser(
        prog="Good Toeplitz",
        description="A very slow, but correct, Toeplitz hashing implementation",
    )

    parser.add_argument(
        "-i",
        "--input_length",
        required=True,
        type=int,
        help="length of the input passed to the extractor",
    )
    parser.add_argument(
        "-o",
        "--output_length",
        required=True,
        type=int,
        help="length of the output hash",
    )
    parser.add_argument(
        "-s",
        "--seed",
        required=True,
        help="the seed as a bit string used to determine the Toeplitz matrix",
    )
    parser.add_argument(
        "-x", "--input", required=True, help="the input as a bit string"
    )

    return parser


if __name__ == "__main__":
    main()
