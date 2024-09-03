import argparse
import sys


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parsed_args = get_args(args)
    print("1" * int(parsed_args.output_length), end="")


def get_args(args):
    return get_parser().parse_args(args)


def get_parser():
    parser = argparse.ArgumentParser(
        prog="Wrong Toeplitz",
        description="The worst Toeplitz hashing implementation ever",
    )

    parser.add_argument(
        "-i",
        "--input_length",
        required=True,
        help="length of the input passed to the extractor",
    )
    parser.add_argument(
        "-o", "--output_length", required=True, help="length of the output hash"
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
