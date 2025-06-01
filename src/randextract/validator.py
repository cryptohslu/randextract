import datetime
import subprocess
import types
import warnings
from numbers import Integral
from pathlib import Path
from typing import Callable

import numpy as np
from galois import GF2
from numpy.random import Generator, default_rng
from termcolor import colored, cprint

from ._verify import verify_kwargs, verify_number_is_positive, verify_number_type, verify_type
from .randomness_extractor import RandomnessExtractor
from .utilities.converter import binary_array_to_hex_string, integer_to_binary_array
from .validator_custom_class import ValidatorCustomClassAbs

# Custom warnings
warnings.formatwarning = lambda msg, *args, **kwargs: f"{colored('UserWarning', 'light_red')}: {msg}\n\n"


class Validator:
    r"""
    The Validator class implements a way to test an arbitrary implementation of a randomness extractors against our
    reference library.

    The constructor only expects a valid :obj:`RandomnessExtractor`. Implementations to be tested can be added with
    the method obj:`add_implementation()`. To validate the provided implementation(s) against our reference library use
    the method :obj:`validate()`. Finally, to check whether the added implementation(s) have passed the tests you can
    check the boolean ``validator.all_passed`` or simply print the validator object to get a summary
    ``print(validator)``.

    Arguments:
        extractor: A valid :obj:`RandomnessExtractor` object

    Examples:
        Detailed examples using this class to test other implementations can be found
        :ref:`in the package documentation <Validating other implementations>`.

        Simple examples can also be found on the unit tests, i.e., ``tests/unit/test_validator.py``.
    """

    def __init__(self, extractor: RandomnessExtractor):
        verify_type(extractor, RandomnessExtractor)
        self._ext = extractor
        self._implementations = {}
        self._all_passed = None

    def __str__(self):
        # TODO: implement __str__ in RnadomnessExtractor as well
        msg = "Added implementations:\n"
        for impl in self._implementations:
            msg += "-> " + colored(impl, attrs=["bold"]) + "\n"
            msg += "   Valid: "
            # impl has not been validated (yet)
            if not self._implementations[impl]["validated"]:
                msg += colored("Not validated yet", "yellow", attrs=["bold"])
                msg += "\n"
            # impl has been validated
            else:
                # impl passed the validation
                if self._implementations[impl]["valid"]:
                    msg += colored("Yes", "green", attrs=["bold"])
                # impl failed the validation
                else:
                    msg += colored("No", "red", attrs=["bold", "blink"])
                msg += "\n"

        return msg

    def add_implementation(self, label: str, input_method: str, **kwargs) -> None:
        r"""
        Arguments:
            label: A name to identify the provided implementation, e.g., "rust-fft"
            input_method: This determines how the results from the provided implementation are obtained. Three methods
                are implemented at the moment: "stdio" (standard input/output), "read_files" and "custom". Use "stdio"
                when you have a binary that uses stdin to take the parameters, seeds and input, and provides the result
                via stdout. Use "read_files" if the results are saved in one or more files. Use "custom" if previous
                methods do not fit your use case. Each method expects additional keyword arguments. Read below for
                details.

        Keyword Arguments:
            command (``str``): (``input_method="stdio"``) Provide the command that gives the output hash to compare with
                the reference implementation. You can use the following variables in your command: ``$INPUT_LENGTH$``,
                ``$OUTPUT_LENGTH$``, ``$SEED$``, ``$INPUT$``. By default, ``$INPUT_LENGTH$`` and ``$OUTPUT_LENGTH$``
                are passed as strings, and ``$SEED$`` and ``$INPUT$`` as bit strings (without spaces). If you want any
                other format, check the :obj:`format_dict` kwarg.
            format_dict (``dict``): (``input_method="stdio"``) A dictionary containing functions to convert the
                variables mentioned in the command kwarg from their usual Python representation to any arbitrary format
                expected by the implementation to be tested. Use the whole variable name as key for the dict,
                e.g. ``{"$SEED$": <function>}``.
            parser (``dict``): (``input_method="read_files"``) A dict containing generators to parse the files, i.e.,
                ``{"input": <generator_input>, "seed": <generator_seed>, "output": <generator_output>}``.
                Check the examples below.
            custom_class (``class``): (``input_method="custom"``) An implementation class for the provided
                :obj:`ValidatorCustomClassAbs` abstract class. Check the documentation of the abstract class for more
                details about what you should implement. Check the examples below and the use cases subsection in the
                documentation for real scenarios using the "custom" ``input_method``.

        Examples:
            A detailed example using ``input_method="stdio"`` can be found :ref:`here <Validating using stdio>`. For
            ``input_method="read_files"`` you can check :ref:`this other example <Validating using files>`. Finally, for
            the ``input_method="custom"`` check :ref:`this complete example <Validator with custom class>`.

            In addition, you can check the very simple examples from the unit tests in ``tests/unit/test_validator.py``.
        """
        verify_type(label, str)
        if label in self._implementations.keys():
            warnings.warn(
                f"An implementation with label {label} was already added to this validator."
                + "Previous implementation will be replaced. Please, use replace_implementation() instead",
                UserWarning,
            )

        verify_type(input_method, str)
        if input_method not in ["stdio", "read_files", "custom"]:
            raise ValueError(
                f"input_method can only takes values 'stdio', 'read_files' or 'custom', but {input_method} was given."
            )

        # "stdio" mode
        if input_method == "stdio":
            if "command" not in kwargs.keys():
                raise ValueError("With input_method='stdio', you should provide, at least, the kwarg 'command'.")

            command = kwargs["command"]

            # Default conversion functions to go from integers and arrays to strings
            default_format_dict = {
                "$INPUT_LENGTH$": str,
                "$OUTPUT_LENGTH$": str,
                "$SEED$": lambda array: "".join([str(_) for _ in array]),
                "$INPUT$": lambda array: "".join([str(_) for _ in array]),
                "$OUTPUT$": lambda out: GF2([int(_) for _ in out[:-1].decode("utf-8")]),
            }

            if "format_dict" in kwargs.keys():
                format_dict = kwargs["format_dict"]
            else:
                format_dict = {}

            # If format was not provided, use default functions
            for variable in [
                "$INPUT_LENGTH$",
                "$OUTPUT_LENGTH$",
                "$SEED$",
                "$INPUT$",
                "$OUTPUT$",
            ]:
                if variable not in format_dict.keys():
                    format_dict[variable] = default_format_dict[variable]

            self._implementations[label] = {
                "input_method": "stdio",
                "command": command,
                "format_dict": format_dict,
                "validated": False,
                "valid": None,
            }

        # "read_files" mode
        if input_method == "read_files":
            if "parser" not in kwargs.keys():
                raise ValueError("With input_method='read_files', you should provide the kwarg 'parser'.")
            parser = kwargs["parser"]
            verify_type(parser, dict)

            # Check that parser dict contains the correct keys
            if set(parser.keys()) != {"input", "seed", "output"}:
                raise ValueError("parser should be a dictionary with keys: 'input', 'seed' and 'output'.")

            # Check that parser contains generators
            for value in parser.values():
                verify_type(value, types.GeneratorType)

            self._implementations[label] = {
                "input_method": "read_files",
                "parser": parser,
                "validated": False,
                "valid": None,
            }

        # "custom" mode
        if input_method == "custom":
            if "custom_class" not in kwargs.keys():
                raise ValueError("With input_method='custom', you should provide the kwarg 'custom_class'.")
            custom_class = kwargs["custom_class"]

            if not isinstance(custom_class, ValidatorCustomClassAbs):
                raise TypeError(
                    "'custom_class' must be an implementation class of the ValidatorCustomClassAbs abstract base class."
                )

            self._implementations[label] = {
                "input_method": "custom",
                "custom_class": custom_class,
                "validated": False,
                "valid": None,
            }

        self.__update_all_passed()
        return

    def remove_implementation(self, label: str) -> None:
        r"""
        It removes the implementation saved with name ``label``. If there is no implementation with passed label a
        UserWarning is raised.

        Arguments:
            label: The name associated with a saved implementation included using :obj:`add_implementation()`
        """
        verify_type(label, str)
        if label not in self._implementations:
            warnings.warn(f"There is no implementation with label={label}", UserWarning)
            return

        del self._implementations[label]

    def replace_implementation(self, label: str, input_method: str = "stdio", **kwargs) -> None:
        r"""
        If :obj:`add_implementation()` is used with a label that already exists, a UserWarning is raised. This method
        does exactly the same as :obj:`add_implementation()` but issues no warning.
        """
        self.remove_implementation(label)
        self.add_implementation(label=label, input_method=input_method, **kwargs)

    def __update_all_passed(self):
        if len(self._implementations) == 0:
            self._all_passed = None
            return

        for label in self._implementations:
            impl = self._implementations[label]

            if not impl["validated"]:
                self._all_passed = False
                return

            if not impl["valid"]:
                self._all_passed = False
                return

        self._all_passed = True

    @staticmethod
    def _save_test_case(ext_input, ext_seed, ref_output, impl_output) -> None:
        """
        Stores arrays from a fail test to a NumPy uncompressed .npz file
        """
        timestamp = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
        filename = f"{timestamp}_failed_test_arrays.npz"
        np.savez(
            filename,
            ext_input=ext_input,
            ext_seed=ext_seed,
            ref_output=ref_output,
            impl_output=impl_output,
        )
        print(f"Test failed. All involved arrays were saved to {Path.cwd()}/{filename}")

    def _validate_stdio(self, label: str, extractor_input: GF2, seed: GF2) -> bool:
        """
        It validates one implementation with ``input_method="stdio"`` against the reference one.
        """
        assert label in self._implementations.keys()
        impl = self._implementations[label]
        cmd = impl["command"]
        fmt = impl["format_dict"]

        # TODO: this could be made more robust. Paths with some characters (e.g. "-") may be problematic
        args = cmd.split()

        for i, var in enumerate(args):
            if var == "$INPUT_LENGTH$":
                args[i] = fmt["$INPUT_LENGTH$"](self._ext.input_length)
            elif var == "$OUTPUT_LENGTH$":
                args[i] = fmt["$OUTPUT_LENGTH$"](self._ext.output_length)
            elif var == "$SEED$":
                args[i] = fmt["$SEED$"](seed)
            elif var == "$INPUT$":
                args[i] = fmt["$INPUT$"](extractor_input)

        ref_output = self._ext.extract(extractor_input, seed)

        res = subprocess.run(args=args, capture_output=True)
        # Raises CalledProcessError with nonzero exit status
        res.check_returncode()

        try:
            impl_output = fmt["$OUTPUT$"](res.stdout)
        except Exception:
            warnings.warn(
                f"The output of the implementation {label} could not be converted to compare with the reference "
                + "implementation using the $OUTPUT$ conversion function in the format_dict. Implementation might be "
                + "correct but is marked as not valid because of this. Please, fix the $OUTPUT$ conversion function "
                + "and run validate() again.",
                UserWarning,
            )
            return False

        passed = np.array_equal(ref_output, impl_output)

        if not passed:
            self._save_test_case(extractor_input, seed, ref_output, impl_output)

        return passed

    def _validate_brute(self, label, max_attempts: Integral) -> bool:
        attempts = 0
        passed = False
        for i in range(2**self._ext.seed_length):
            seed = integer_to_binary_array(i, pad=self._ext.seed_length)
            for j in range(2**self._ext.input_length):
                extractor_input = integer_to_binary_array(j, pad=self._ext.input_length)
                passed = self._validate_stdio(label, extractor_input, seed)
                if not passed:
                    return passed
                attempts += 1
                if attempts > max_attempts:
                    return passed
        return passed

    def _validate_read_files(self, label) -> bool:
        parser = self._implementations[label]["parser"]

        # This may raise StopIteration, it is handle in validate()
        impl_input = next(parser["input"])
        impl_seed = next(parser["seed"])
        impl_output = next(parser["output"])

        try:
            ref_output = self._ext.extract(impl_input, impl_seed)
            passed = np.array_equal(ref_output, impl_output)
        except Exception:
            return False

        if not passed:
            self._save_test_case(impl_input, impl_seed, ref_output, impl_output)

        return passed

    def _validate_custom(self, label) -> bool:
        custom_class = self._implementations[label]["custom_class"]

        inputs = custom_class.get_extractor_inputs()
        seeds = custom_class.get_extractor_seeds()

        for impl_input in inputs:
            try:
                impl_seed = next(seeds)
            except StopIteration:
                warnings.warn(
                    "ext_seeds generator run out of seeds, but not all available inputs were tested.",
                    UserWarning,
                )
                break

            try:
                impl_output = custom_class.get_extractor_output(impl_input, impl_seed)
            except Exception:
                warnings.warn(
                    f"The custom_class failed to return the implementation output. Implementation {label} is marked as "
                    + "not valid, but it could be a false negative caused by a bug in the provided custom_class.",
                    UserWarning,
                )
                return False

            try:
                ref_output = self._ext.extract(impl_input, impl_seed)
                passed = np.array_equal(ref_output, impl_output)
            except Exception:
                return False

            if not passed:
                self._save_test_case(impl_input, impl_seed, ref_output, impl_output)
                return False

        try:
            # Since we iterate over the inputs generator, we test now if some seeds were not used
            impl_seed = next(seeds)
        except StopIteration:
            # This is actually what we expect, the seeds generator is exhausted at the same time as the inputs one
            pass
        else:
            warnings.warn(
                "At least one seed from the get_extractor_seeds() generator was not used. This does not affect the "
                + "validation outcome, but it may indicate that there is a bug in the provided custom_class.",
                UserWarning,
            )
        finally:
            return passed

    def validate(self, **kwargs) -> None:
        r"""
        It validates the added implementation(s) with :obj:`add_implementation()` against the reference extractor with
        which the validator was constructed. Implementations with ``input_method="stdio"`` can be tested using two
        different methods: randomly or with a brute force comparison. Check the keyword arguments for more details. This
        method does not return anything, but it updates the keys "validated" and "valid"

        Keyword Arguments:
            mode (``str``): This only affects implementations with ``input_method="stdio"``. Two modes are available:
                "random" and "brute-force". Random tests take seeds and inputs uniformly at random and compares the
                output of the reference implementation with the added implementation(s)
            sample_size (``int``): (``mode="random"``) The number of random inputs and seeds that will be used to
                validate the implementations added with ``input_method="stdio"``
            max_attempts (``int`` | ``str``): (``mode="brute-force"``) The max number of testing rounds. Use
                ``max_attempts="all"`` if you want to run an exhaustive brute-force testing trying all possible input
                and seeds.
            rng (``int`` | ``Generator`` | ``None``): (``mode="random"``) Seed to initialize the NumPy RNG or,
                alternatively, an already initialized ``Generator``, e.g. ``numpy.random.default_rng(1337)``. This only
                affects implementations with ``input_method="stdio"``
        """
        for label in self._implementations:
            impl = self._implementations[label]
            # "stdio" validation
            if impl["input_method"] == "stdio":
                if "mode" not in kwargs:
                    warnings.warn(
                        f"You have added {label} as an implementation with input_method='stdio' but not provided a "
                        + "mode to validate it. Using mode='random'",
                        UserWarning,
                    )
                    mode = "random"
                else:
                    mode = kwargs["mode"]
                    verify_type(mode, str)
                    if mode not in ["random", "brute-force"]:
                        raise ValueError(f"mode can be either 'random' or 'brute-force', but {mode} was given")

                # "random" testing
                if mode == "random":
                    if "sample_size" not in kwargs:
                        warnings.warn(
                            "mode='random' is used but sample_size not provided. Using sample_size=1000",
                            UserWarning,
                        )
                        sample_size = 1000
                    else:
                        sample_size = kwargs["sample_size"]
                        verify_number_type(sample_size, Integral)

                    if "rng" in kwargs:
                        rng = kwargs["rng"]
                        verify_type(rng, [int, Generator])
                        rng = default_rng(rng)
                    else:
                        rng = default_rng()

                    for _ in range(sample_size):
                        extractor_input = GF2.Random(self._ext.input_length, seed=rng)
                        seed = GF2.Random(self._ext.seed_length, seed=rng)

                        try:
                            passed = self._validate_stdio(label, extractor_input, seed)
                        except subprocess.CalledProcessError:
                            warnings.warn(
                                f"Provided command to run the implementation {label} terminated with a nonzero exit "
                                + "code. Double check the command for errors.",
                                UserWarning,
                            )
                            passed = False
                        except FileNotFoundError:
                            warnings.warn(
                                f"Provided command to run the implementation {label} failed to run because the "
                                + f"binary {impl['command'].split()[0]} does not exist.",
                                UserWarning,
                            )
                            passed = False

                        if not passed:
                            break

                # "brute-force" testing
                else:
                    if "max_attempts" not in kwargs:
                        warnings.warn(
                            "mode='brute-force' is used but sample_size not provided. " + "Using max_attempts='all'",
                            UserWarning,
                        )
                        max_attempts = "all"
                    else:
                        max_attempts = kwargs["max_attempts"]

                    if "rng" in kwargs:
                        warnings.warn("The RNG is not used when mode='brute-force'", UserWarning)

                    verify_type(max_attempts, [Integral, str])
                    if isinstance(max_attempts, str):
                        if max_attempts != "all":
                            raise ValueError(
                                f"max_attempts can only be an integer or 'all' but {max_attempts} was given."
                            )
                        else:
                            max_attempts = 2**self._ext.seed_length * 2**self._ext.input_length
                    else:
                        verify_number_is_positive(max_attempts)

                    try:
                        passed = self._validate_brute(label, max_attempts)
                    except subprocess.CalledProcessError:
                        warnings.warn(
                            f"Provided command to run the implementation {label} terminated with a nonzero exit "
                            + "code. Double check the command for errors.",
                            UserWarning,
                        )
                        passed = False
                    except FileNotFoundError:
                        warnings.warn(
                            f"Provided command to run the implementation {label} failed to run because the "
                            + f"binary {impl['command'].split()[0]} does not exist.",
                            UserWarning,
                        )
                        passed = False

            # "read_files" validation
            elif impl["input_method"] == "read_files":
                first = True
                while True:
                    try:
                        passed = self._validate_read_files(label)
                        first = False
                    except StopIteration:
                        # Case where we never read anything (generators failed on first attempt)
                        if first:
                            passed = False
                        break

                    if not passed:
                        break

            # "custom" validation
            elif impl["input_method"] == "custom":
                if len(kwargs) != 0:
                    warnings.warn(
                        "Some additional kwargs were given, but input_method='custom' will ignore them.",
                        UserWarning,
                    )
                passed = self._validate_custom(label)

            impl["validated"] = True
            if passed:
                impl["valid"] = True
            else:
                impl["valid"] = False

        self.__update_all_passed()

    def generate_test_vector(self, output_filename: str | Path, number_tests: int, mode: str = "rsp", **kwargs) -> None:
        r"""
        It generates `CAVP-alike`_ random test vectors compatible with the randomness extractor used to instantiate the
        :obj:`Validator` class. The function can be used to generate "request" files (.req) or "response" files (.rsp).
        The difference is that the request files only contain the input bit strings, while the response files also
        contain the expected output of the extractor. By default, response files are generated.

        .. _CAVP-alike: https://csrc.nist.gov/Projects/Cryptographic-Algorithm-Validation-Program

        Arguments:
            output_filename: The name of the output file (or a ``pathlib.Path`` object) to store the test vectors.
            number_tests: Number of tests to store in the output file.
            mode: Either ``"req"`` for request files, containing just the input for the extractor, or ``"rsp"`` for
                response files, containing both the inputs and the expected output.

        Keyword Arguments:
            overwrite (``bool``): Whether to overwrite the output file if it already exists, or not (default).
            rng (``int`` | ``Generator`` | ``None``): Seed to initialize the NumPy RNG or, alternatively, an already
                initialized ``Generator``, e.g. ``numpy.random.default_rng(1337)``.

        Examples:
            The following code snippet will generate a file ``toeplitz_hashing_testvec_1e6_cr_1_2.rsp`` containing 8
            test vectors for the Toeplitz hashing taking inputs of :math:`10^6` and a compression ratio of 1/2.

            .. code-block:: python

                from randextract import ToeplitzHashing, Validator

                ref_ext = ToeplitzHashing(10**6, 5*10**5)
                val = Validator(ref_ext)
                val.generate_test_vector(
                    output_filename="toeplitz_hashing_testvec_1e6_cr_1_2.rsp",
                    number_tests=8,
                    mode="rsp",
                )

        """
        verify_type(output_filename, [str, Path])
        verify_number_type(number_tests, Integral)
        verify_number_is_positive(number_tests)
        verify_type(mode, str)
        verify_kwargs(kwargs, optional_args=["overwrite", "rng"])

        if mode not in ("req", "rsp"):
            raise ValueError(f"Mode can only take values 'req' or 'rsp', but {mode} was passed.")

        if isinstance(output_filename, str):
            out = Path(Path.cwd() / output_filename)
        else:
            out = output_filename

        if "overwrite" in kwargs:
            overwrite = kwargs.get("overwrite")
            verify_type(overwrite, bool)
        else:
            overwrite = False

        if "rng" in kwargs:
            rng = kwargs.get("rng")
            verify_type(rng, [int, Generator])
            rng = default_rng(rng)
        else:
            rng = default_rng()

        if out.exists() and not overwrite:
            raise ValueError(
                f"File {out.name} already exists. To overwrite it call the function with argument overwrite=True"
            )

        with open(out, "w") as file:
            file.write(
                f"""# CAVS
# {type(self._ext).__name__}
# Input Length : {self._ext.input_length}
# Compression ratio: {round(self._ext.output_length / self._ext.input_length, 2)}
# Generated on {datetime.datetime.now(datetime.timezone.utc):%a %B %d %H:%M:%S %Y (UTC)}

[EXTRACT]
"""
            )
            for i in range(number_tests):
                file.write(f"COUNT = {i}\n")

                ext_input = GF2.Random(self._ext.input_length, seed=rng)
                file.write(f"INPUT = {binary_array_to_hex_string(ext_input)}\n")

                ext_seed = GF2.Random(self._ext.seed_length, seed=rng)
                file.write(f"SEED = {binary_array_to_hex_string(ext_seed)}\n")

                if mode == "rsp":
                    ext_output = self._ext.extract(ext_input, ext_seed)
                    file.write(f"OUTPUT = {binary_array_to_hex_string(ext_output)}\n")

                file.write("\n")

    @staticmethod
    def analyze_failed_test(
        filename: str | Path | None = None, func: Callable | None = None
    ) -> tuple[GF2, GF2, GF2, GF2]:
        r"""
        It loads to memory the arrays from a failed test, saved in a npz file by the ``validate()`` method, and prints
        some useful information to help debug the wrong implementation.

        Arguments:
            filename: The name of the npz file saved by ``validate()`` on a failed test. If no
                filename is provided (default), the newest npz file will be loaded.
            func: By default, a simple analysis of the two outputs is performed. Here you can provide an additional
                function to run on the arrays. It should accept for arrays as input and return nothing.

        Returns:
            ext_input: GF2 array with the input passed to both the reference and the implementation being tested
            ext_seed: GF2 array with the seed passed to both the reference and the implementation being tested
            ref_output: GF2 array with the output of the reference implementation
            impl_output: GF2 array with the output from the implementation being tested
        """
        if filename is None:
            try:
                filename = sorted(Path.cwd().glob("*.npz"))[-1]
            except IndexError:
                raise ValueError(
                    f"There are no .npz files in your current directory: {Path.cwd()}",
                    UserWarning,
                )

        verify_type(filename, [str, Path])
        try:
            data = np.load(filename)
        except FileNotFoundError:
            raise ValueError(
                f"You tried to read {filename} but the file does not exist. Your current working directory is {Path.cwd()}."
            )

        if data.files != ["ext_input", "ext_seed", "ref_output", "impl_output"]:
            raise ValueError(
                "The provider file does not contain all the expected arrays. If this file was generated by our"
                + "Validator, please report a bug."
            )

        ext_input = data["ext_input"]
        ext_seed = data["ext_seed"]
        ref_output = data["ref_output"]
        impl_output = data["impl_output"]

        # TODO: uncomment when __str__ is implemented
        # print(self._ext)

        len_n = len(f"{max(ext_input.size, ext_seed.size):_}")
        print("\nArray sizes (bits):")
        print(f"  ext_input  : {ext_input.size:>{len_n}_}")
        print(f"  ext_seed   : {ext_seed.size:>{len_n}_}")
        print(f"  ref_output : {ref_output.size:>{len_n}_}")
        print(f"  impl_output: {impl_output.size:>{len_n}_}")

        if ref_output.shape != impl_output.shape:
            print("\nExtractor outputs have different shapes:")
            print(f"  ref : {ref_output.shape}")
            print(f"  impl: {impl_output.shape}")

        elif not np.array_equal(ref_output, impl_output):
            print("\nExtractor outputs are not equal:")
            mismatched = ref_output != impl_output
            mismatched_percentage = f"{mismatched.sum() / ref_output.size:.1%}"
            print(f"  Mismatched elements: {mismatched.sum()} / {ref_output.size} ({mismatched_percentage})")
            print(f"  Sample of mismatched indices:\n{np.argwhere(mismatched).flatten()}")

        if func is not None:
            if not callable(func):
                raise TypeError("func should be callable")
            print("\nOutput of custom function:\n")
            func(ext_input, ext_seed, ref_output, impl_output)
            print("\nEnd of custom function output.\n")

        return ext_input, ext_seed, ref_output, impl_output

    @property
    def all_passed(self):
        return self._all_passed
