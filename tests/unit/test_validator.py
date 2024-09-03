import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from galois import GF2

from randextract import (
    ModifiedToeplitzHashing,
    ToeplitzHashing,
    Validator,
    ValidatorCustomClassAbs,
)

from .._types import NOT_NUMBERS, REALS


def _npz_file_exists():
    return len(list(Path.cwd().glob("*.npz"))) != 0


def _delete_npz_files():
    for file in Path.cwd().glob("*.npz"):
        if file.name != "toeplitz_4_3_bad.npz":
            file.unlink()


def _string_to_gf2(string):
    return GF2(np.frombuffer(string.encode(), dtype=np.uint8) - ord("0"))


def _read_lines(file, lines):
    with open(file, "r") as f:
        # Ignore first line
        f.readline()
        for i, line in enumerate(f):
            if i + 2 in lines:
                yield _string_to_gf2(line.strip())


class TestConstructor(unittest.TestCase):
    # TODO: add supported objects
    def test_good_init(self):
        with self.subTest(ext="Toeplitz"):
            ext = ToeplitzHashing(input_length=10, output_length=3)
            Validator(ext)
        with self.subTest(ext="ModifiedToeplitz"):
            ext = ModifiedToeplitzHashing(input_length=10, output_length=3)
            Validator(ext)

    def test_missing_argument(self):
        with self.assertRaises(TypeError):
            Validator()

    def test_wrong_type_argument(self):
        # We reuse some of the types from _types.py module
        for wrong_type in REALS + NOT_NUMBERS:
            with self.subTest(type=type(wrong_type)):
                with self.assertRaises(TypeError):
                    Validator(wrong_type)

    def test_too_many_arguments(self):
        ext = ToeplitzHashing(input_length=10, output_length=3)
        for extra_arg in REALS + NOT_NUMBERS:
            with self.assertRaises(TypeError):
                Validator(ext, extra_arg)


class TestAddImplementation(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(input_length=10, output_length=3)
        self.val = Validator(ext)

        class GoodCustomClass(ValidatorCustomClassAbs):
            def get_extractor_inputs(self):
                pass

            def get_extractor_seeds(self):
                pass

            def get_extractor_output(self, ext_input, ext_seed):
                pass

        self.good_custom_class = GoodCustomClass
        self.good_custom_class_instance = GoodCustomClass()

    def test_good_args(self):
        with self.subTest(input_method="stdio"):
            with self.subTest(msg="Without format_dict"):
                self.val.add_implementation(
                    "impl1",
                    input_method="stdio",
                    command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
                )
                self.assertEqual(len(self.val._implementations), 1)
                self.assertEqual(list(self.val._implementations.keys()), ["impl1"])
                self.assertSetEqual(
                    set(self.val._implementations["impl1"]),
                    {"input_method", "command", "format_dict", "validated", "valid"},
                )

            with self.subTest(msg="With format_dict"):
                self.val.add_implementation(
                    "impl2",
                    input_method="stdio",
                    command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
                    format_dict={"$INPUT_LENGTH$": lambda num: str(num) + "\n"},
                )
                self.assertEqual(len(self.val._implementations), 2)
                self.assertEqual(
                    list(self.val._implementations.keys()),
                    ["impl1", "impl2"],
                )
                self.assertSetEqual(
                    set(self.val._implementations["impl2"]),
                    {"input_method", "command", "format_dict", "validated", "valid"},
                )

        with self.subTest(input_method="read_files"):
            parser = {
                "input": (_ for _ in range(5)),
                "seed": (_ for _ in range(5)),
                "output": (_ for _ in range(5)),
            }
            self.val.add_implementation(
                "impl3", input_method="read_files", parser=parser
            )
            self.assertEqual(len(self.val._implementations), 3)
            self.assertEqual(
                list(self.val._implementations.keys()), ["impl1", "impl2", "impl3"]
            )
            self.assertSetEqual(
                set(self.val._implementations["impl3"]),
                {"input_method", "parser", "validated", "valid"},
            )

        with self.subTest(input_method="custom"):
            self.val.add_implementation(
                "impl4",
                input_method="custom",
                custom_class=self.good_custom_class_instance,
            )

        self.assertEqual(len(self.val._implementations), 4)
        self.assertEqual(
            list(self.val._implementations.keys()), ["impl1", "impl2", "impl3", "impl4"]
        )
        self.assertSetEqual(
            set(self.val._implementations["impl4"]),
            {"input_method", "custom_class", "validated", "valid"},
        )

    def test_same_label(self):
        self.val.add_implementation(
            "impl1",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )
        with self.assertWarns(UserWarning):
            self.val.add_implementation(
                "impl1",
                input_method="stdio",
                command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
            )
        self.assertEqual(len(self.val._implementations), 1)
        self.assertEqual(list(self.val._implementations.keys()), ["impl1"])

    def test_wrong_input_method(self):
        for wrong_type in REALS:
            with self.assertRaises(TypeError):
                self.val.add_implementation("impl1", input_method=wrong_type)

        with self.assertRaises(ValueError):
            # Typo: "read_file" instead of "read_files"
            self.val.add_implementation("impl1", input_method="read_file")

    def test_stdio_missing_command(self):
        with self.assertRaises(ValueError):
            self.val.add_implementation(
                "impl1",
                input_method="stdio",
            )

    def test_read_files_missing_parser(self):
        with self.assertRaises(ValueError):
            self.val.add_implementation("impl1", input_method="read_files")

    def test_read_files_wrong_parser_keys(self):
        parser = {
            "ext_input": (_ for _ in range(5)),
            "seed": (_ for _ in range(5)),
            "ext_output": (_ for _ in range(5)),
        }
        with self.assertRaises(ValueError):
            self.val.add_implementation(
                "impl1", input_method="read_files", parser=parser
            )

    def test_read_files_wrong_parser_values(self):
        parser = {
            "input": np.sin,
            "seed": np.sin,
            "output": np.sin,
        }
        with self.assertRaises(TypeError):
            self.val.add_implementation(
                "impl1", input_method="read_files", parser=parser
            )

    def test_custom_missing_custom_class(self):
        with self.assertRaises(ValueError):
            self.val.add_implementation("impl1", input_method="custom")

    def test_custom_wrong_custom_class(self):
        with self.subTest(custom_class="not a class"):
            with self.assertRaises(TypeError):
                self.val.add_implementation(
                    "impl1", input_method="custom", custom_class=True
                )

        with self.subTest(custom_class="class object"):
            with self.assertRaises(TypeError):
                self.val.add_implementation(
                    "impl1",
                    input_method="custom",
                    custom_class=self.good_custom_class,
                )

        with self.subTest(custom_class="wrong subclass"):

            class CustomClass:
                pass

            with self.assertRaises(TypeError):
                self.val.add_implementation(
                    "impl1", input_method="custom", custom_class=CustomClass()
                )


class TestRemoveImplementation(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(input_length=10, output_length=3)
        self.val = Validator(ext)
        self.val.add_implementation(
            "impl1",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )

    def test_wrong_label_type(self):
        for wrong_type in REALS:
            with self.assertRaises(TypeError):
                self.val.remove_implementation(wrong_type)

    def test_non_existing_label(self):
        with self.assertWarns(UserWarning):
            self.val.remove_implementation("impl2")

    def test_good_label(self):
        self.val.remove_implementation("impl1")
        self.assertEqual(self.val._implementations, {})


class TestReplaceImplementation(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(input_length=10, output_length=3)
        self.val = Validator(ext)
        self.val.add_implementation(
            "impl1",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )

    def test_wrong_label_type(self):
        for wrong_type in REALS:
            with self.assertRaises(TypeError):
                self.val.replace_implementation(
                    wrong_type,
                    input_method="stdio",
                    command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
                )

    def test_non_existing_label(self):
        with self.assertWarns(UserWarning):
            self.val.replace_implementation(
                "impl2",
                input_method="stdio",
                command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
            )

    def test_good_label(self):
        self.val.replace_implementation(
            "impl1",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
            format_dict={"$INPUT_LENGTH$": lambda num: str(num) + "\n"},
        )


class TestStr(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(input_length=10, output_length=3)
        self.val = Validator(ext)

    def tearDown(self):
        try:
            del os.environ["FORCE_COLOR"]
        except KeyError:
            pass
        try:
            del os.environ["NO_COLOR"]
        except KeyError:
            pass

    def test_no_implementations(self):
        self.assertEqual(self.val.__str__(), "Added implementations:\n")

    def test_one_implementation_color(self):
        os.environ["FORCE_COLOR"] = "true"
        self.val.add_implementation(
            "impl1",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )
        self.assertEqual(
            self.val.__str__(),
            "Added implementations:\n-> \x1b[1mimpl1\x1b[0m\n   Valid: \x1b[1m\x1b[33mNot validated yet\x1b[0m\n",
        )

    def test_one_implementation_no_color(self):
        os.environ["NO_COLOR"] = "true"
        self.val.add_implementation(
            "impl1",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )
        self.assertEqual(
            self.val.__str__(),
            "Added implementations:\n-> impl1\n   Valid: Not validated yet\n",
        )

    def test_two_implementations_color(self):
        os.environ["FORCE_COLOR"] = "true"
        self.val.add_implementation(
            "impl1",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )
        self.val.add_implementation(
            "impl2",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )
        self.assertEqual(
            self.val.__str__(),
            """Added implementations:
-> \x1b[1mimpl1\x1b[0m
   Valid: \x1b[1m\x1b[33mNot validated yet\x1b[0m
-> \x1b[1mimpl2\x1b[0m
   Valid: \x1b[1m\x1b[33mNot validated yet\x1b[0m
""",
        )

    def test_two_implementations_no_color(self):
        os.environ["NO_COLOR"] = "true"
        self.val.add_implementation(
            "impl1",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )
        self.val.add_implementation(
            "impl2",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )
        self.assertEqual(
            self.val.__str__(),
            "Added implementations:\n-> impl1\n   Valid: Not validated yet\n-> impl2\n   Valid: Not validated yet\n",
        )

    def test_two_implementations_color_validated(self):
        os.environ["FORCE_COLOR"] = "true"
        self.val.add_implementation(
            "impl1",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )
        self.val.add_implementation(
            "impl2",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )
        self.val._implementations["impl1"]["validated"] = True
        self.val._implementations["impl2"]["validated"] = True
        self.val._implementations["impl1"]["valid"] = True
        self.val._implementations["impl2"]["valid"] = False
        self.assertEqual(
            self.val.__str__(),
            """Added implementations:
-> \x1b[1mimpl1\x1b[0m
   Valid: \x1b[1m\x1b[32mYes\x1b[0m
-> \x1b[1mimpl2\x1b[0m
   Valid: \x1b[5m\x1b[1m\x1b[31mNo\x1b[0m
""",
        )

    def test_two_implementations_no_color_validated(self):
        os.environ["NO_COLOR"] = "true"
        self.val.add_implementation(
            "impl1",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )
        self.val.add_implementation(
            "impl2",
            input_method="stdio",
            command="./toeplitz $INPUT_LENGTH$ $OUTPUT_LENGTH$ $SEED$ $INPUT$",
        )
        self.val._implementations["impl1"]["validated"] = True
        self.val._implementations["impl2"]["validated"] = True
        self.val._implementations["impl1"]["valid"] = True
        self.val._implementations["impl2"]["valid"] = False
        self.assertEqual(
            self.val.__str__(),
            "Added implementations:\n-> impl1\n   Valid: Yes\n-> impl2\n   Valid: No\n",
        )


class TestValidateNoImplementation(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(4, 2)
        self.val = Validator(ext)

    def test_all_passed(self):
        self.assertIsNone(self.val.all_passed)
        self.val.validate(mode="random", sample_size=100)
        self.assertIsNone(self.val.all_passed)


class TestValidateWrongMode(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(4, 2)
        self.val = Validator(ext)
        path = (Path(__file__).parent / "_good_toeplitz.py").relative_to(Path.cwd())
        self.val.add_implementation(
            "wrong-toeplitz",
            input_method="stdio",
            command=f"python {path} -i $INPUT_LENGTH$ -o $OUTPUT_LENGTH$ -s $SEED$ -x $INPUT$",
        )

    def test_wrong_mode_type(self):
        for wrong_type in REALS:
            with self.subTest(mode=wrong_type):
                with self.assertRaises(TypeError):
                    self.val.validate(mode=wrong_type)

    def test_wrong_mode_value(self):
        with self.assertRaises(ValueError):
            # Typo: mode is "brute-force" not "brute"
            self.val.validate(mode="brute")


class TestValidateStdioWrongCommand(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(4, 2)
        self.path = (Path(__file__).parent / "_wrong_toeplitz.py").relative_to(
            Path.cwd()
        )
        self.val = Validator(ext)

    def tearDown(self):
        self.val.remove_implementation("wrong-toeplitz")

    def test_non_zero_exit(self):
        self.val.add_implementation(
            "wrong-toeplitz",
            input_method="stdio",
            command=f"python {self.path} -x $INPUT$",
        )
        self.assertFalse(self.val.all_passed)
        with self.assertWarns(UserWarning):
            self.val.validate(mode="random", sample_size=3)

    def test_wrong_path(self):
        self.val.add_implementation(
            "wrong-toeplitz",
            input_method="stdio",
            # typo: "pyton" instead of "python"
            command=f"pyton {self.path} -i $INPUT_LENGTH$ -o $OUTPUT_LENGTH$ -s $SEED$ -x $INPUT$",
        )
        self.assertFalse(self.val.all_passed)
        with self.assertWarns(UserWarning):
            self.val.validate(mode="random", sample_size=3)


class TestValidateStdioWrongImplementation(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(4, 2)
        path = (Path(__file__).parent / "_wrong_toeplitz.py").relative_to(Path.cwd())
        self.val = Validator(ext)
        self.val.add_implementation(
            "wrong-toeplitz",
            input_method="stdio",
            command=f"python {path} -i $INPUT_LENGTH$ -o $OUTPUT_LENGTH$ -s $SEED$ -x $INPUT$",
        )

    def tearDown(self):
        self.val.remove_implementation("wrong-toeplitz")
        _delete_npz_files()

    def test_validate_missing_mode(self):
        self.assertIsNotNone(self.val.all_passed)
        self.assertFalse(self.val.all_passed)
        with self.assertWarns(UserWarning):
            self.val.validate()
        self.assertIsNotNone(self.val.all_passed)
        self.assertFalse(self.val.all_passed)
        self.assertTrue(_npz_file_exists())

    def test_validate_random_mode_missing_sample_size(self):
        self.assertIsNotNone(self.val.all_passed)
        self.assertFalse(self.val.all_passed)
        with self.assertWarns(UserWarning):
            self.val.validate(mode="random")
        self.assertIsNotNone(self.val.all_passed)
        self.assertFalse(self.val.all_passed)
        self.assertTrue(_npz_file_exists())

    def test_validate_random(self):
        self.assertIsNotNone(self.val.all_passed)
        self.assertFalse(self.val.all_passed)
        self.val.validate(mode="random", sample_size=3)
        self.assertIsNotNone(self.val.all_passed)
        self.assertFalse(self.val.all_passed)
        self.assertTrue(_npz_file_exists())


class TestValidateStdioGoodImplementation(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(4, 2)
        if platform.system() == "Linux":
            python = "python"
        else:
            python = Path(sys.executable)
        path = (Path(__file__).parent / "_good_toeplitz.py").relative_to(Path.cwd())
        self.val = Validator(ext)
        self.val.add_implementation(
            "slow-toeplitz",
            input_method="stdio",
            command=f"{python} {path} -i $INPUT_LENGTH$ -o $OUTPUT_LENGTH$ -s $SEED$ -x $INPUT$",
            format_dict={
                "$SEED$": lambda array: str(array.tolist()),
                "$INPUT$": lambda array: str(array.tolist()),
                "$OUTPUT$": lambda out: GF2(
                    [
                        int(_)
                        for _ in out[1:5]
                        .decode("utf-8")
                        .replace(",", "")
                        .replace(" ", "")
                    ]
                ),
            },
        )

    def test_wrong_output_format(self):
        del self.val._implementations["slow-toeplitz"]["format_dict"]["$OUTPUT$"]
        with self.assertWarns(UserWarning):
            self.val.validate(mode="random", sample_size=3)
        self.assertFalse(self.val.all_passed)

    def test_validate_random(self):
        self.assertFalse(self.val.all_passed)
        self.val.validate(mode="random", sample_size=3)
        self.assertTrue(self.val.all_passed)

    def test_validate_random_wrong_rng(self):
        with self.assertRaises(TypeError):
            self.val.validate(mode="random", sample_size=3, rng="1337")

    def test_validate_random_good_rng(self):
        rng = [np.random.default_rng(1), 1337]
        for _ in rng:
            self.val.validate(mode="random", sample_size=3, rng=_)


class TestValidateBruteForceGoodImplementation(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(2, 1)
        if platform.system() == "Linux":
            python = "python"
        else:
            python = Path(sys.executable)
        path = (Path(__file__).parent / "_good_toeplitz.py").relative_to(Path.cwd())
        self.val = Validator(ext)
        self.val.add_implementation(
            "slow-toeplitz",
            input_method="stdio",
            command=f"{python} {path} -i $INPUT_LENGTH$ -o $OUTPUT_LENGTH$ -s $SEED$ -x $INPUT$",
            format_dict={
                "$SEED$": lambda array: str(array.tolist()),
                "$INPUT$": lambda array: str(array.tolist()),
                "$OUTPUT$": lambda out: GF2(
                    [
                        int(_)
                        for _ in out[1:2]
                        .decode("utf-8")
                        .replace(",", "")
                        .replace(" ", "")
                    ]
                ),
            },
        )

    def tearDown(self):
        self.val.remove_implementation("slow-toeplitz")

    def test_max_attempts(self):
        self.assertFalse(self.val.all_passed)
        self.val.validate(mode="brute-force", max_attempts=3)
        self.assertTrue(self.val.all_passed)

    def test_max_attempts_all(self):
        self.assertFalse(self.val.all_passed)
        self.val.validate(mode="brute-force", max_attempts="all")
        self.assertTrue(self.val.all_passed)

    def test_no_max_attempts(self):
        self.assertFalse(self.val.all_passed)
        with self.assertWarns(UserWarning):
            self.val.validate(mode="brute-force")
        self.assertTrue(self.val.all_passed)

    def test_max_attempts_wrong_type(self):
        for wrong_type in NOT_NUMBERS:
            if isinstance(wrong_type, str):
                continue
            with self.assertRaises(TypeError):
                self.val.validate(mode="brute-force", max_attempts=wrong_type)

    def test_max_attempts_wrong_value(self):
        with self.subTest(max_attempts=-1):
            with self.assertRaises(ValueError):
                self.val.validate(mode="brute-force", max_attempts=-1)

        with self.subTest(max_attempts="full"):
            with self.assertRaises(ValueError):
                self.val.validate(mode="brute-force", max_attempts="full")


class TestValidateCustomGoodImplementation(unittest.TestCase):
    def setUp(self):
        class CustomValidator(ValidatorCustomClassAbs):
            def __init__(self, extractor):
                self.rng = np.random.default_rng(1337)
                self.ext = extractor
                self._count = 0

            def get_extractor_inputs(self):
                while self._count < 10:
                    yield GF2.Random(self.ext.input_length, seed=self.rng)
                    self._count += 1
                return

            def get_extractor_seeds(self):
                while self._count < 10:
                    yield GF2.Random(self.ext.seed_length, seed=self.rng)
                return

            def get_extractor_output(self, ext_input, ext_seed):
                if platform.system() == "Linux":
                    python = "python"
                else:
                    python = Path(sys.executable)
                path = (
                    Path(__file__).parent.parent.parent
                    / "tests"
                    / "unit"
                    / "_good_toeplitz.py"
                )

                ext_input = str(ext_input.tolist())
                ext_seed = str(ext_seed.tolist())

                args = [
                    python,
                    path,
                    "-i",
                    str(self.ext.input_length),
                    "-o",
                    str(self.ext.output_length),
                    "-s",
                    ext_seed,
                    "-x",
                    ext_input,
                ]
                res = subprocess.run(args=args, capture_output=True)
                res.check_returncode()

                return GF2(
                    [
                        int(_)
                        for _ in res.stdout[1:2]
                        .decode("utf-8")
                        .replace(",", "")
                        .replace(" ", "")
                    ]
                )

        ext = ToeplitzHashing(2, 1)
        self.custom_class = CustomValidator(ext)
        self.val = Validator(ext)
        self.val.add_implementation(
            "slow-toeplitz", input_method="custom", custom_class=self.custom_class
        )

    def tearDown(self):
        self.val.remove_implementation("slow-toeplitz")

    def test_validate_custom_class(self):
        self.assertFalse(self.val.all_passed)
        self.val.validate()
        self.assertTrue(self.val.all_passed)

    def test_validate_warning_extra_kwargs(self):
        self.assertFalse(self.val.all_passed)
        with self.assertWarns(UserWarning):
            self.val.validate(max_attempts=2)
        self.assertTrue(self.val.all_passed)


class TestValidateBruteForceWrongImplementation(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(4, 2)
        self.path = (Path(__file__).parent / "_wrong_toeplitz.py").relative_to(
            Path.cwd()
        )
        self.val = Validator(ext)

    def tearDown(self):
        self.val.remove_implementation("wrong-toeplitz")
        _delete_npz_files()

    def test_non_zero_exit(self):
        self.val.add_implementation(
            "wrong-toeplitz",
            input_method="stdio",
            command=f"python {self.path} -x $INPUT$",
        )
        self.assertFalse(self.val.all_passed)
        with self.assertWarns(UserWarning):
            self.val.validate(mode="brute-force", max_attempts=3)

    def test_wrong_path(self):
        self.val.add_implementation(
            "wrong-toeplitz",
            input_method="stdio",
            # typo: "pyton" instead of "python"
            command=f"pyton {self.path} -i $INPUT_LENGTH$ -o $OUTPUT_LENGTH$ -s $SEED$ -x $INPUT$",
        )
        self.assertFalse(self.val.all_passed)
        with self.assertWarns(UserWarning):
            self.val.validate(mode="brute-force", sample_size=3)

    def test_wrong_results(self):
        self.val.add_implementation(
            "wrong-toeplitz",
            input_method="stdio",
            # typo: "pyton" instead of "python"
            command=f"python {self.path} -i $INPUT_LENGTH$ -o $OUTPUT_LENGTH$ -s $SEED$ -x $INPUT$",
        )
        self.assertFalse(self.val.all_passed)
        self.val.validate(mode="brute-force", max_attempts=3)
        self.assertFalse(self.val.all_passed)
        self.assertTrue(_npz_file_exists())


class TestValidateCustomWrongImplementation(unittest.TestCase):
    def setUp(self):
        class CustomValidator(ValidatorCustomClassAbs):
            def __init__(self, extractor):
                self.rng = np.random.default_rng(1337)
                self.ext = extractor
                self._count = 0

            def get_extractor_inputs(self):
                while self._count < 10:
                    yield GF2.Random(self.ext.input_length, seed=self.rng)
                    self._count += 1
                return

            def get_extractor_seeds(self):
                while self._count < 10:
                    yield GF2.Random(self.ext.seed_length, seed=self.rng)
                return

            def get_extractor_output(self, ext_input, ext_seed):
                if platform.system() == "Linux":
                    python = "python"
                else:
                    python = Path(sys.executable)
                path = (
                    Path(__file__).parent.parent.parent
                    / "tests"
                    / "unit"
                    / "_wrong_toeplitz.py"
                )

                ext_input = str(ext_input.tolist())
                ext_seed = str(ext_seed.tolist())

                args = [
                    python,
                    path,
                    "-i",
                    str(self.ext.input_length),
                    "-o",
                    str(self.ext.output_length),
                    "-s",
                    ext_seed,
                    "-x",
                    ext_input,
                ]
                res = subprocess.run(args=args, capture_output=True)
                res.check_returncode()

                return GF2(
                    [
                        int(_)
                        for _ in res.stdout[1:2]
                        .decode("utf-8")
                        .replace(",", "")
                        .replace(" ", "")
                    ]
                )

        ext = ToeplitzHashing(2, 1)
        self.custom_class = CustomValidator(ext)
        self.val = Validator(ext)
        self.val.add_implementation(
            "slow-wrong-toeplitz", input_method="custom", custom_class=self.custom_class
        )

    def tearDown(self):
        self.val.remove_implementation("slow-wrong-toeplitz")
        _delete_npz_files()

    def test_validate_custom_class(self):
        self.assertFalse(self.val.all_passed)
        self.val.validate()
        self.assertFalse(self.val.all_passed)
        self.assertTrue(_npz_file_exists())

    def test_validate_warning_extra_kwargs(self):
        self.assertFalse(self.val.all_passed)
        with self.assertWarns(UserWarning):
            self.val.validate(max_attempts=2)
        self.assertFalse(self.val.all_passed)
        self.assertTrue(_npz_file_exists())


class TestValidateReadFiles(unittest.TestCase):
    def setUp(self):
        self.ext = ToeplitzHashing(4, 3)

    def tearDown(self):
        _delete_npz_files()

    def test_toeplitz_4_3_good(self):
        file = (Path(__file__).parent / "toeplitz_4_3_good.txt").relative_to(Path.cwd())
        val = Validator(self.ext)
        val.add_implementation(
            "good",
            input_method="read_files",
            parser={
                "input": _read_lines(file, range(2, 3072, 3)),
                "seed": _read_lines(file, range(3, 3073, 3)),
                "output": _read_lines(file, range(4, 3074, 3)),
            },
        )
        val.validate()
        self.assertTrue(val.all_passed)

    def test_toeplitz_4_3_bad(self):
        file = (Path(__file__).parent / "toeplitz_4_3_bad.txt").relative_to(Path.cwd())
        val = Validator(self.ext)
        val.add_implementation(
            "bad",
            input_method="read_files",
            parser={
                "input": _read_lines(file, range(2, 3072, 3)),
                "seed": _read_lines(file, range(3, 3073, 3)),
                "output": _read_lines(file, range(4, 3074, 3)),
            },
        )
        val.validate()
        self.assertFalse(val.all_passed)
        self.assertTrue(_npz_file_exists())


class TestAnalyzeFailedTest(unittest.TestCase):
    def setUp(self):
        ext = ToeplitzHashing(4, 3)
        self.val = Validator(ext)
        self.ext_input = GF2([1, 1, 0, 1])
        self.ext_seed = GF2([1, 0, 0, 0, 1, 1])
        self.ref_output = GF2([0, 0, 1])
        self.impl_output = GF2([0, 1, 1])
        self.val.validate()

    def test_empty_filename(self):
        # Avoid failing test when manually running from unit directory
        if Path.cwd().name == "unit":
            ext_input, ext_seed, ref_output, impl_output = (
                self.val.analyze_failed_test()
            )

            np.testing.assert_array_equal(self.ext_input, ext_input)
            np.testing.assert_array_equal(self.ext_seed, ext_seed)
            np.testing.assert_array_equal(self.ref_output, ref_output)
            np.testing.assert_array_equal(self.impl_output, impl_output)
        else:
            with self.assertRaises(ValueError):
                _ = self.val.analyze_failed_test()

    def test_wrong_path(self):
        with self.assertRaises(ValueError):
            _ = self.val.analyze_failed_test("/toeplitz_4_3_bad.npz")

    def test_good_path(self):
        ext_input, ext_seed, ref_output, impl_output = self.val.analyze_failed_test(
            Path(__file__).parent / "toeplitz_4_3_bad.npz"
        )
        np.testing.assert_array_equal(self.ext_input, ext_input)
        np.testing.assert_array_equal(self.ext_seed, ext_seed)
        np.testing.assert_array_equal(self.ref_output, ref_output)
        np.testing.assert_array_equal(self.impl_output, impl_output)

    def test_wrong_func(self):
        with self.assertRaises(TypeError):
            self.val.analyze_failed_test(
                Path(__file__).parent / "toeplitz_4_3_bad.npz", func="func"
            )

    def test_good_func(self):
        def trivial_func(ext_input, ext_seed, ref_output, impl_output):
            print("Trivial func")

        _ = self.val.analyze_failed_test(
            Path(__file__).parent / "toeplitz_4_3_bad.npz", func=trivial_func
        )
