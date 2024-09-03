import platform
import sys
import subprocess
from pathlib import Path
import numpy as np
from galois import GF2
from randextract import ToeplitzHashing, Validator, ValidatorCustomClassAbs


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
            Path(__file__).parent.parent.parent / "tests" / "unit" / "_good_toeplitz.py"
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


def main():
    ext = ToeplitzHashing(2, 1)
    val = Validator(ext)
    custom_class = CustomValidator(ext)
    val.add_implementation(
        "slow-toeplitz", input_method="custom", custom_class=custom_class
    )
    print(val)
    val.validate()
    print(val)


if __name__ == "__main__":
    main()
