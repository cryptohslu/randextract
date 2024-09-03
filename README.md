<div align="center">
<picture>
  <source srcset="logo/logo-dark-mode.png" media="(prefers-color-scheme: dark)">
  <img width=500px alt="randExtract logo" src="logo/logo-light-mode.png">
</picture>
</div>
<br>

The `randextract` library is a Python 3 package implementing randomness extractors that can be used to transform
weak random sources into almost-uniform ones. The library implements quantum-proof strong randomness extractors that can
be used in the Privacy Amplification (PA) step of any Quantum Key Distribution (QKD) or Quantum Random Number Generator
(QRNG) protocol. Since quantum-proof strong randomness extractors are also classical-proof, these extractors are also
well-suited for applications that involve only classical or no side-information at all.

Our goal is to provide an easy-to-read *reference* library, whose correctness can be easily verified, that can be used
to validate high performance implementations (usually hardware based) that are more difficult to audit and test.

## Structure of the repo
The source code of the library is in `src/randextract`, unit and integration tests in `tests`, and the documentation in
`docs/source`. Additional tools such as plots, datasets used for testing and the scripts to generate them, Jupyter
notebooks, etc. are all in `tools`.

## Build & install
You can install the latest release of the package using `pip`:

```bash
pip install randextract
```

You can also install the current dev version cloning this git repo:
```bash
git clone https://github.com/cryptohslu/randextract.git
cd randextract
# (Optionally, create a virtual environment)
python -m venv .venv
source .venv/bin/activate
pip install .
```

The documentation is available at https://randextract.crypto-lab.ch.

You can also build the documentation locally:
```bash
cd docs
make html
```
