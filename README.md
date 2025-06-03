<div align="center">
<picture>
  <source srcset="https://github.com/cryptohslu/randextract/blob/main/resources/logo/logo-dark-mode.png?raw=true" media="(prefers-color-scheme: dark)">
  <img width=500px alt="randExtract logo" src="https://github.com/cryptohslu/randextract/blob/main/resources/logo/logo-light-mode.png?raw=true">
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

* [`src/randextract`](src/randextract): source code of the library.
* [`tests`](tests): unit and integration tests.
* [`docs/source`](docs/source): source code of the online documentation.
* [`examples`](examples): scripts validating real world privacy amplification implementations.
* [`resources`](resources): additional resources such as plots, datasets used in testing and the scripts to generate them,
Jupyter notebooks, test vectors, etc.

## Build & Install
You can install the latest release of the package using `pip`:

```bash
pip install randextract
```

Alternatively, you can install the current dev version cloning this git repo:
```bash
git clone https://github.com/cryptohslu/randextract.git
cd randextract
# (Optionally, create a virtual environment)
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Documentation
The documentation is available at https://randextract.crypto-lab.ch.

You can also build the documentation locally:
```bash
cd docs
make html
```

## Citation & Contact
If `randextract` was useful to you in your research, please cite us.

### BibTeX

```bibtex
@misc{randextract,
  title={randextract: a Reference Library to Test and Validate Privacy Amplification Implementations}, 
  author={Iyán Méndez Veiga and Esther Hänggi},
  year={2025},
  eprint={2506.00124},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2506.00124},
}
```

If you want to collaborate [with us](https://www.hslu.ch/en/lucerne-school-of-information-technology/research/labs/applied-cyber-security/)
at the please send us [an email](mailto:iyan.mendezveiga@hslu.ch).
