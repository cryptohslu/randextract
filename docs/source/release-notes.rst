=============
Release Notes
=============

:obj:`randextract` follows the `semantic versioning`_ convention. Releases are versioned ``major.minor.patch``.

Major versions may introduce API-changing features. Minor versions add features that are backwards-compatible
with previous releases. Patch versions make backwards-compatible bug fixes.

.. warning::
   Until version ``1.0.0`` is released, no API compatibility is guaranteed for any release.

.. _semantic versioning: https://semver.org/

------
v0.2.2
------

- Add missing module that got lost when migrating from internal private repo

------
v0.2.1
------

- Fix logo paths in PyPI

------
v0.2.0
------

- Add new method ``generate_test_vector()`` to :obj:`Validator` to create CAVP-alike test vectors
- Include examples of test vectors for Toeplitz hashing
- Update repo structure and citation to match `arXiv preprint <https://arxiv.org/abs/2506.00124>`_

------
v0.1.1
------

- Add suport for Python 3.13

------
v0.1.0
------

- First public release of :obj:`randextract`
- Python implementations of (modified) Toeplitz hashing and Trevisan's construction
- Documentation with introduction to theory, usage and examples
