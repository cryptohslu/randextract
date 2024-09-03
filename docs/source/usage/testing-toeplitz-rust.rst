.. py:module:: randextract
   :no-index:

============================
Rust (CPU) Toeplitz hashing
============================

.. seealso::

   The full source code of this example is in ``examples/validation_rust_toeplitz.py``

In recent years, Rust has emerged as a compelling programming language for a variety of applications. Its versatility
stems from its ability to produce both high-performance, low-level code (e.g., `Asix PHY driver`_, the first Rust driver
incorporated into the Linux kernel, or `Asahi DRM`_, the first Rust GPU kernel driver) as well as sophisticated,
memory-safe high-level applications (e.g., `sudo-rs`_ or `uutils coreutils`_).

.. _Asix PHY driver: https://git.kernel.org/pub/scm/linux/kernel/git/netdev/net-next.git/commit/?id=cbe0e415089636170aa6eb540ca4af5dc9842a60
.. _Asahi DRM: https://lore.kernel.org/asahi/20230307-rust-drm-v1-18-917ff5bc80a8@asahilina.net/
.. _sudo-rs: https://github.com/memorysafety/sudo-rs
.. _uutils coreutils: https://github.com/uutils/coreutils

`Said Aroua`_ implemented the Toeplitz hashing using Rust. In this example, we validate his implementation in two
different ways using the :obj:`Validator` class. First, we use standard input and output to brute-force testing all
possible inputs and seeds for a small family of Toeplitz hash functions. Second, we validate a large family by reading
files with random cases generated directly with the Rust implementation.

.. _Said Aroua: https://github.com/Daaiid

----------------------------------
Setting up the Rust implementation
----------------------------------

In order to replicate this example, you first need to download and compile the Rust implementation. This can be done
with the following commands:

.. code-block:: bash

   git clone https://github.com/cryptohslu/toeplitz-rust.git
   cd toeplitz-rust
   cargo build --release

The relevant binary, ``toeplitz``, can be found in the ``target/releases`` directory.

----------------------
Validating using stdio
----------------------

The Rust program contains actually three different implementations for the same family of Toeplitz hash functions:
``simple`` computes the hash by performing the matrix-vector multiplication explicitly. This only works for very small
input and output lengths, as the required memory grows linearly with the product of these two values. ``fft`` and
``realfft`` use the fast fourier transform trick to compute the matrix-vector multiplication as described
:ref:`in this theory subsection <Efficient hashing with Fast Fourier Transform>`. We brute-force test the three of them
by adding three implementations with different labels and passing the correct ``command``. This is done in the
``brute_force_stdio_test()`` function.

.. literalinclude:: /../../examples/validation_rust_toeplitz.py
   :language: python
   :pyobject: brute_force_stdio_test

----------------------
Validating using files
----------------------

In the function ``read_files_test()`` we try the ``input_method="read_files"``. Together with ``input_method="custom"``
that was presented in :ref:`the GPU example <GPU modified Toeplitz hashing>` we have covered all the (current) modes to
pass a implementation to :obj:`Validator`.

.. literalinclude:: /../../examples/validation_rust_toeplitz.py
   :language: python
   :pyobject: read_files_test

.. figure:: /images/screenshot-validation-rust-example.png
   :width: 90%
   :figclass: margin-caption
   :alt: Screenshot validation

   Screenshot after running this example
