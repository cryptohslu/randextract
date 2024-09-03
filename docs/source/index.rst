.. Randextract documentation master file, created by
   sphinx-quickstart on Mon Mar 13 21:49:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

-----------
randExtract
-----------

The :obj:`randextract` library is a Python 3 package implementing randomness extractors that can be used to transform
weak random sources into almost-uniform ones. The library implements quantum-proof strong randomness extractors that can
be used in the Privacy Amplification (PA) step of any Quantum Key Distribution (QKD) or Quantum Random Number Generator
(QRNG) protocol. Since quantum-proof strong randomness extractors are also classical-proof, these extractors are also
well-suited for applications that involve only classical or no side-information at all.

Our goal is to provide an easy-to-read *reference* library, whose correctness can be easily verified, and that can be
used to validate high performance implementations (usually hardware based) that are more difficult to audit and test.

.. toctree::
   :hidden:

   getting-started
   theory/index
   usage/index
   usage/examples
   tests/index
   api
   release-notes
   references


-------------------------------------
Why do we need randomness extractors?
-------------------------------------

Imagine you have a shared secret with a friend, and assume this secret is a binary string\ [#binary]_ picked uniformly
at random. It turns out that this kind of secret is a very useful resource for numerous cryptographic applications.
For example, you could employ it to authenticate your messages over a public channel or even encrypt them.
Unfortunately, before utilizing the secret, you discover that someone else has learned some of its bits, although
you are uncertain which ones exactly. If you knew which ones got compromised, you could simply discard them and utilize
the remaining ones. This serves as an example where randomness extractors prove to be extremely useful. They enable you
to *extract* the *good* bits by employing some additional randomness.

.. [#binary] This is not limiting at all since any kind of information can be encoded into a bit string.

If you are interested in learning more about randomness extractors check :ref:`the theory section <Theory>`. If you
just want to learn how to use this Python package, you can jump directly to :ref:`the usage section <Usage>`.


--------
Citation
--------

.. code-block:: bibtex

   @software{randextract_2024,
       title = {{randextract: A reference Python package for testing & validating Privacy Amplification implementations}},
       author = {Iyán Méndez Veiga and Esther Hänggi},
       year = {2024},
       url = {https://github.com/cryptolab/randextract},
   }
