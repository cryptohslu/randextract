==========
Motivation
==========

In the :ref:`home page <randExtract>` we showed the usefulness of randomness extractors with an example. While it may
have seemed like an artificial scenario, the need to eliminate *side-information* or *extract* good\ [#good]_ randomness
from a biased or weak source is, in fact, a common occurrence in numerous cryptographic protocols. In addition,
randomness extractors have played and still play an important role in theoretical computer science.

.. [#good] Different definitions for *good* randomness can be found in the literature. In this context, by *good* we
   mean that it looks uniform from the point of view of an adversary that may have access to some side-information. This
   definition aligns with the criterion of *cryptographic randomness*, which requires that the randomness remains
   *unpredictable* even for an adversary.


---------------------------
Randomness and cryptography
---------------------------

Randomness plays a crucial role in cryptography. Almost all cryptographic protocols require good randomness. Moreover,
*information-theoretically* secure protocols require strict conditions for the randomness being used. For example, the
security of the one-time pad encryption only holds if the key is truly uniform. If this is not the case and the key is,
for example, the output of a `pseudo random number generator`_ (PRNG), then the encryption is only as strong as the PRNG.

.. _pseudo random number generator: https://en.wikipedia.org/wiki/Pseudorandom_number_generator


-----------------------------------
Imperfect randomness and extractors
-----------------------------------

The problem is that it is difficult to obtain *true randomness*\ [#truerand]_ from any physical observation. One
possible approach is to design and run quantum experiments whose possible outcomes are uniformly distributed. In an
ideal setup, we could, *in principle*, obtain perfectly uniform and unpredictable randomness. This is not the case in
more realistic scenarios where imperfections, either in the preparation of the quantum states or in the measurements,
will affect the overall *quality* of the output randomness.

.. [#truerand] Similar to *good* randomness, it is possible to find different definitions for *true randomness* in the
   literature. In this context, we mean randomness whose unpredictability does not arise from the lack of information
   about the setup (e.g. a fair dice would be predictable if we knew the exact initial conditions of the throw), but
   from a more fundamental reason (e.g. from the inherent random behaviour of a quantum system).

Here is where randomness extractors come into play. Randomness extractors are deterministic functions that can
convert imperfect randomness\ [#imperfect]_ into :math:`\epsilon`-close-to-perfect randomness\ [#perfect]_, where
:math:`\epsilon` can be a positive number as small as desired, at the expense of shortening the output length.

.. [#imperfect] One possible way of characterizing an imperfect randomness source is by giving a lower bound on its
   conditional min-entropy. Roughly speaking, this is a bound on the probability of correctly guessing the outcome.

.. [#perfect] How close the output is to the uniform distribution is measured in terms of the statistical distance.

There are different families of randomness extractors. For example,

1. *seeded randomness extractors* take as input a weak randomness source and a small uniform seed, or
2. *multi-source randomness extractors* that, instead of a perfect seed, take two or more weak
   imperfect randomness sources.

:obj:`randextract` currently implements seeded randomness extractors, specifically, seeded quantum-proof strong
randomness extractors that are well-suited for applications that deal with quantum, classical or no side-information.


--------
Examples
--------

.. toctree::
   :maxdepth: 1

   use-cases/qkd.rst
   use-cases/qrng.rst
