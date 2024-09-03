==================================
Use case: Quantum Key Distribution
==================================

Although randomness extractors are useful in many different scenarios, writing this Python package was motivated mainly
by their usage in Quantum Key Distribution (QKD). Because of that, we want to give a brief introduction to QKD. Feel
free to jump directly to the :ref:`next section <Seeded randomness extractors>` if you are not interested in QKD.

.. admonition:: Quantum cryptography :math:`\neq` Quantum computing
   :class: note

   Quantum cryptography, and in particular QKD, **has nothing to do with quantum computing**. QKD can be implemented
   *without* access to a quantum computer. The only link between these two fields is that quantum cryptography protocols
   remain secure *even* if an attack has access to unbounded computational capabilities, including access to quantum
   computers.

   Quantum-proof strong randomness extractor remain secure even if the adversary has access to a quantum memory or a
   quantum computer.


----------------------
Key exchange protocols
----------------------

To communicate securely over an insecure channel we need to encrypt the messages. To the eyes of an
eavesdropper\ [#eavesdropper]_ these *ciphertexts* must look like random noise, so that no information about the
original messages gets *leaked*.
Encryption requires secret keys, and to establish a secret key we need a secret channel. This sounds like
the chicken or the egg problem. So what can we do?

.. [#eavesdropper] In cryptography this eavesdropper is usually called Eve, while the honest parties are referred to as
   Alice and Bob.

If Alice wants to communicate with Bob at some time in the future they could both meet in person and exchange
secret keys. Later, they could use those keys to encrypt their communication. This solution solves the problem, but it
is not very practical. What if Alice wants to communicate with Charlie, who is very far away, and they cannot meet in
person to exchange some keys? And what if Alice wants to communicate with tens or hundreds of different people and not
just one?

In the late 70s this problem was solved with the invention of public-key cryptography\ [#publickey]_ and key exchange
methods based on it, such as `Diffie–Hellman–Merkle`_ and `RSA`_.

.. [#publickey] Public-key cryptography is also called *asymmetric* cryptography.

.. _Diffie–Hellman–Merkle: https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange
.. _RSA: https://en.wikipedia.org/wiki/RSA_(cryptosystem)

Nowadays, these, and also newer public-key protocols, are widely used to establish secure communications. However, they
all belong to the same category of *computationally secure* cryptography. What does this mean and, can we do any better?


------------------------------
Information-theoretic security
------------------------------

One of the main results highlighted in Shannon's seminal paper :cite:`shannon1948` is that to achieve perfect secrecy, that is,
encryption that leaks *absolutely* nothing\ [#perfectsecrecy]_ about the original message, we need secret keys
*at least* as long as the message we want to encrypt. As an example, the `one-time pad`_ requires keys exactly as long
as the messages and achieves perfect secrecy.\ [#leak]_

.. [#perfectsecrecy] One intuitive way of thinking about this is in terms of guessing probabilities. We say that a
   ciphertext leaks nothing about the message if, after observing the ciphertext, the best we can do is guess at random
   the original message.

.. [#leak] The one-time pad leaks the length of the plaintext. This is in general true for any perfect secrecy
   encryption scheme as it is not possible to obtain a uniform probability distribution on an infinite set. However,
   it can be avoided by using a proper protocol that fixes the length of all messages to the same size (by padding with
   zeros the small messages, for example) or by splitting the messages in blocks of fixed size.

.. _one-time pad: https://en.wikipedia.org/wiki/One-time_pad

The one-time pad is an example of an *information-theoretically secure* encryption method. On the one hand, we say that
a protocol is information-theoretically secure if its security can be *proven* without any assumptions about an
adversary's computational power and time. On the other hand, protocols that are *only* secure after assuming certain
limitations on the adversary's computational power are called *computationally secure*. In addition, computationally
secure protocols may rely on the *unproven* hardness of certain problems such as `factoring large numbers`_.

.. _factoring large numbers: https://en.wikipedia.org/wiki/Factorization

Going back to the key exchange methods we mentioned before, both Diffie–Hellman–Merkle and RSA are computationally
secure. Diffie–Hellman–Merkle relies on the hardness of the `discrete logarithm problem`_ and RSA relies on the hardness
of `factoring large numbers`_. These assumptions will no longer make sense when we have access to large enough quantum
computers, since the `Shor's algorithm`_ will allow to efficiently solve both problems.

.. _discrete logarithm problem: https://en.wikipedia.org/wiki/Discrete_logarithm
.. _Shor's algorithm: https://en.wikipedia.org/wiki/Shor%27s_algorithm

Unfortunately, there is no information-theoretically secure key exchange method using classical communication. And here
is where quantum physics comes to the rescue...


------------------------
Quantum Key Distribution
------------------------

If we don't limit ourselves to *only* classical communication, it is possible for two honest parties to agree on a
common secret key by using an insecure quantum channel and an authentic\ [#authentic]_ classical channel using QKD
protocols.

.. [#authentic] An attacker can see any messages sent over an authentic channel but it cannot modify them or impersonate
   any of the honest parties. It is possible to establish an authentic channel from an insecure channel using a small
   shared secret. Because of this, QKD should be seen as a *key expansion* protocol, rather than an agreement or
   distribution one.

Any QKD protocol\ [#QKD]_ always has two parts:

.. [#QKD] If you want to learn more about QKD you can check our `QKD for Engineers website`_.

.. _QKD for Engineers website: https://www.hslu.ch/en/lucerne-university-of-applied-sciences-and-arts/research/projects/detail/?pid=5814

1. a quantum part, where Alice and Bob exchange and measure quantum states over a quantum channel
2. a classical post-processing part, where Alice and Bob derive a secret key from the raw output of the quantum phase.

Here is a brief description of the quantum part of the first QKD protocol :cite:`BB84`.

.. admonition:: BB84 protocol (quantum part only)
   :class: hint

   1. Alice prepares a photon, polarized in one of the following angles: 0º, 45º, 90º or 135º. Polarizations
      corresponding to angles 0º and 90º are denoted as the horizontal basis. Polarizations 45º and 135º as the diagonal
      basis. Alice chooses the polarization at random and takes note of her choice, i.e. two bits, one bit corresponding
      to the basis choice, and one for the actual angle from that basis.
   2. She sends the photon to Bob over the untrusted quantum channel.
   3. Bob, also at random, chooses whether to measure the photon in the horizontal or diagonal basis, and takes note of
      his choice and the measurement result.
   4. They repeat the previous steps several times.

After the quantum part, Alice and Bob have two (classical) raw bit strings :math:`X_A` and :math:`X_B`. These bit
strings will generally not be equal, nor perfectly secret. That is why the second part is crucial. The classical
post-processing will

1. determine if Eve has learned too much about the raw key, and *abort* if that is the case
2. fix the errors and remove any side information Eve may still have.

The first part is done by sacrificing part of the raw bit strings and performing what is usually called
*parameter estimation* or *testing*. Alice and Bob will take a random sample, announce all the details about the quantum
part and determine if it is possible for them to agree on a secure key. They do this by basically counting errors.
If they observed higher discrepancies this could be due to a noiser channel, but in the worst-case, they have to assume
that an eavesdropper has learned some information about their remaining bit strings. If they estimate this leaked
information to be higher that some threshold, they abort the protocol to avoid agreeing on an insecure key.

The second part, assuming the protocol didn't abort, is done in two sequential steps:

1. *Information reconciliation* will fix the errors between :math:`X_A` and :math:`X_B` with high probability.\ [#prob]_
   This can be achieved by sending additional information over the authentic channel and applying some `syndrome decoding`_.
2. *Privacy amplification* will reduce the side information Eve may have about the bit strings.\ [#side]_ This side
   information is quantum because, in addition to eavesdropping the classical communication, Eve is free to do
   *anything* with the quantum states sent over the quantum channel. In particular, she can store them in a
   quantum memory.

.. [#prob] In QKD protocols this is tuned by specifying a particular value for the security parameter
   :math:`\epsilon_\text{corr}`-correctness.

.. [#side] The probability of successfully removing this side information is tuned by a different security parameter,
   usually denoted as :math:`\epsilon_\text{sec}`-secrecy.

.. _syndrome decoding: https://en.wikipedia.org/wiki/Decoding_methods#Syndrome_decoding


Privacy Amplification
---------------------

Roughly speaking, privacy amplification is the task of shrinking a partially secret string to a highly secret key.

To achieve this task, we can use randomness extractors. However, not all randomness extractors are suitable for the
privacy amplification step in QKD. We require that they are

1. *quantum-proof*, meaning that they are still secure in the presence of quantum side information, because an adversary
   could have a quantum memory\ [#memory]_, and
2. *strong*, which means that the output is independent of the seed used, because the seed is communicated over the
   classical channel and therefore known to the adversary.

.. [#memory] An adversary with a quantum memory could delay the read-out of the side information, i.e., perform a
   quantum measurement, until the optimal moment, which could be even after the protocol has finished.

The security of QKD protocols strongly depends on the correct implementation of the privacy amplification step.
Any bugs in this step may lead to partially insecure keys in the best case, and totally breaking the security
in the worst case. That is why it is crucial to have a reliable library to perform privacy amplification.
