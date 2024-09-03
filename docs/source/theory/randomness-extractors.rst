============================
Seeded randomness extractors
============================

We define a *seeded randomness extractor* as a function

.. math::
   \text{Ext}\,:\,\{0,1\}^n\times\{0,1\}^d\rightarrow\{0,1\}^m

that takes a weak random source :math:`X`, a bit string of length :math:`n`, and a uniform seed :math:`Y`, a bit
string of length :math:`d`, and outputs :math:`\text{Ext}(X,Y)`, a bit string of length :math:`m` that is *almost*
uniform, given that the original string had sufficiently high entropy.

.. admonition:: Notation
   :class: note

   Throughout this documentation we use the letters :math:`X`, :math:`Y` and :math:`Z` to denote classical random
   variables, while other letters like :math:`A`, :math:`B` or :math:`E` denote quantum information.

Here the term "*almost* uniform"  refers to the statistical distance with respect to the uniform probability
distribution. In particular, if the output of the extractor is characterized by the probability
distribution :math:`P_Z`, and :math:`U_Z` denotes the uniform probability distribution\ [#uniform]_, we say that the
output of the extractor is :math:`\epsilon`-close to uniform if

.. [#uniform] Remember that in a uniform probability distribution each event is equally likely.

.. math::
   \frac{1}{2}\sum_{z\in\{0,1\}^m} \left|p_Z(z)-U_Z(z)\right|=\frac{1}{2} \sum_{z\in\{0,1\}^m} \left|p_Z(z)-\frac{1}{2^m}\right|\leq \epsilon\;.

Since we will be working with quantum side information, it will be convenient to express this statistical distance in
terms of the trace distance between two quantum states.\ [#classical]_ In particular, if we denote
:math:`\rho_{\text{Ext}(X,Y)}` the state representing the output of the extractor, and :math:`\sigma_U` is the fully
mixed state on a system of dimension :math:`2^m`, then the previous equation is equivalent to

.. [#classical] Any classical information, like our bit strings, can be expressed as a diagonal density matrix operator
   using the formalism of Quantum Information Theory. For more details about this, you can check any textbook about
   Quantum Information Theory such as this one :cite:`nielsen-chuang` or this more recent one :cite:`renes`.

.. math::
   d(\rho_{\text{Ext}(X,Y)}, \sigma_U):=\frac{1}{2}\|\rho_{\text{Ext}(X,Y)}-\sigma_U\|_\text{tr}\leq\epsilon\,,

where :math:`\|A\|_\text{tr}:=\text{tr}\sqrt{A^\dagger A}` is the trace norm.

We say that an extractor is strong if the output is independent of the seed :math:`Y`. This can be enforced by adding
the seed to the trace distance condition

.. math::
   d(\rho_{\text{Ext}(X,Y)Y}, \sigma_U):=\frac{1}{2}\|\rho_{\text{Ext}(X,Y)}-\sigma_U\otimes\rho_Y\|_\text{tr}\leq\epsilon\,.

This allows to extract randomness even if the seed becomes known to an adversary, such as in
:ref:`QKD <Use case: Quantum Key Distribution>`.

With all this information we are ready to define strong randomness extractors. The only missing part is introducing a
new parameter :math:`k`, which will bound *how bad* (or good) the weak random source actually is.\ [#boundweakrandomness]_
This parameter governs how long the resulting string can be or, in oder words, how much the input needs to be compressed.
Intuitively, one has to "pay" a price for improving the randomness: a shorter output length.

The correct information-theoretic measure for this is the min-entropy, which is directly related to the
guessing probability

.. [#boundweakrandomness] To understand why this is important, you can think of the extreme case where the weak random
   source always outputs the all-zeros (or all-ones) :math:`n`-bit string. It is not possible to extract
   any randomness from this source since there is no randomness at all. This should be taken into account in the
   definition of a randomness extractor.

.. math::
   \text{H}_\text{min}(X):=-\log \max_x p_X(x)=:-\log p_\text{guess}\,.

This definition is classical because it only involves the classical random variable :math:`X`. Later, we will generalize
this definition to include the fact that an adversary may have some side-information about :math:`X` and this
information may be quantum.


---------------------------
Strong randomness extractor
---------------------------

We define a :math:`(k,\epsilon)`-strong randomness extractor as a function

.. math::
   \text{Ext}\,:\,\{0,1\}^n\times\{0,1\}^d\rightarrow\{0,1\}^m

that outputs an :math:`m`-bit bit string that is :math:`\epsilon`-close from the ideal output, a uniform :math:`m`-bit
string independent of the seed :math:`Y`, as long as this seed is uniform and the min-entropy of the weak random source
:math:`X` is lower bounded :math:`H_\text{min}(X)\geq k`.


-----------------------------------------
Quantum-proof strong randomness extractor
-----------------------------------------

When considering adversarial scenarios, we must take into account the side-information that an adversary may have about
the weak random source since this knowledge affects the guessing probability. Randomness extractors that are useful even
when an adversary has classical side-information are called *classical-proof extractors*. If they can also be used when
the side-information is quantum, then they are called *quantum-proof extractors*.

To properly define quantum-proof strong extractors we have to introduce the conditional min-entropy.

.. math::
   H_\text{min}(X|E):=-\log \sum_x p_X(x)\text{tr}\left[\rho_x E_x\right]=:-\log p_\text{guess}\,.

Let's go through this new expression in detail. The quantum side-information is denoted by the system :math:`E`. The
best an attacker can do to learn the value :math:`x` of the random variable :math:`X` is measuring this system with an
optimal measurement, which mathematically we denote with the POVM\ [#povm]_ :math:`\{E_x\}_x`, and make a guess based
on the outcome of this measurement. For each possible value :math:`x`, the side-information is in some conditional
state :math:`\rho_x`. The measurement postulate of quantum mechanics tells us how to calculate the probability of
getting the correct guess for each :math:`x`.

.. [#povm] A POVM, or positive operator-valued measurement, is the most general description of a quantum measurement.

.. math::
   p_\text{correct guess}=\text{tr}\left[\rho_x E_x\right]\,.

Finally, we average over all possible values of :math:`X` to get the new guessing probability

.. math::
   p_\text{guess}:=\sum_x p_X(x)\text{tr}\left[\rho_x E_x\right]\,.

We define a quantum-proof :math:`(k,\epsilon)`-strong randomness extractor as a function

.. math::
   \text{Ext}\,:\,\{0,1\}^n\times\{0,1\}^d\rightarrow\{0,1\}^m\,,

whose output is :math:`\epsilon`-close to the ideal output in terms of the trace distance

.. math::
   \frac{1}{2}\|\rho_{\text{Ext}(X,Y)YE}-\sigma_{U}\otimes\rho_Y\otimes\rho_E\|_\text{tr}\leq\epsilon\,,

as long as the seed :math:`Y` is uniform and the conditional min-entropy of the weak random source is lower bounded
by :math:`H_\text{min}(X|E)\geq k`.

Quantum-proof strong randomness extractors are also classical-proof, and, for course, also satisfy the definition of
strong randomness extractors without side-information. They are the correct extractors for applications where an
attacker may be quantum such as in :ref:`QKD <Use case: Quantum Key Distribution>` and
:ref:`QRNG <Use case: Quantum Random Number Generators>`.