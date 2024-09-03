=======================
Two-universal functions
=======================

So far we have *only* given mathematically definitions of functions, the randomness extractors, that fulfill certain
conditions. To show that the previous sections are not just a collection of mathematically convenient definitions, in
this section we define a class of functions with well-known explicit constructions since the 70s that satisfy the
definition of quantum-proof strong randomness extractors: the class of *two-universal functions*.\ [#twouniversal]_

.. [#twouniversal] Originally denoted as universal\ :math:`_2` functions.

.. admonition:: Two-universal functions
   :class: note

   This class of functions was originally studied in the context of input independent average
   linear time computation. Only later they were found useful for cryptographic tasks.

We say that that a family of functions :math:`\mathcal{F}`, such that :math:`f:\mathcal{X}\rightarrow\mathcal{Z}`, is
*two-universal* if :math:`\Pr_f[f(x)=f(x')]\leq\frac{1}{|\mathcal{Z}|}` for any distinct :math:`x,x'\in\mathcal{X}`, and
:math:`f\in\mathcal{F}` chosen uniformly at random.

Carter and Wegman :cite:`CW77,WC81` showed explicit constructions of this class of functions. That is why we know that
for any integers :math:`0 < m\leq n` we can always find a family of two-universal functions from :math:`\{0,1\}^n` to
:math:`\{0,1\}^m`.

A randomness extractor, as defined in :ref:`the previous section <Seeded randomness extractors>`, is obtained from a
family of two-universal functions in the following way

1. A uniform seed :math:`y` is used to select a particular function :math:`f_y\in\mathcal{F}` from the family
2. The output of the extractor is obtained by applying this function to the input, i.e.,

.. math::
   \text{Ext}(x,y) = f_y(x)


---------------------------
Quantum leftover hash lemma
---------------------------

In the late 80s it was already proven that families of two-universal functions can be used in cryptographic applications
and, in particular, in the task of randomness extraction. However, in the original research only classical side
information was considered. Later, it was proved a quantum version of the leftover hash lemma :cite:`RK05,TSTR11`,
generalizing the original result and showing that these functions can be used also against quantum side information.

The quantum leftover hash lemma states the following: For a random variable :math:`X`, quantum side information
:math:`E`, and a family of two-universal functions :math:`\mathcal{F}` from :math:`\{0,1\}^n` to :math:`\{0,1\}^m`,
on average over the choices of :math:`f` from :math:`\mathcal{F}`, the output :math:`f(X)` is :math:`\epsilon`-close to
uniform conditioned on :math:`E`, where :math:`\epsilon` is given by

.. math::
   \epsilon = \frac{1}{2}\sqrt{2^{m-H_\text{min}(X|E)}}\,.

In other words, and going back to our quantum-proof randomness extractor, if we know that our weak source of randomness
has a lower bound on its quantum conditional min-entropy, i.e., :math:`H_\text{min}(X|E)\geq k`, and we want our QKD
protocol to be :math:`\epsilon_\text{sec}`-secret, then privacy amplification can be achieved by choosing a proper
family of two-universal functions with

.. math::
   m = \Big\lfloor k + 2 - 2\log\frac{1}{\epsilon_\text{sec}} \Big\rfloor\,.

Notice that, as we defined the statistical distance in the :ref:`previous section <Seeded randomness extractors>`,
the value of :math:`\epsilon_\text{sec}`-secret could, in principle, be in the range :math:`[0,1]`. However, the lemma
only works for :math:`\epsilon>0`, and because it is not possible to extract more bits than the amount of bits we input,
even in the extreme case :math:`m=n=H_\text{min}(X|E)`, we obtain that
:math:`\epsilon_\text{sec}\text{-secret}=\frac{1}{2}`. This extreme case represents a situation where the weak random
source is already uniform from the adversary's point of view, so we don't need an extractor to begin with. Therefore,
for practical implementations that make use of this lemma to compute the output length, the value of
:math:`\epsilon_\text{sec}`-secret should be in the range :math:`(0, \tfrac{1}{2}\sqrt{2^{m-n}})`.
