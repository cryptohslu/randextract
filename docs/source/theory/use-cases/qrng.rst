==========================================
Use case: Quantum Random Number Generators
==========================================

:ref:`The previous use case <Use case: Quantum Key Distribution>` showed that quantum physics can be used to achieve
something that is impossible classically: agree on a secure key over a public insecure channel. But remember that in the
QKD protocol we presented, both Alice and Bob had to choose at random some values. What if these values are not truly
random? Then, the security of QKD can be totally broken. That is why access to good randomness is crucial in any QKD
protocol.

PRNGs are not good enough if we want to agree on a information-theoretically secure key using a QKD protocol.
Fortunately, quantum physics also allows for the creation of devices whose output is provably random. This was
introduced briefly while :ref:`motivating randomness extractors <Imperfect randomness and extractors>` but here we will
explain this in more detail. Feel free to jump to :ref:`the next section of the theory <Seeded randomness extractors>`
if you don't want to learn more about QRNGs.


--------------------------------
Quantum Random Number Generators
--------------------------------

Quantum Random Number Generators (QRNGs) are devices that take advantage of quantum physics to generate truly provable
random bits. This is possible because randomness and unpredictability are inherent to quantum phenomena. Even if we know
absolutely everything about a quantum experiment, the outcome of a measurement is not deterministic in general. We do
not observe this in our day-to-day life\ [#dice]_, but this surprising quantum property has been shown in the lab in
thousands of different experiments since the early developments of the theory.

.. [#dice] For example, if we record the throw of a dice with a super fast camera, it is possible to predict its value
   before it stops moving. If we don't have such camera we say that the dice is "random", but this randomness is only an
   illusion and a manifestation of our ignorance about the state of the dice when we throw it. Quantum randomness is not
   the same kind of randomness as a dice.

QRNGs are a type of `hardware random number generators`_ (HRNG) since they use a physical process to generate randomness.
However, currently, most HRNGs use physical processes that are not quantum. Since classical physics is deterministic,
these HRNGs based on non-quantum phenomena always need to make an assumption that the initial state of the system is
only partially known.

.. _hardware random number generators: https://en.wikipedia.org/wiki/Hardware_random_number_generator

.. admonition:: QRNG protocol (quantum part only)
   :class: hint

   1. Alice prepares a photon polarized in the horizontal basis and in the state :math:`\ket{0}`.
   2. Alice measures the photon in the diagonal basis. The outcome of the measurement is totally random, with a
      50-50 chance of getting any of the diagonal polarized basis states, i.e., :math:`\ket{+}` or :math:`\ket{-}`.


----------------------------------
Privacy amplification still needed
----------------------------------

The above simplified QRNG protocol should, in theory, output truly random bits. However, running this kind of quantum
experiment is very challenging. It is hard to prepare and measure single photons accurately. In practice, these
experimental errors result in a certain bias in the outcome of the experiment. This can be fixed using randomness
extractors, which not only can remove this bias but also any side-information that an adversary may have about the
device.

However, it is interesting to emphasize that different use cases may have different requirements or constraints, and
therefore, some randomness extractors may be better suited for certain scenarios than others. This is the main
motivation to implement different families of extractors in the :obj:`randextract` package. For example, in the QRNG
scenario, we may want to minimize the required seed to maximize the net randomness output. But in a QKD scenario, it is
perhaps more important to be able to run the extractor as fast as possible, rather than trying to minimize the usage of
local randomness. This will be further expanded upon when we explain and describe the different extractors available
in the package.
