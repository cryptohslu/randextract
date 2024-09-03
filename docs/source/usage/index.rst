.. py:module:: randextract
   :no-index:

=====
Usage
=====

The typical workflow with :obj:`randextract` is the following:

1. Decide what randomness extractor you want to use
2. Calculate the optimal parameters based on the weak randomness source and the acceptable error
3. Pick one particular family of extractor with fixed input and output lengths
4. Process the weak randomness, or
5. Verify a different implementation

The first step is something you have to decide before starting to write any code. The :ref:`theory section <Theory>` can
help you make an informed decision. Once you know what extractor you want to use, the second step can be done with the
help of :obj:`randextract` by using the :obj:`calculate_length()` method of any :obj:`RandomnessExtractor` class. Third
step is done by using the class method :obj:`create()` of :obj:`RandomnessExtractor` or by calling directly the
constructor of the implementation class (e.g. :obj:`ToeplitzHashing`). The actual randomness extraction is always done
by calling the :obj:`extract()` method of a :obj:`RandomnessExtractor`. And finally, testing and validating other
implementations using the class :obj:`Validator`. In the next section we show some practical examples using our package
to validate high-performance implementations used in current QKD experiments.

-------------
First example
-------------

First of all, if you haven't install :obj:`randextract`,

You can use :obj:`randextract` in your own Python script. The recommended way to create a
:ref:`randomness extractor <Seeded randomness extractors>` object is to use the method :code:`create()` of our factory
class :obj:`RandomnessExtractor`.

The first thing you need to decide is the type of extractor you want to use. Currently, we offer these three types:
:code:`toeplitz`, :code:`modified_toeplitz`, and :code:`trevisan`. For a summary of these implementations you can read
their docstrings in the :ref:`API Reference` page. For a more detailed explanation, as well as some examples, check
:ref:`the theory section <Theory>`.

.. code-block:: python

   import randextract
   from randextract import RandomnessExtractor

   ext = RandomnessExtractor.create(extractor_type="YOUR TYPE CHOICE", ...)

Once you know what extractor you want to use, you should read the required parameters from its implementation class.
For example, if you want to use a Toeplitz hashing extractor, check :obj:`ToeplitzHashing`. For a Trevisan's
construction, check :obj:`TrevisanExtractor`.

The simplest extractor requires, at least, that you specify

1. the length of the bit string from the weak random source,
2. a lower bound on the weak random source entropy,
3. and the tolerable error.

For example, the following code creates a Toeplitz hashing extractor that expects an input of 1 MiB from a weak random
and outputs around 512 KiB (up to an error :math:`10^{-6}`).

.. code-block:: python

   from galois import GF2

   import randextract
   from randextract import RandomnessExtractor, ToeplitzHashing

   MiB = 8 * 2**20

   optimal_output_length = ToeplitzHashing.calculate_length(
       extractor_type="quantum",
       input_length=MiB,
       relative_source_entropy=0.5,
       error_bound=1e-6,
   )

   ext = RandomnessExtractor.create(
       extractor_type="toeplitz", input_length=MiB, output_length=optimal_output_length
   )

   input_ext = GF2.Random(MiB)
   seed_ext = GF2.Random(ext.seed_length)

   extracted = ext.extract(input_ext, seed_ext)

The Trevisan's construction requires a weak design and a one-bit extractor, each of which accepts a set of parameters.
For example, the following code creates a Trevisan's construction using the polynomial one-bit extractor and the finite
field polynomial weak design.

.. code-block:: python

   import randextract
   from randextract import RandomnessExtractor
   from galois import GF2

   ext = RandomnessExtractor.create(
      extractor_type="trevisan",
      weak_design_type="finite_field",
      one_bit_extractor_type="polynomial",
      one_bit_extractor_seed_length=1024,
      input_length=2**20,
      output_length=2**10,
   )

   input_ext = GF2.Random(ext.input_length)
   seed_ext  = GF2.Random(ext.seed_length)

   extracted = ext.extract(input_ext, seed_ext)
