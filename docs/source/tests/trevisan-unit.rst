==================================
Unit tests Trevisan's construction
==================================


-----------------------------
Unit tests one-bit extractors
-----------------------------


Unit tests XOR one-bit extractor
================================


TestXOROneBitExtractorInitialization
------------------------------------


Unit tests polynomial one-bit extractor
=======================================

The polynomial one-bit extractor, as explained in :ref:`the theory section <Polynomial hashing one-bit extractor>` is
actually a concatenation of two hash functions. That is why here we show unit tests for these two hash functions
separately.


TestReedSolomonHashing
----------------------

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestReedSolomonHashing.test_l_1

.. math::
   :nowrap:

   \begin{alignat}{3}
   x &= \begin{bmatrix} 0 & 0 \end{bmatrix}, \quad \alpha &&= [0], \quad p_\alpha(x) &&&= 0\cdot0 + 0 = [0] \\
   x &= \begin{bmatrix} 0 & 1 \end{bmatrix}, \quad \alpha &&= [0], \quad p_\alpha(x) &&&= 0\cdot0 + 1 = [1] \\
   x &= \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad \alpha &&= [0], \quad p_\alpha(x) &&&= 1\cdot0 + 0 = [0] \\
   x &= \begin{bmatrix} 1 & 1 \end{bmatrix}, \quad \alpha &&= [0], \quad p_\alpha(x) &&&= 1\cdot0 + 1 = [1] \\
   x &= \begin{bmatrix} 0 & 0 \end{bmatrix}, \quad \alpha &&= [1], \quad p_\alpha(x) &&&= 0\cdot1 + 0 = [0] \\
   x &= \begin{bmatrix} 0 & 1 \end{bmatrix}, \quad \alpha &&= [1], \quad p_\alpha(x) &&&= 0\cdot1 + 1 = [1] \\
   x &= \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad \alpha &&= [1], \quad p_\alpha(x) &&&= 1\cdot1 + 0 = [1] \\
   x &= \begin{bmatrix} 1 & 1 \end{bmatrix}, \quad \alpha &&= [1], \quad p_\alpha(x) &&&= 1\cdot1 + 1 = [0] \\
   \end{alignat}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestReedSolomonHashing.test_l_4

The arithmetic of extended finite fields is not unique, but determined by the choice of an irreducible polynomial. Our
library, by default, uses the irreducible polynomial with the least number of nonzero terms. For these unit tests, the
irreducible polynomial is :math:`\gamma^4+\gamma+1`.

.. math::
   :nowrap:

   \begin{align}
   x &=
   \begin{bmatrix}
   0 & 0 & 0 & 1 & 0 & 1 & 0 & 1
   \end{bmatrix} = [1, \gamma^2 + 1], \\
   \alpha &=
   \begin{bmatrix}
   0 & 1 & 1 & 1
   \end{bmatrix} = \gamma^2 + \gamma + 1, \\
   p_\alpha(x) &= 1\cdot(\gamma^2 + \gamma + 1) + (\gamma^2 + 1) = \gamma =
   \begin{bmatrix}
   0 & 0 & 1 & 0
   \end{bmatrix}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   x &=
   \begin{bmatrix}
   1 & 0 & 1 & 0 & 0 & 1 & 1 & 1
   \end{bmatrix} = [\gamma^3 + \gamma, \gamma^2 + \gamma + 1], \\
   \alpha &=
   \begin{bmatrix}
   0 & 1 & 0 & 1
   \end{bmatrix} = \gamma^2 + 1, \\
   p_\alpha(x) &= (\gamma^3 + \gamma)(\gamma^2 + 1) + (\gamma^2 + \gamma + 1) = \gamma^5+\gamma^2+1 \\
   &= \gamma(\gamma^4+\gamma+1) + (\gamma + 1) = \gamma + 1 =
   \begin{bmatrix}
   0 & 0 & 1 & 1
   \end{bmatrix}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   x &=
   \begin{bmatrix}
   0 & 1 & 0 & 1 & 0 & 0 & 0 & 0
   \end{bmatrix} = [\gamma^2 + 1, 0], \\
   \alpha &=
   \begin{bmatrix}
   1 & 0 & 0 & 1
   \end{bmatrix} = \gamma^3 + 1, \\
   p_\alpha(x) &= (\gamma^2 + 1)(\gamma^3 + 1) + 0 = \gamma^5+\gamma^3+\gamma^2+1 \\
   & = \gamma(\gamma^4+\gamma+1) + (\gamma^3+\gamma+1) = \gamma^3+\gamma+1 =
   \begin{bmatrix}
   1 & 0 & 1 & 1
   \end{bmatrix}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   x &=
   \begin{bmatrix}
   0 & 0 & 0 & 1 & 1 & 1 & 1 & 1
   \end{bmatrix} = [1, \gamma^3 + \gamma^2 + \gamma + 1], \\
   \alpha &=
   \begin{bmatrix}
   1 & 1 & 1 & 1
   \end{bmatrix} = \gamma^3 + \gamma^2 + \gamma, \\
   p_\alpha(x) &= \gamma^3 + \gamma^2 + \gamma + \gamma^3 + \gamma^2 + \gamma + 1 = 1 =
   \begin{bmatrix}
   0 & 0 & 0 & 1
   \end{bmatrix}
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestReedSolomonHashing.test_correctness_example_1

This is :ref:`the example <example-polynomial-hashing>` that was done in detail in the theory section.

..
   .. literalinclude:: /../../tests/unit/trevisan/one_bit_extractor/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestReedSolomonHashing.test_correctness_example_1_custom_poly

This unit test uses the same input and seed but a different irreducible polynomial for the arithmetic calculations.
In particular, it uses the `Conway polynomial`_ for :math:`\text{GF}(2^{12})`:
:math:`\gamma^{12} + \gamma^7 + \gamma^6 + \gamma^5 + \gamma^3 + \gamma + 1`.

.. _Conway polynomial: https://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/CP2.html

Calculations for this test will be done in detail using the polynomial representation of the finite field. This is the
most transparent way of looking at these calculations when dealing with `extended finite fields`_, since the effect of
the irreducible polynomial is explicitly shown in the calculations. However, calculations quickly get tedious as the
size of the finite field increases. For this reason, the remaining long test cases we will be done using the integer
representation of the field.

.. _extended finite fields: https://en.wikipedia.org/wiki/Finite_field#Non-prime_fields

Let's compute each of the polynomials that appear in the expression of :math:`p_\alpha(x)`:

.. math::
   :nowrap:

   \begin{align}
   \alpha^2 &= (\gamma^9+\gamma^6+\gamma^5+\gamma^4+\gamma+1)^2\\ &= \gamma^{18}+\gamma^{15}+\gamma^{14}+\gamma^{13}+
   \gamma^{10}+\gamma^9+\gamma^{15}+\gamma^{12}+\gamma^{11}+\gamma^{10}+\gamma^7+\gamma^6\\
   &\quad +\, \gamma^{14}+\gamma^{11}+\gamma^{10}+\gamma^9+\gamma^6+\gamma^5+\gamma^{13}+\gamma^{10}+\gamma^9+\gamma^8
   +\gamma^5+\gamma^4\\
   &\quad +\, \gamma^{10}+\gamma^7+\gamma^6+\gamma^5+\gamma^2+\gamma+\gamma^9+\gamma^6+\gamma^5+\gamma^4+\gamma+1\\
   &=\gamma^{18}+\gamma^{12}+\gamma^{10}+\gamma^8+\gamma^2+1
   \end{align}

We are not done yet. The above polynomial is reducible in this finite field, i.e. we can express it as a product of some
quotient and the irreducible polynomial we have chosen plus some remainder, which is the irreducible expression we are
interested in.

.. math::
   :nowrap:

   \begin{align}
   \gamma^{18}&+\gamma^{12}+\gamma^{10}+\gamma^8+\gamma^2+1 = \\&(\gamma^6+\gamma)(\gamma^{12} + \gamma^7 + \gamma^6 +
   \gamma^5 + \gamma^3 + \gamma + 1) + (\gamma^{11}+\gamma^{10}+\gamma^9+\gamma^4+\gamma+1)
   \end{align}

Therefore,

.. math::
   \alpha^2 = \gamma^{11}+\gamma^{10}+\gamma^9+\gamma^4+\gamma+1\,.

Similarly, we can compute all the terms that appear in :math:`p_\alpha(x)` and the final hash:

.. math::
   :nowrap:

   \begin{gather}
   x_0\alpha^2=(\gamma^{11}+\gamma^9+\gamma^7+\gamma^6+\gamma^5+\gamma^4+\gamma^3+\gamma)(\gamma^{11}+\gamma^{10}+
   \gamma^9+\gamma^4+\gamma+1) = \gamma^9 + \gamma^7 + \gamma^2\\
   \begin{aligned}
   x_1\alpha &= (\gamma^{10}+\gamma^9+\gamma^8+\gamma^5+\gamma^4)(\gamma^{11}+\gamma^{10}+\gamma^9+\gamma^4+\gamma+1)\\
   &= \gamma^{11}+\gamma^{10}+\gamma^9+\gamma^7+\gamma^5+\gamma^4+\gamma^3+\gamma
   \end{aligned}\\
   \begin{aligned}
   p_\alpha(x)&=x_0\alpha^2+x_1\alpha+x_2 = (\gamma^9 + \gamma^7 + \gamma^2) + (\gamma^{11}+\gamma^{10}+\gamma^9+\gamma^7+
   \gamma^5+\gamma^4+\gamma^3+\gamma)\\
   &= \gamma^{11}+\gamma^{10}+\gamma^5+\gamma^4+\gamma^3+\gamma^2+\gamma = \begin{bmatrix}
   1 & 1 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 0
   \end{bmatrix}\,.
   \end{aligned}
   \end{gather}

..
   .. literalinclude:: /../../tests/unit/trevisan/one_bit_extractor/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestReedSolomonHashing.test_correctness_example_2

This example uses, again, :math:`\gamma^{12} + \gamma^3 + 1` as the irreducible polynomial of :math:`\text{GF}(2^{12})`.

.. math::
   :nowrap:

   \begin{align}
   x &= \begin{bmatrix}
   1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 1 & 1 & 0 & 1 & 1 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 1
   \end{bmatrix} \\ &= [3653, 2856, 2048] \\
   \alpha &= \begin{bmatrix}
   1 & 1 & 0 & 0 & 1 & 1 & 0 & 1 & 0 & 1 & 1 & 0
   \end{bmatrix} = 3286\\
   p_\alpha(x) &= x_0\alpha^2+x_1\alpha+x_2 = 3653*3286^2+2856*3286+2048 = 1460 + 2128 + 2048 = 1508 \\
   &= \begin{bmatrix}
   0 & 1 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix}
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestReedSolomonHashing.test_correctness_example_2_custom_poly

Same input and seed but with :math:`\gamma^{12} + \gamma^7 + \gamma^6 + \gamma^5 + \gamma^3 + \gamma + 1` as irreducible
polynomial.

.. math::
   :nowrap:

   \begin{align}
   p_\alpha(x) &= x_0\alpha^2+x_1\alpha+x_2 = 3653*3286^2+2856*3286+2048 = 3709 + 510 + 2048 = 1923 \\
   &= \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 1 & 1
   \end{bmatrix}
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestReedSolomonHashing.test_correctness_example_3

Last unit cases involve a smaller finite field :math:`\text{GF}(2^6)`. Calculations are done explicitly using the
polynomial representation of the finite field.

.. math::
   :nowrap:

   \begin{align}
   \text{irr_poly} &= \gamma^6+\gamma+1 \\
   x &= \begin{bmatrix}
   1 & 0 & 1 & 0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0
   \end{bmatrix} =
   [\gamma^5+\gamma^3+\gamma, \gamma^5+\gamma^4+\gamma^3,\gamma^5] \\
   \alpha &= \begin{bmatrix}
   1 & 1 & 0 & 0 & 0 & 1
   \end{bmatrix} =
   \gamma^5+\gamma^4+1 \\
   \alpha^2 &= (\gamma^4+\gamma^2)(\gamma^6+\gamma+1)+(\gamma^5+\gamma^4+\gamma^3+\gamma^2+1)=\gamma^5+\gamma^4+
   \gamma^3+\gamma^2+1 \\
   x_0\alpha^2 &= (\gamma^4+\gamma^3)(\gamma^6+\gamma+1)+(\gamma^4+\gamma^3+\gamma)=\gamma^4+\gamma^3+\gamma \\
   x_1\alpha &= (\gamma^4+\gamma)(\gamma^6+\gamma+1)+(\gamma^3+\gamma^2+\gamma)=\gamma^3+\gamma^2+\gamma \\
   p_\alpha(x) &= x_0\alpha^2+x_1\alpha+x_2 = \gamma^5+\gamma^4+\gamma^2=
   \begin{bmatrix}
   1 & 1 & 0 & 1 & 0 & 0
   \end{bmatrix}
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestReedSolomonHashing.test_correctness_example_3_custom_poly

.. math::
   :nowrap:

   \begin{align}
   \text{irr_poly} &= \gamma^6 + \gamma^4 + \gamma^3 + \gamma + 1 \\
   x &= \begin{bmatrix}
   1 & 0 & 1 & 0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0
   \end{bmatrix} =
   [\gamma^5+\gamma^3+\gamma, \gamma^5+\gamma^4+\gamma^3,\gamma^5] \\
   \alpha &= \begin{bmatrix}
   1 & 1 & 0 & 0 & 0 & 1
   \end{bmatrix} =
   \gamma^5+\gamma^4+1 \\
   \alpha^2 &= (\gamma^4+\gamma)(\gamma^6+\gamma^4+\gamma^3+\gamma+1)+(\gamma^2+\gamma+1)=\gamma^2+\gamma+1 \\
   x_0\alpha^2 &= (\gamma+1)(\gamma^6+\gamma^4+\gamma^3+\gamma+1)+(\gamma^5+\gamma^4+\gamma^3+\gamma+1)=
   \gamma^5+\gamma^4+\gamma^3+\gamma+1 \\
   x_1\alpha &= (\gamma^4+\gamma^2+1)(\gamma^6+\gamma^4+\gamma^3+\gamma+1)+(\gamma^5+\gamma^4+\gamma^3+\gamma^2+
   \gamma+1) \\
   &= \gamma^5+\gamma^4+\gamma^3+\gamma^2+\gamma+1 \\
   p_\alpha(x) &= x_0\alpha^3+x_1\alpha+x_2 = \gamma^5+\gamma^2 =
   \begin{bmatrix}
   1 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix}
   \end{align}


TestHadamardHashing
-------------------

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestHadamardHashing.test_l_1

.. math::
   :nowrap:

   \begin{alignat}{3}
   p_\alpha(x) &= [0],\quad \beta &&= [0],\quad \text{Ext}_1 &&&= 0\cdot0 = 0 \\
   p_\alpha(x) &= [0],\quad \beta &&= [1],\quad \text{Ext}_1 &&&= 0\cdot1 = 0 \\
   p_\alpha(x) &= [1],\quad \beta &&= [0],\quad \text{Ext}_1 &&&= 1\cdot0 = 0 \\
   p_\alpha(x) &= [1],\quad \beta &&= [1],\quad \text{Ext}_1 &&&= 1\cdot1 = 1 \\
   \end{alignat}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestHadamardHashing.test_l_4

.. math::
   :nowrap:

   \begin{alignat}{3}
   p_\alpha(x) &=
   \begin{bmatrix}
   1 & 0 & 1 & 1
   \end{bmatrix},\quad \beta &&=
   \begin{bmatrix}
   1 & 1 & 0 & 0
   \end{bmatrix}
   ,\quad \text{Ext}_1 &&&= 1\cdot1 \oplus 0\cdot1 \oplus 1\cdot0 \oplus 1\cdot0 = 1 \\
   p_\alpha(x) &=
   \begin{bmatrix}
   1 & 1 & 0 & 0
   \end{bmatrix},\quad \beta &&=
   \begin{bmatrix}
   0 & 0 & 1 & 1
   \end{bmatrix}
   ,\quad \text{Ext}_1 &&&= 1\cdot0 \oplus 1\cdot0 \oplus 0\cdot1 \oplus 0\cdot1 = 0 \\
   p_\alpha(x) &=
   \begin{bmatrix}
   1 & 1 & 1 & 1
   \end{bmatrix},\quad \beta &&=
   \begin{bmatrix}
   1 & 1 & 0 & 1
   \end{bmatrix}
   ,\quad \text{Ext}_1 &&&= 1\cdot1 \oplus 1\cdot1 \oplus 1\cdot0 \oplus 1\cdot1 = 1 \\
   p_\alpha(x) &=
   \begin{bmatrix}
   1 & 1 & 0 & 1
   \end{bmatrix},\quad \beta &&=
   \begin{bmatrix}
   1 & 0 & 0 & 0
   \end{bmatrix}
   ,\quad \text{Ext}_1 &&&= 1\cdot1 \oplus 1\cdot0 \oplus 0\cdot0 \oplus 1\cdot0 = 1
   \end{alignat}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestHadamardHashing.test_correctness_example_1

.. math::
   :nowrap:

   \begin{align}
   p_\alpha(x) =
   &\begin{bmatrix}
   1 & 0 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 0 & 0 & 0
   \end{bmatrix} \\
   \beta =
   &\begin{bmatrix}
   1 & 0 & 0 & 1 & 0 & 0 & 1 & 0 & 1 & 1 & 1 & 1
   \end{bmatrix} \\
   \text{Ext}_1 = \bigoplus
   &\begin{bmatrix}
   1 & 0 & 0 & 1 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0
   \end{bmatrix} = 0
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestHadamardHashing.test_correctness_example_1_custom_poly

.. math::
   :nowrap:

   \begin{align}
   p_\alpha(x) =
   &\begin{bmatrix}
   1 & 1 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 0
   \end{bmatrix} \\
   \beta =
   &\begin{bmatrix}
   1 & 0 & 0 & 1 & 0 & 0 & 1 & 0 & 1 & 1 & 1 & 1
   \end{bmatrix} \\
   \text{Ext}_1 = \bigoplus
   &\begin{bmatrix}
   1 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 1 & 1 & 1 & 0
   \end{bmatrix} = 1
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestHadamardHashing.test_correctness_example_2

.. math::
   :nowrap:

   \begin{align}
   p_\alpha(x) =
   &\begin{bmatrix}
   0 & 1 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   \beta =
   &\begin{bmatrix}
   0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   \text{Ext}_1 = \bigoplus
   &\begin{bmatrix}
   0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} = 1
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestHadamardHashing.test_correctness_example_2_custom_poly

.. math::
   :nowrap:

   \begin{align}
   p_\alpha(x) =
   &\begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 1 & 1
   \end{bmatrix} \\
   \beta =
   &\begin{bmatrix}
   0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   \text{Ext}_1 = \bigoplus
   &\begin{bmatrix}
   0 & 1 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0
   \end{bmatrix} = 1
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestHadamardHashing.test_correctness_example_3

.. math::
   :nowrap:

   \begin{align}
   p_\alpha(x) =
   &\begin{bmatrix}
   1 & 1 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   \beta =
   &\begin{bmatrix}
   0 & 1 & 1 & 0 & 0 & 1
   \end{bmatrix} \\
   \text{Ext}_1 = \bigoplus
   &\begin{bmatrix}
   0 & 1 & 0 & 0 & 0 & 0
   \end{bmatrix} = 1
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestHadamardHashing.test_correctness_example_3_custom_poly

.. math::
   :nowrap:

   \begin{align}
   p_\alpha(x) =
   &\begin{bmatrix}
   1 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   \beta =
   &\begin{bmatrix}
   0 & 1 & 1 & 0 & 0 & 1
   \end{bmatrix} \\
   \text{Ext}_1 = \bigoplus
   &\begin{bmatrix}
   0 & 0 & 0 & 0 & 0 & 0
   \end{bmatrix} = 0
   \end{align}


TestPolynomialOneBitExtractor
-----------------------------

These tests basically combine previous results, as the polynomial hashing is the concatenation of the Reed Solomon and
the Hadamard hashing.

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestPolynomialOneBitExtractor.test_correctness_example_1

.. math::
   :nowrap:

   \begin{align}
   x &=
   \begin{bmatrix}
   1 & 0 & 1 & 0 & 1 & 1 & 1 & 1 & 1 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0
   \end{bmatrix} \\
   y &=
   \begin{bmatrix}
   0 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 1 & 0 & 1 & 1 & 1 & 1
   \end{bmatrix} \\
   \alpha &=
   \begin{bmatrix}
   0 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 1
   \end{bmatrix} \\
   p_\alpha(x) &=
   \begin{bmatrix}
   1 & 0 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 0 & 0 & 0
   \end{bmatrix} \\
   \beta &=
   \begin{bmatrix}
   1 & 0 & 0 & 1 & 0 & 0 & 1 & 0 & 1 & 1 & 1 & 1
   \end{bmatrix} \\
   p_\alpha(x) \odot \beta &=
   \begin{bmatrix}
   1 & 0 & 0 & 1 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0
   \end{bmatrix} \\
   \text{Ext}_1 &= \bigoplus p_\alpha(x)\odot\beta = 0
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestPolynomialOneBitExtractor.test_correctness_example_1_custom_poly

.. math::
   :nowrap:

   \begin{align}
   x &=
   \begin{bmatrix}
   1 & 0 & 1 & 0 & 1 & 1 & 1 & 1 & 1 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0
   \end{bmatrix} \\
   y &=
   \begin{bmatrix}
   0 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 1 & 0 & 1 & 1 & 1 & 1
   \end{bmatrix} \\
   \alpha &=
   \begin{bmatrix}
   0 & 0 & 1 & 0 & 0 & 1 & 1 & 1 & 0 & 0 & 1 & 1
   \end{bmatrix} \\
   p_\alpha(x) &=
   \begin{bmatrix}
   1 & 1 & 0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 0
   \end{bmatrix} \\
   \beta &=
   \begin{bmatrix}
   1 & 0 & 0 & 1 & 0 & 0 & 1 & 0 & 1 & 1 & 1 & 1
   \end{bmatrix} \\
   p_\alpha(x) \odot \beta &=
   \begin{bmatrix}
   1 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 1 & 1 & 1 & 0
   \end{bmatrix} \\
   \text{Ext}_1 &= \bigoplus p_\alpha(x)\odot\beta = 1
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestPolynomialOneBitExtractor.test_correctness_example_2

.. math::
   :nowrap:

   \begin{align}
   x &=
   \begin{bmatrix}
   1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 1 & 1 & 0 & 1 & 1 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 1
   \end{bmatrix} \\
   y &=
   \begin{bmatrix}
   1 & 1 & 0 & 0 & 1 & 1 & 0 & 1 & 0 & 1 & 1 & 0 & 0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   \alpha &=
   \begin{bmatrix}
   1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 1
   \end{bmatrix} \\
   p_\alpha(x) &=
   \begin{bmatrix}
   0 & 1 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   \beta &=
   \begin{bmatrix}
   0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   p_\alpha(x) \odot \beta &=
   \begin{bmatrix}
   0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   \text{Ext}_1 &= \bigoplus p_\alpha(x)\odot\beta = 1
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestPolynomialOneBitExtractor.test_correctness_example_2_custom_poly

.. math::
   :nowrap:

   \begin{align}
   x &=
   \begin{bmatrix}
   1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 1 & 1 & 0 & 1 & 1 & 0 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 1
   \end{bmatrix} \\
   y &=
   \begin{bmatrix}
   1 & 1 & 0 & 0 & 1 & 1 & 0 & 1 & 0 & 1 & 1 & 0 & 0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   \alpha &=
   \begin{bmatrix}
   1 & 1 & 1 & 0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 1
   \end{bmatrix} \\
   p_\alpha(x) &=
   \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 1 & 1
   \end{bmatrix} \\
   \beta &=
   \begin{bmatrix}
   0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   p_\alpha(x) \odot \beta &=
   \begin{bmatrix}
   0 & 1 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0
   \end{bmatrix} \\
   \text{Ext}_1 &= \bigoplus p_\alpha(x)\odot\beta = 1
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestPolynomialOneBitExtractor.test_correctness_example_3

.. math::
   :nowrap:

   \begin{align}
   x &=
   \begin{bmatrix}
   1 & 0 & 1 & 0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0
   \end{bmatrix} \\
   y &=
   \begin{bmatrix}
   1 & 1 & 0 & 0 & 0 & 1 & 0 & 1 & 1 & 0 & 0 & 1
   \end{bmatrix} \\
   \alpha &=
   \begin{bmatrix}
   1 & 1 & 0 & 0 & 0 & 1
   \end{bmatrix} \\
   p_\alpha(x) &=
   \begin{bmatrix}
   1 & 1 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   \beta &=
   \begin{bmatrix}
   0 & 1 & 1 & 0 & 0 & 1
   \end{bmatrix} \\
   p_\alpha(x) \odot \beta &=
   \begin{bmatrix}
   0 & 1 & 0 & 0 & 0 & 0
   \end{bmatrix} \\
   \text{Ext}_1 &= \bigoplus p_\alpha(x)\odot\beta = 1
   \end{align}

..
   .. literalinclude:: /../../tests/unit/test_polynomial_one_bit_extractor.py
      :language: python
      :pyobject: TestPolynomialOneBitExtractor.test_correctness_example_3_custom_poly

.. math::
   :nowrap:

   \begin{align}
   x &=
   \begin{bmatrix}
   1 & 0 & 1 & 0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 0
   \end{bmatrix} \\
   y &=
   \begin{bmatrix}
   1 & 1 & 0 & 0 & 0 & 1 & 0 & 1 & 1 & 0 & 0 & 1
   \end{bmatrix} \\
   \alpha &=
   \begin{bmatrix}
   1 & 1 & 0 & 0 & 0 & 1
   \end{bmatrix} \\
   p_\alpha(x) &=
   \begin{bmatrix}
   1 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix} \\
   \beta &=
   \begin{bmatrix}
   0 & 1 & 1 & 0 & 0 & 1
   \end{bmatrix} \\
   p_\alpha(x) \odot \beta &=
   \begin{bmatrix}
   0 & 0 & 0 & 0 & 0 & 0
   \end{bmatrix} \\
   \text{Ext}_1 &= \bigoplus p_\alpha(x)\odot\beta = 0
   \end{align}


-----------------------
Unit tests weak designs
-----------------------


Unit tests finite field polynomial design
=========================================


TestFiniteFieldPolynomialDesignConstantPolynomials
--------------------------------------------------

This class tests the smallest possible weak design with this construction.

..
   .. literalinclude:: /../../tests/unit/trevisan/test_finite_field_polynomial_design.py
      :language: python
      :pyobject: TestFiniteFieldPolynomialDesignConstantPolynomials

The first method tests that the weak design is properly computed.

.. math::
   :nowrap:

   \begin{gather}
   m = 2\,,\quad t = 2\,,\quad d = t^2 = 4\,,\quad c = \Big\lceil \frac{\log 2}{\log 2} - 1\Big\rceil = 0 \\
   \alpha_0(0) = \Big\lfloor\frac{0}{1}\Big\rfloor\mod 2 = 0 \,,\quad
   \alpha_1(0) = \Big\lfloor\frac{1}{1}\Big\rfloor\mod 2 = 1 \\
   p_0(\gamma) = 0\,,\quad p_1(\gamma) = 1 \\
   S_0 = \Big\{\big(0, p_0(0)\big), \big(1, p_0(1)\big)\Big\} = \Big\{\big(0,0\big), \big(1,0\big)\Big\} =
   \big\{0, 1\big\} \\
   S_1 = \Big\{\big(0, p_1(0)\big), \big(1, p_1(1)\big)\Big\} = \Big\{\big(0,1\big), \big(1,1\big)\Big\} =
   \big\{2, 3\big\} \\
   W = \Big[\big\{0,1\big\}, \big\{2, 3\big\}\Big]
   \end{gather}

The remaining methods test how the weak design is used to split the input seed into the smaller seeds used by the
one-bit extractor.

.. math::
   :nowrap:

   \begin{alignat}{4}
   y &= \begin{bmatrix}0 & 0 & 0 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 0 & 0 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 1 & 0 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 1 & 0 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 0 & 1 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 0 & 1 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 1 & 1 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 1 & 1 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 0 & 0 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 0 & 0 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 1 & 0 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 1 & 0 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 0 & 1 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 0 & 1 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 1 & 1 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 1 & 1 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 1\end{bmatrix} \\
   \end{alignat}


TestFiniteFieldPolynomialDesignSmallestOutputLengthSmallestOutputSeedLength
---------------------------------------------------------------------------

..
   .. literalinclude:: /../../tests/unit/trevisan/test_finite_field_polynomial_design.py
      :language: python
      :pyobject: TestFiniteFieldPolynomialDesignSmallestOutputLengthSmallestOutputSeedLength

.. math::
   :nowrap:

   \begin{gather}
   m = 3\,,\quad t = 2\,,\quad d = t^2 = 4\,,\quad c = \Big\lceil \frac{\log 3}{\log 2} - 1\Big\rceil = 1 \\
   \alpha_0(0) = \Big\lfloor\frac{0}{1}\Big\rfloor\mod 2 = 0 \,,\quad
   \alpha_0(1) = \Big\lfloor\frac{0}{2}\Big\rfloor\mod 2 = 0 \\
   \alpha_1(0) = \Big\lfloor\frac{1}{1}\Big\rfloor\mod 2 = 1 \,,\quad
   \alpha_1(1) = \Big\lfloor\frac{1}{2}\Big\rfloor\mod 2 = 0 \\
   \alpha_2(0) = \Big\lfloor\frac{2}{1}\Big\rfloor\mod 2 = 0 \,,\quad
   \alpha_2(1) = \Big\lfloor\frac{2}{2}\Big\rfloor\mod 2 = 1 \\
   p_0(\gamma) = \alpha_0(0) + \alpha_0(1)\gamma = 0 \\
   p_1(\gamma) = \alpha_1(0) + \alpha_1(1)\gamma = 1 \\
   p_2(\gamma) = \alpha_2(0) + \alpha_2(2)\gamma = \gamma \\
   S_0 = \Big\{\big(0, p_0(0)\big), \big(1, p_0(1)\big)\Big\} = \Big\{\big(0,0\big), \big(1,0\big)\Big\} =
   \big\{0, 1\big\} \\
   S_1 = \Big\{\big(0, p_1(0)\big), \big(1, p_1(1)\big)\Big\} = \Big\{\big(0,1\big), \big(1,1\big)\Big\} =
   \big\{2, 3\big\} \\
   S_2 = \Big\{\big(0, p_2(0)\big), \big(1, p_2(1)\big)\Big\} = \Big\{\big(0,0\big), \big(1,1\big)\Big\} =
   \big\{0, 3\big\} \\
   W = \Big[\big\{0,1\big\}, \big\{2, 3\big\}, \big\{0,3\big\}\Big]
   \end{gather}

.. math::
   :nowrap:

   \begin{alignat}{5}
   y &= \begin{bmatrix}0 & 0 & 0 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 0\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}0 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 0 & 0 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 0\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}1 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 1 & 0 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 0\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}0 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 1 & 0 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 0\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}1 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 0 & 1 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 0\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}0 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 0 & 1 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 0\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}1 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 1 & 1 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 0\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}0 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 1 & 1 & 0\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 0\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}1 & 0\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 0 & 0 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 1\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}0 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 0 & 0 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 1\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}1 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 1 & 0 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 1\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}0 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 1 & 0 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}0 & 1\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}1 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 0 & 1 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 1\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}0 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 0 & 1 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 0\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 1\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}1 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}0 & 1 & 1 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}0 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 1\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}0 & 1\end{bmatrix} \\

   y &= \begin{bmatrix}1 & 1 & 1 & 1\end{bmatrix}\quad&&\Rightarrow\quad y_{S_0} &&&=
   \begin{bmatrix}1 & 1\end{bmatrix}\,,\quad y_{S_1} &&&&= \begin{bmatrix}1 & 1\end{bmatrix}
   \,,\quad y_{S_2} &&&&&= \begin{bmatrix}1 & 1\end{bmatrix} \\
   \end{alignat}


TestFiniteFieldPolynomialDesignMediumOutputLengthMediumOutputSeedLength
-----------------------------------------------------------------------

..
   .. literalinclude:: /../../tests/unit/trevisan/test_finite_field_polynomial_design.py
      :language: python
      :pyobject: TestFiniteFieldPolynomialDesignMediumOutputLengthMediumOutputSeedLength

This test is small enough to be doable by hand but already large enough to be quite tedious since we need to compute 60
different polynomials.

.. math::
   :nowrap:

   \begin{align}
   m &= 60\,,\quad t=5\,,\quad d=t^2=25\,,\quad c=\Big\lceil\frac{\log60}{\log5}-1\Big\rceil=2 \\
   \alpha_0(0) &= \Big\lfloor\frac{0}{1}\Big\rfloor\mod 5 = 0\,,\quad
   \alpha_0(1)  = \Big\lfloor\frac{0}{5}\Big\rfloor\mod 5 = 0\,,\quad
   \alpha_0(2)  = \Big\lfloor\frac{0}{25}\Big\rfloor\mod 5 = 0 \\
   p_0(\gamma) &= \alpha_0(0) + \alpha_0(1)\gamma + \alpha_0(2)\gamma^2 = 0 \\
   S_0 &= \Big\{\big(0,p_0(0)\big), \big(1,p_0(1)\big), \big(2,p_0(2)\big), \big(3,p_0(3)\big), \big(4,p_0(4)\big)\Big\} \\
       &= \Big\{\big(0,0\big), \big(1,0\big), \big(2,0\big), \big(3,0\big), \big(4,0\big)\Big\} = \big\{0,1,2,3,4\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_1(0) &= \Big\lfloor\frac{1}{1}\Big\rfloor\mod 5 = 1\,,\quad
   \alpha_1(1)  = \Big\lfloor\frac{1}{5}\Big\rfloor\mod 5 = 0\,,\quad
   \alpha_1(2)  = \Big\lfloor\frac{1}{25}\Big\rfloor\mod 5 = 0 \\
   p_1(\gamma) &= \alpha_1(0) + \alpha_1(1)\gamma + \alpha_1(2)\gamma^2 = 1 \\
   S_1 &= \Big\{\big(0,p_1(0)\big), \big(1,p_1(1)\big), \big(2,p_1(2)\big), \big(3,p_1(3)\big), \big(4,p_1(4)\big)\Big\} \\
       &= \Big\{\big(0,1\big), \big(1,1\big), \big(2,1\big), \big(3,1\big), \big(4,1\big)\Big\} = \big\{5,6,7,8,9\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_2(0) &= \Big\lfloor\frac{2}{1}\Big\rfloor\mod 5 = 2\,,\quad
   \alpha_2(1)  = \Big\lfloor\frac{2}{5}\Big\rfloor\mod 5 = 0\,,\quad
   \alpha_2(2)  = \Big\lfloor\frac{2}{25}\Big\rfloor\mod 5 = 0 \\
   p_2(\gamma) &= \alpha_2(0) + \alpha_2(1)\gamma + \alpha_2(2)\gamma^2 = 2 \\
   S_2 &= \Big\{\big(0,p_2(0)\big), \big(1,p_2(1)\big), \big(2,p_2(2)\big), \big(3,p_2(3)\big), \big(4,p_2(4)\big)\Big\} \\
   &= \Big\{\big(0,2\big), \big(1,2\big), \big(2,2\big), \big(3,2\big), \big(4,2\big)\Big\} = \big\{10,11,12,13,14\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_3(0) &= \Big\lfloor\frac{3}{1}\Big\rfloor\mod 5 = 3\,,\quad
   \alpha_3(1)  = \Big\lfloor\frac{3}{5}\Big\rfloor\mod 5 = 0\,,\quad
   \alpha_3(2)  = \Big\lfloor\frac{3}{25}\Big\rfloor\mod 5 = 0 \\
   p_3(\gamma) &= \alpha_3(0) + \alpha_3(1)\gamma + \alpha_3(2)\gamma^2 = 3 \\
   S_3 &= \Big\{\big(0,p_3(0)\big), \big(1,p_3(1)\big), \big(2,p_3(2)\big), \big(3,p_3(3)\big), \big(4,p_3(4)\big)\Big\} \\
   &= \Big\{\big(0,3\big), \big(1,3\big), \big(2,3\big), \big(3,3\big), \big(4,3\big)\Big\} = \big\{15,16,17,18,19\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_4(0) &= \Big\lfloor\frac{4}{1}\Big\rfloor\mod 5 = 4\,,\quad
   \alpha_4(1)  = \Big\lfloor\frac{4}{5}\Big\rfloor\mod 5 = 0\,,\quad
   \alpha_4(2)  = \Big\lfloor\frac{4}{25}\Big\rfloor\mod 5 = 0 \\
   p_4(\gamma) &= \alpha_4(0) + \alpha_4(1)\gamma + \alpha_4(2)\gamma^2 = 4 \\
   S_4 &= \Big\{\big(0,p_4(0)\big), \big(1,p_4(1)\big), \big(2,p_4(2)\big), \big(3,p_4(3)\big), \big(4,p_4(4)\big)\Big\} \\
   &= \Big\{\big(0,4\big), \big(1,4\big), \big(2,4\big), \big(3,4\big), \big(4,4\big)\Big\} = \big\{20,21,22,23,24\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_5(0) &= \Big\lfloor\frac{5}{1}\Big\rfloor\mod 5 = 0\,,\quad
   \alpha_5(1)  = \Big\lfloor\frac{5}{5}\Big\rfloor\mod 5 = 1\,,\quad
   \alpha_5(2)  = \Big\lfloor\frac{5}{25}\Big\rfloor\mod 5 = 0 \\
   p_5(\gamma) &= \alpha_5(0) + \alpha_5(1)\gamma + \alpha_5(2)\gamma^2 = \gamma \\
   S_5 &= \Big\{\big(0,p_5(0)\big), \big(1,p_5(1)\big), \big(2,p_5(2)\big), \big(3,p_5(3)\big), \big(4,p_5(4)\big)\Big\} \\
   &= \Big\{\big(0,0\big), \big(1,1\big), \big(2,2\big), \big(3,3\big), \big(4,4\big)\Big\} = \big\{0,6,12,18,24\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_6(0) &= \Big\lfloor\frac{6}{1}\Big\rfloor\mod 5 = 1\,,\quad
   \alpha_6(1)  = \Big\lfloor\frac{6}{5}\Big\rfloor\mod 5 = 1\,,\quad
   \alpha_6(2)  = \Big\lfloor\frac{6}{25}\Big\rfloor\mod 5 = 0 \\
   p_6(\gamma) &= \alpha_6(0) + \alpha_6(1)\gamma + \alpha_6(2)\gamma^2 = \gamma + 1\\
   S_6 &= \Big\{\big(0,p_6(0)\big), \big(1,p_6(1)\big), \big(2,p_6(2)\big), \big(3,p_6(3)\big), \big(4,p_6(4)\big)\Big\} \\
   &= \Big\{\big(0,1\big), \big(1,2\big), \big(2,3\big), \big(3,4\big), \big(4,0\big)\Big\} = \big\{5,11,17,23,4\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_7(0) &= \Big\lfloor\frac{7}{1}\Big\rfloor\mod 5 = 2\,,\quad
   \alpha_7(1)  = \Big\lfloor\frac{7}{5}\Big\rfloor\mod 5 = 1\,,\quad
   \alpha_7(2)  = \Big\lfloor\frac{7}{25}\Big\rfloor\mod 5 = 0 \\
   p_7(\gamma) &= \alpha_7(0) + \alpha_7(1)\gamma + \alpha_7(2)\gamma^2 = \gamma + 2\\
   S_7 &= \Big\{\big(0,p_7(0)\big), \big(1,p_7(1)\big), \big(2,p_7(2)\big), \big(3,p_7(3)\big), \big(4,p_7(4)\big)\Big\} \\
   &= \Big\{\big(0,2\big), \big(1,3\big), \big(2,4\big), \big(3,0\big), \big(4,1\big)\Big\} = \big\{10,16,22,3,9\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_8(0) &= \Big\lfloor\frac{8}{1}\Big\rfloor\mod 5 = 3\,,\quad
   \alpha_8(1)  = \Big\lfloor\frac{8}{5}\Big\rfloor\mod 5 = 1\,,\quad
   \alpha_8(2)  = \Big\lfloor\frac{8}{25}\Big\rfloor\mod 5 = 0 \\
   p_8(\gamma) &= \alpha_8(0) + \alpha_8(1)\gamma + \alpha_8(2)\gamma^2 = \gamma + 3\\
   S_8 &= \Big\{\big(0,p_8(0)\big), \big(1,p_8(1)\big), \big(2,p_8(2)\big), \big(3,p_8(3)\big), \big(4,p_8(4)\big)\Big\} \\
   &= \Big\{\big(0,3\big), \big(1,4\big), \big(2,0\big), \big(3,1\big), \big(4,2\big)\Big\} = \big\{15,21,2,8,14\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_9(0) &= \Big\lfloor\frac{9}{1}\Big\rfloor\mod 5 = 4\,,\quad
   \alpha_9(1)  = \Big\lfloor\frac{9}{5}\Big\rfloor\mod 5 = 1\,,\quad
   \alpha_9(2)  = \Big\lfloor\frac{9}{25}\Big\rfloor\mod 5 = 0 \\
   p_9(\gamma) &= \alpha_9(0) + \alpha_9(1)\gamma + \alpha_9(2)\gamma^2 = \gamma + 4\\
   S_9 &= \Big\{\big(0,p_9(0)\big), \big(1,p_9(1)\big), \big(2,p_9(2)\big), \big(3,p_9(3)\big), \big(4,p_9(4)\big)\Big\} \\
   &= \Big\{\big(0,4\big), \big(1,0\big), \big(2,1\big), \big(3,2\big), \big(4,3\big)\Big\} = \big\{20,1,7,13,19\big\}
   \end{align}

Remaining sets are shown using the following compact notation: :math:`\alpha_j:=[\alpha_j(0),\alpha_j(1),\alpha_j(2)]`.

.. math::
   :nowrap:

   \begin{align}
   \alpha_{10} &= [0, 2, 0]\,\quad p_{10}(\gamma) = 2\gamma \\
   S_{10} &= \Big\{\big(0,0\big), \big(1,2\big), \big(2,4\big), \big(3,1\big), \big(4,3\big)\Big\} = \big\{0,11,22,8,19\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{11} &= [1, 2, 0]\,\quad p_{11}(\gamma) = 2\gamma + 1\\
   S_{11} &= \Big\{\big(0,1\big), \big(1,3\big), \big(2,0\big), \big(3,2\big), \big(4,4\big)\Big\} = \big\{5,16,2,13,24\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{12} &= [2, 2, 0]\,\quad p_{12}(\gamma) = 2\gamma + 2\\
   S_{12} &= \Big\{\big(0,2\big), \big(1,4\big), \big(2,1\big), \big(3,3\big), \big(4,0\big)\Big\} = \big\{10,21,7,18,4\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{13} &= [3, 2, 0]\,\quad p_{13}(\gamma) = 2\gamma + 3\\
   S_{13} &= \Big\{\big(0,3\big), \big(1,0\big), \big(2,2\big), \big(3,4\big), \big(4,1\big)\Big\} = \big\{15,1,12,23,9\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{14} &= [4, 2, 0]\,\quad p_{14}(\gamma) = 2\gamma + 4\\
   S_{14} &= \Big\{\big(0,4\big), \big(1,1\big), \big(2,3\big), \big(3,0\big), \big(4,2\big)\Big\} = \big\{20,6,17,3,14\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{15} &= [0, 3, 0]\,\quad p_{15}(\gamma) = 3\gamma\\
   S_{15} &= \Big\{\big(0,0\big), \big(1,3\big), \big(2,1\big), \big(3,4\big), \big(4,2\big)\Big\} = \big\{0,16,7,23,14\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{16} &= [1, 3, 0]\,\quad p_{16}(\gamma) = 3\gamma + 1\\
   S_{16} &= \Big\{\big(0,1\big), \big(1,4\big), \big(2,2\big), \big(3,0\big), \big(4,3\big)\Big\} = \big\{5,21,12,3,19\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{17} &= [2, 3, 0]\,\quad p_{17}(\gamma) = 3\gamma + 2\\
   S_{17} &= \Big\{\big(0,2\big), \big(1,0\big), \big(2,3\big), \big(3,1\big), \big(4,4\big)\Big\} = \big\{10,1,17,8,24\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{18} &= [3, 3, 0]\,\quad p_{18}(\gamma) = 3\gamma + 3\\
   S_{18} &= \Big\{\big(0,3\big), \big(1,1\big), \big(2,4\big), \big(3,2\big), \big(4,0\big)\Big\} = \big\{15,6,22,13,4\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{19} &= [4, 3, 0]\,\quad p_{19}(\gamma) = 3\gamma + 4\\
   S_{19} &= \Big\{\big(0,4\big), \big(1,2\big), \big(2,0\big), \big(3,3\big), \big(4,1\big)\Big\} = \big\{20,11,2,18,9\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{20} &= [0, 4, 0]\,\quad p_{20}(\gamma) = 4\gamma\\
   S_{20} &= \Big\{\big(0,0\big), \big(1,4\big), \big(2,3\big), \big(3,2\big), \big(4,1\big)\Big\} = \big\{0,21,17,13,9\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{21} &= [1, 4, 0]\,\quad p_{21}(\gamma) = 4\gamma + 1\\
   S_{21} &= \Big\{\big(0,1\big), \big(1,0\big), \big(2,4\big), \big(3,3\big), \big(4,2\big)\Big\} = \big\{5,1,22,18,14\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{22} &= [2, 4, 0]\,\quad p_{22}(\gamma) = 4\gamma + 2\\
   S_{22} &= \Big\{\big(0,2\big), \big(1,1\big), \big(2,0\big), \big(3,4\big), \big(4,3\big)\Big\} = \big\{10,6,2,23,19\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{23} &= [3, 4, 0]\,\quad p_{23}(\gamma) = 4\gamma + 3\\
   S_{23} &= \Big\{\big(0,3\big), \big(1,2\big), \big(2,1\big), \big(3,0\big), \big(4,4\big)\Big\} = \big\{15,11,7,3,24\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{24} &= [4, 4, 0]\,\quad p_{24}(\gamma) = 4\gamma + 4\\
   S_{24} &= \Big\{\big(0,4\big), \big(1,3\big), \big(2,2\big), \big(3,1\big), \big(4,0\big)\Big\} = \big\{20,16,12,8,4\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{25} &= [0, 0, 1]\,\quad p_{25}(\gamma) = \gamma^2\\
   S_{25} &= \Big\{\big(0,0\big), \big(1,1\big), \big(2,4\big), \big(3,4\big), \big(4,1\big)\Big\} = \big\{0,6,22,23,9\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{26} &= [1, 0, 1]\,\quad p_{26}(\gamma) = \gamma^2 + 1\\
   S_{26} &= \Big\{\big(0,1\big), \big(1,2\big), \big(2,0\big), \big(3,0\big), \big(4,2\big)\Big\} = \big\{5,11,2,3,14\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{27} &= [2, 0, 1]\,\quad p_{27}(\gamma) = \gamma^2 + 2\\
   S_{27} &= \Big\{\big(0,2\big), \big(1,3\big), \big(2,1\big), \big(3,1\big), \big(4,3\big)\Big\} = \big\{10,16,7,8,19\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{28} &= [3, 0, 1]\,\quad p_{28}(\gamma) = \gamma^2 + 3\\
   S_{28} &= \Big\{\big(0,3\big), \big(1,4\big), \big(2,2\big), \big(3,2\big), \big(4,4\big)\Big\} = \big\{15,21,12,13,24\big\}
   \end{align}

.. math::
   :nowrap:

   \begin{align}
   \alpha_{29} &= [4, 0, 1]\,\quad p_{29}(\gamma) = \gamma^2+4 \\
   S_{29} &= \Big\{\big(0,4\big), \big(1,0\big), \big(2,3\big), \big(3,3\big), \big(4,0\big)\Big\} = \big\{[20, 1, 17, 18, 4]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{30} &= [0, 1, 1]\,\quad p_{30}(\gamma) = \gamma^2+\gamma \\
   S_{30} &= \Big\{\big(0,0\big), \big(1,2\big), \big(2,1\big), \big(3,2\big), \big(4,0\big)\Big\} = \big\{[0, 11, 7, 13, 4]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{31} &= [1, 1, 1]\,\quad p_{31}(\gamma) = \gamma^2+\gamma+1 \\
   S_{31} &= \Big\{\big(0,1\big), \big(1,3\big), \big(2,2\big), \big(3,3\big), \big(4,1\big)\Big\} = \big\{[5, 16, 12, 18, 9]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{32} &= [2, 1, 1]\,\quad p_{32}(\gamma) = \gamma^2+\gamma+2 \\
   S_{32} &= \Big\{\big(0,2\big), \big(1,4\big), \big(2,3\big), \big(3,4\big), \big(4,2\big)\Big\} = \big\{[10, 21, 17, 23, 14]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{33} &= [3, 1, 1]\,\quad p_{33}(\gamma) = \gamma^2+\gamma+3 \\
   S_{33} &= \Big\{\big(0,3\big), \big(1,0\big), \big(2,4\big), \big(3,0\big), \big(4,3\big)\Big\} = \big\{[15, 1, 22, 3, 19]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{34} &= [4, 1, 1]\,\quad p_{34}(\gamma) = \gamma^2+\gamma+4 \\
   S_{34} &= \Big\{\big(0,4\big), \big(1,1\big), \big(2,0\big), \big(3,1\big), \big(4,4\big)\Big\} = \big\{[20, 6, 2, 8, 24]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{35} &= [0, 2, 1]\,\quad p_{35}(\gamma) = \gamma^2+2\gamma \\
   S_{35} &= \Big\{\big(0,0\big), \big(1,3\big), \big(2,3\big), \big(3,0\big), \big(4,4\big)\Big\} = \big\{[0, 16, 17, 3, 24]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{36} &= [1, 2, 1]\,\quad p_{36}(\gamma) = \gamma^2+2\gamma+1 \\
   S_{36} &= \Big\{\big(0,1\big), \big(1,4\big), \big(2,4\big), \big(3,1\big), \big(4,0\big)\Big\} = \big\{[5, 21, 22, 8, 4]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{37} &= [2, 2, 1]\,\quad p_{37}(\gamma) = \gamma^2+2\gamma+2 \\
   S_{37} &= \Big\{\big(0,2\big), \big(1,0\big), \big(2,0\big), \big(3,2\big), \big(4,1\big)\Big\} = \big\{[10, 1, 2, 13, 9]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{38} &= [3, 2, 1]\,\quad p_{38}(\gamma) = \gamma^2+2\gamma+3 \\
   S_{38} &= \Big\{\big(0,3\big), \big(1,1\big), \big(2,1\big), \big(3,3\big), \big(4,2\big)\Big\} = \big\{[15, 6, 7, 18, 14]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{39} &= [4, 2, 1]\,\quad p_{39}(\gamma) = \gamma^2+2\gamma+4 \\
   S_{39} &= \Big\{\big(0,4\big), \big(1,2\big), \big(2,2\big), \big(3,4\big), \big(4,3\big)\Big\} = \big\{[20, 11, 12, 23, 19]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{40} &= [0, 3, 1]\,\quad p_{40}(\gamma) = \gamma^2+3\gamma \\
   S_{40} &= \Big\{\big(0,0\big), \big(1,4\big), \big(2,0\big), \big(3,3\big), \big(4,3\big)\Big\} = \big\{[0, 21, 2, 18, 19]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{41} &= [1, 3, 1]\,\quad p_{41}(\gamma) = \gamma^2+3\gamma+1 \\
   S_{41} &= \Big\{\big(0,1\big), \big(1,0\big), \big(2,1\big), \big(3,4\big), \big(4,4\big)\Big\} = \big\{[5, 1, 7, 23, 24]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{42} &= [2, 3, 1]\,\quad p_{42}(\gamma) = \gamma^2+3\gamma+2 \\
   S_{42} &= \Big\{\big(0,2\big), \big(1,1\big), \big(2,2\big), \big(3,0\big), \big(4,0\big)\Big\} = \big\{[10, 6, 12, 3, 4]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{43} &= [3, 3, 1]\,\quad p_{43}(\gamma) = \gamma^2+3\gamma+3 \\
   S_{43} &= \Big\{\big(0,3\big), \big(1,2\big), \big(2,3\big), \big(3,1\big), \big(4,1\big)\Big\} = \big\{[15, 11, 17, 8, 9]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{44} &= [4, 3, 1]\,\quad p_{44}(\gamma) = \gamma^2+3\gamma+4 \\
   S_{44} &= \Big\{\big(0,4\big), \big(1,3\big), \big(2,4\big), \big(3,2\big), \big(4,2\big)\Big\} = \big\{[20, 16, 22, 13, 14]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{45} &= [0, 4, 1]\,\quad p_{45}(\gamma) = \gamma^2+4\gamma \\
   S_{45} &= \Big\{\big(0,0\big), \big(1,0\big), \big(2,2\big), \big(3,1\big), \big(4,2\big)\Big\} = \big\{[0, 1, 12, 8, 14]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{46} &= [1, 4, 1]\,\quad p_{46}(\gamma) = \gamma^2+4\gamma+1 \\
   S_{46} &= \Big\{\big(0,1\big), \big(1,1\big), \big(2,3\big), \big(3,2\big), \big(4,3\big)\Big\} = \big\{[5, 6, 17, 13, 19]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{47} &= [2, 4, 1]\,\quad p_{47}(\gamma) = \gamma^2+4\gamma+2 \\
   S_{47} &= \Big\{\big(0,2\big), \big(1,2\big), \big(2,4\big), \big(3,3\big), \big(4,4\big)\Big\} = \big\{[10, 11, 22, 18, 24]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{48} &= [3, 4, 1]\,\quad p_{48}(\gamma) = \gamma^2+4\gamma+3 \\
   S_{48} &= \Big\{\big(0,3\big), \big(1,3\big), \big(2,0\big), \big(3,4\big), \big(4,0\big)\Big\} = \big\{[15, 16, 2, 23, 4]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{49} &= [4, 4, 1]\,\quad p_{49}(\gamma) = \gamma^2+4\gamma+4 \\
   S_{49} &= \Big\{\big(0,4\big), \big(1,4\big), \big(2,1\big), \big(3,0\big), \big(4,1\big)\Big\} = \big\{[20, 21, 7, 3, 9]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{50} &= [0, 0, 2]\,\quad p_{50}(\gamma) = 2\gamma^2 \\
   S_{50} &= \Big\{\big(0,0\big), \big(1,2\big), \big(2,3\big), \big(3,3\big), \big(4,2\big)\Big\} = \big\{[0, 11, 17, 18, 14]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{51} &= [1, 0, 2]\,\quad p_{51}(\gamma) = 2\gamma^2+1 \\
   S_{51} &= \Big\{\big(0,1\big), \big(1,3\big), \big(2,4\big), \big(3,4\big), \big(4,3\big)\Big\} = \big\{[5, 16, 22, 23, 19]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{52} &= [2, 0, 2]\,\quad p_{52}(\gamma) = 2\gamma^2+2 \\
   S_{52} &= \Big\{\big(0,2\big), \big(1,4\big), \big(2,0\big), \big(3,0\big), \big(4,4\big)\Big\} = \big\{[10, 21, 2, 3, 24]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{53} &= [3, 0, 2]\,\quad p_{53}(\gamma) = 2\gamma^2+3 \\
   S_{53} &= \Big\{\big(0,3\big), \big(1,0\big), \big(2,1\big), \big(3,1\big), \big(4,0\big)\Big\} = \big\{[15, 1, 7, 8, 4]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{54} &= [4, 0, 2]\,\quad p_{54}(\gamma) = 2\gamma^2+4 \\
   S_{54} &= \Big\{\big(0,4\big), \big(1,1\big), \big(2,2\big), \big(3,2\big), \big(4,1\big)\Big\} = \big\{[20, 6, 12, 13, 9]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{55} &= [0, 1, 2]\,\quad p_{55}(\gamma) = 2\gamma^2+\gamma \\
   S_{55} &= \Big\{\big(0,0\big), \big(1,3\big), \big(2,0\big), \big(3,1\big), \big(4,1\big)\Big\} = \big\{[0, 16, 2, 8, 9]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{56} &= [1, 1, 2]\,\quad p_{56}(\gamma) = 2\gamma^2+\gamma+1 \\
   S_{56} &= \Big\{\big(0,1\big), \big(1,4\big), \big(2,1\big), \big(3,2\big), \big(4,2\big)\Big\} = \big\{[5, 21, 7, 13, 14]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{57} &= [2, 1, 2]\,\quad p_{57}(\gamma) = 2\gamma^2+\gamma+2 \\
   S_{57} &= \Big\{\big(0,2\big), \big(1,0\big), \big(2,2\big), \big(3,3\big), \big(4,3\big)\Big\} = \big\{[10, 1, 12, 18, 19]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{58} &= [3, 1, 2]\,\quad p_{58}(\gamma) = 2\gamma^2+\gamma+3 \\
   S_{58} &= \Big\{\big(0,3\big), \big(1,1\big), \big(2,3\big), \big(3,4\big), \big(4,4\big)\Big\} = \big\{[15, 6, 17, 23, 24]\big\}
   \end{align}


.. math::
   :nowrap:

   \begin{align}
   \alpha_{59} &= [4, 1, 2]\,\quad p_{59}(\gamma) = 2\gamma^2+\gamma+4 \\
   S_{59} &= \Big\{\big(0,4\big), \big(1,2\big), \big(2,4\big), \big(3,0\big), \big(4,0\big)\Big\} = \big\{[20, 11, 22, 3, 4]\big\}
   \end{align}

The weak design is the collection of the 60 sets:

.. math::
   :nowrap:

   \begin{align}
   W = \Big[
       &\big\{0, 1, 2, 3, 4\big\}, \big\{5, 6, 7, 8, 9\big\}, \big\{10, 11, 12, 13, 14\big\}, \big\{15, 16, 17, 18, 19\big\}, \big\{20, 21, 22, 23, 24\big\}, \\
       &\big\{0, 6, 12, 18, 24\big\}, \big\{5, 11, 17, 23, 4\big\}, \big\{10, 16, 22, 3, 9\big\}, \big\{15, 21, 2, 8, 14\big\}, \big\{20, 1, 7, 13, 19\big\}, \\
       &\big\{0, 11, 22, 8, 19\big\}, \big\{5, 16, 2, 13, 24\big\}, \big\{10, 21, 7, 18, 4\big\}, \big\{15, 1, 12, 23, 9\big\}, \big\{20, 6, 17, 3, 14\big\}, \\
       &\big\{0, 16, 7, 23, 14\big\}, \big\{5, 21, 12, 3, 19\big\}, \big\{10, 1, 17, 8, 24\big\}, \big\{15, 6, 22, 13, 4\big\}, \big\{20, 11, 2, 18, 9\big\}, \\
       &\big\{0, 21, 17, 13, 9\big\}, \big\{5, 1, 22, 18, 14\big\}, \big\{10, 6, 2, 23, 19\big\}, \big\{15, 11, 7, 3, 24\big\}, \big\{20, 16, 12, 8, 4\big\}, \\
       &\big\{0, 6, 22, 23, 9\big\}, \big\{5, 11, 2, 3, 14\big\}, \big\{10, 16, 7, 8, 19\big\}, \big\{15, 21, 12, 13, 24\big\}, \big\{20, 1, 17, 18, 4\big\}, \\
       &\big\{0, 11, 7, 13, 4\big\}, \big\{5, 16, 12, 18, 9\big\}, \big\{10, 21, 17, 23, 14\big\}, \big\{15, 1, 22, 3, 19\big\}, \big\{20, 6, 2, 8, 24\big\}, \\
       &\big\{0, 16, 17, 3, 24\big\}, \big\{5, 21, 22, 8, 4\big\}, \big\{10, 1, 2, 13, 9\big\}, \big\{15, 6, 7, 18, 14\big\}, \big\{20, 11, 12, 23, 19\big\}, \\
       &\big\{0, 21, 2, 18, 19\big\}, \big\{5, 1, 7, 23, 24\big\}, \big\{10, 6, 12, 3, 4\big\}, \big\{15, 11, 17, 8, 9\big\}, \big\{20, 16, 22, 13, 14\big\}, \\
       &\big\{0, 1, 12, 8, 14\big\}, \big\{5, 6, 17, 13, 19\big\}, \big\{10, 11, 22, 18, 24\big\}, \big\{15, 16, 2, 23, 4\big\}, \big\{20, 21, 7, 3, 9\big\}, \\
       &\big\{0, 11, 17, 18, 14\big\}, \big\{5, 16, 22, 23, 19\big\}, \big\{10, 21, 2, 3, 24\big\}, \big\{15, 1, 7, 8, 4\big\}, \big\{20, 6, 12, 13, 9\big\}, \\
       &\big\{0, 16, 2, 8, 9\big\}, \big\{5, 21, 7, 13, 14\big\}, \big\{10, 1, 12, 18, 19\big\}, \big\{15, 6, 17, 23, 24\big\}, \big\{20, 11, 22, 3, 4\big\}
       \Big]
   \end{align}
