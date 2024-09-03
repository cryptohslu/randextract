===========================
Unit tests Toeplitz hashing
===========================

------------------------------
TestToeplitzMatrixAlmostSquare
------------------------------

..
   .. literalinclude:: /../../tests/unit/test_toeplitz_hashing.py
      :language: python
      :pyobject: TestToeplitzMatrixAlmostSquare

This is the first relevant unit test to be documented. Previous tests check for correct initialization and use of the
correct types for each of the parameters.

The first step is to compute the output length. Given the lower bound on the min-entropy, we compute the largest output
length to achieve a distance from uniform lower than the error bound. This is guaranteed by the leftover hash lemma.

.. math::
   \text{output_length} = \Big\lfloor \alpha n - 2\log_2\frac{1}{\epsilon}\Big\rfloor =
   \Big\lfloor 0.99\times 5 - 2\log_2\frac{1}{0.99} \Big\rfloor = \Big\lfloor 4.921 \Big\rfloor = 4

.. code-block:: python

   >>> import math
   >>> math.floor(0.99*5 - 2*math.log2(1/0.99))
   4

Our code does not compute and store the Toeplitz matrix to use the :obj:`extract()` method because we use the
:ref:`Fast Fourier Transform <Efficient hashing with Fast Fourier Transform>` and work directly with the input and seed
vectors. However, the method :obj:`to_matrix()` allows us to obtain this Toeplitz matrix. The default arrangement of the
seed to form the Toeplitz matrix is the same as described in :ref:`the theory section <Toeplitz matrices>`, i.e.,

.. math::
   \text{seed} =
   \begin{bmatrix}
   s_0 & \dots & s_{m+n-1}
   \end{bmatrix} \equiv
   \begin{bmatrix}
   y_0 & y_1 & \dots & y_{m-1} & y_{-1} & y_{-2} & \dots & y_{n-1}
   \end{bmatrix}

In this particular test we have

.. math::
   :nowrap:

   \begin{align}
   \text{seed} &=
   \begin{bmatrix}
   1_{\color{red}0} & 0_{\color{red}1} & 0_{\color{red}2} & 0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} &
   0_{\color{red}6} & 0_{\color{red}7}
   \end{bmatrix}\\
   T &= \begin{bmatrix}
   1_{\color{red}0} & 0_{\color{red}7} & 0_{\color{red}6} & 0_{\color{red}5} & 0_{\color{red}4} \\
   0_{\color{red}1} & 1 & 0 & 0 & 0 \\
   0_{\color{red}2} & 0 & 1 & 0 & 0 \\
   0_{\color{red}3} & 0 & 0 & 1 & 0
   \end{bmatrix}
   \end{align}

The seed was chosen to emphasize the differences between the available options for ``seed_mode`` and ``seed_order``.
For ``seed_mode="col-first"`` we obtain

.. math::
   :nowrap:

   \begin{align}
   \text{seed} &=
   \begin{bmatrix}
   1_{\color{red}0} & 0_{\color{red}1} & 0_{\color{red}2} & 0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} &
   0_{\color{red}6} & 0_{\color{red}7}
   \end{bmatrix}\\
   T &= \begin{bmatrix}
   0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} & 0_{\color{red}6} & 0_{\color{red}7} \\
   0_{\color{red}2} & 0 & 0 & 0 & 0 \\
   0_{\color{red}1} & 0 & 0 & 0 & 0 \\
   1_{\color{red}0} & 0 & 0 & 0 & 0
   \end{bmatrix}
   \end{align}

And for ``seed_mode="row-first"``

.. math::
   :nowrap:

   \begin{align}
   \text{seed} &=
   \begin{bmatrix}
   1_{\color{red}0} & 0_{\color{red}1} & 0_{\color{red}2} & 0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} &
   0_{\color{red}6} & 0_{\color{red}7}
   \end{bmatrix}\\
   T &= \begin{bmatrix}
   0_{\color{red}4} & 0_{\color{red}3} & 0_{\color{red}2} & 0_{\color{red}1} & 1_{\color{red}0} \\
   0_{\color{red}5} & 0 & 0 & 0 & 0 \\
   0_{\color{red}6} & 0 & 0 & 0 & 0 \\
   0_{\color{red}7} & 0 & 0 & 0 & 0
   \end{bmatrix}
   \end{align}

With ``seed_mode="custom"``, the permutation array ``seed_order`` determines how the bits from the seed are used to
construct the matrix. Three permutations are tested, here we show the first one which swaps the first and second
bit.

.. math::
   :nowrap:

   \begin{align}
   \text{seed} &=
   \begin{bmatrix}
   1_{\color{red}0} & 0_{\color{red}1} & 0_{\color{red}2} & 0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} &
   0_{\color{red}6} & 0_{\color{red}7}
   \end{bmatrix}\\
   \text{seed_order} &=
   \begin{bmatrix}
   1 & 0 & 2 & 3 & 4 & 5 & 6 & 7
   \end{bmatrix}\\
   T &= \begin{bmatrix}
   0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} & 0_{\color{red}6} & 0_{\color{red}7} \\
   0_{\color{red}2} & 0 & 0 & 0 & 0 \\
   1_{\color{red}0} & 0 & 0 & 0 & 0 \\
   0_{\color{red}1} & 1 & 0 & 0 & 0
   \end{bmatrix}
   \end{align}

Finally, the output of the :obj:`extract()` method matches the matrix-vector multiplication between the Toeplitz matrix
and the input from the weak source. In this particular test we have

.. math::
   \text{output} = T \times \text{input} =
   \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 \\
   1 & 0 & 1 & 1 & 1 \\
   1 & 1 & 0 & 1 & 1 \\
   0 & 1 & 1 & 0 & 1
   \end{bmatrix}
   \begin{bmatrix}
   1 \\ 0 \\ 0 \\ 1 \\ 0
   \end{bmatrix} =
   \begin{bmatrix}
   1 \\ 2 \\ 2 \\ 0
   \end{bmatrix} \mod 2=
   \begin{bmatrix}
   1 \\ 0 \\ 0 \\ 0
   \end{bmatrix}


----------------------
TestToeplitzMatrixWide
----------------------

..
   .. literalinclude:: /../../tests/unit/test_toeplitz_hashing.py
      :language: python
      :pyobject: TestToeplitzMatrixWide

.. math::
   \text{output_length} = \Big\lfloor 0.7\times 8 - 2\log_2\frac{1}{0.5} \Big\rfloor = \Big\lfloor 3.6 \Big\rfloor = 3

.. code-block:: python

   >>> import math
   >>> math.floor(0.7*8 - 2*math.log2(1/0.5))
   3

.. math::
   :nowrap:

   \begin{align}
   \text{seed} &=
   \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}1} & 1_{\color{red}2} & 0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} &
   1_{\color{red}6} & 1_{\color{red}7} & 1_{\color{red}8} & 1_{\color{red}9}
   \end{bmatrix}\\
   T &= \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}9} & 1_{\color{red}8} & 1_{\color{red}7} & 1_{\color{red}6} & 0_{\color{red}5} &
   0_{\color{red}4} & 0_{\color{red}3} \\
   1_{\color{red}1} & 0 & 1 & 1 & 1 & 1 & 0 & 0 \\
   1_{\color{red}2} & 1 & 0 & 1 & 1 & 1 & 1 & 0
   \end{bmatrix}
   \end{align}

.. math::
   \text{output} = T \times \text{input} =
   \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 \\
   1 & 0 & 1 & 1 & 1 & 1 & 0 & 0 \\
   1 & 1 & 0 & 1 & 1 & 1 & 1 & 0
   \end{bmatrix}
   \begin{bmatrix}
   1 \\ 1 \\ 0 \\ 0 \\ 0 \\ 1 \\ 0 \\ 1
   \end{bmatrix} =
   \begin{bmatrix}
   1 \\ 2 \\ 3
   \end{bmatrix} \mod 2=
   \begin{bmatrix}
   1 \\ 0 \\ 1
   \end{bmatrix}


------------------------------------
TestToeplitzMatrixOneDimensionalWide
------------------------------------

..
   .. literalinclude:: /../../tests/unit/test_toeplitz_hashing.py
      :language: python
      :pyobject: TestToeplitzMatrixOneDimensionalWide

.. math::
   \text{output_length} = \Big\lfloor 0.5\times 10 - 2\log_2\frac{1}{0.25} \Big\rfloor = 1

.. code-block:: python

   >>> import math
   >>> math.floor(0.5*10 - 2*math.log2(1/0.25))
   1


This is an extreme scenario where the Toeplitz matrix only has one row, but the "matrix" is still computed in the same
way, so it looks reversed except for the first bit.

.. math::
   :nowrap:

   \begin{align}
   \text{seed} &=
   \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}1} & 1_{\color{red}2} & 0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} &
   1_{\color{red}6} & 1_{\color{red}7} & 1_{\color{red}8} & 1_{\color{red}9}
   \end{bmatrix}\\
   T &= \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}9} & 1_{\color{red}8} & 1_{\color{red}7} & 1_{\color{red}6} & 0_{\color{red}5} &
   0_{\color{red}4} & 0_{\color{red}3} & 1_{\color{red}2} & 1_{\color{red}1}
   \end{bmatrix}
   \end{align}


.. math::
   \text{output} = T \times \text{input} =
   \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 1
   \end{bmatrix}
   \begin{bmatrix}
   0 \\ 1 \\ 0 \\ 1 \\ 1 \\ 0 \\ 1 \\ 1 \\ 1 \\ 0
   \end{bmatrix} =
   4 \mod 2 = 0


====================================
Unit tests Modified Toeplitz hashing
====================================

--------------------------------------
TestModifiedToeplitzMatrixAlmostSquare
--------------------------------------

..
   .. literalinclude:: /../../tests/unit/test_modified_toeplitz_hashing.py
      :language: python
      :pyobject: TestModifiedToeplitzMatrixAlmostSquare

.. math::
   \text{output_length} = \Big\lfloor 0.75\times 9 - 2\log_2\frac{1}{0.5} \Big\rfloor = \Big\lfloor 4.75 \Big\rfloor = 4

.. code-block:: python

   >>> import math
   >>> math.floor(0.75*9 - 2*math.log2(1/0.5))
   4

Remember that the :ref:`modified Toeplitz hashing <Modified Toeplitz hashing>` appends an identity matrix to reduce the
required seed. Because of this, instead of a seed of length :math:`\text{input_length}+\text{output_length}-1`, we only
need :math:`\text{input_length}-1` bits. The order of these bits to form the matrix is exactly the same as in the normal
Toeplitz hashing, and ``seed_mode`` and ``seed_order`` kwargs can also be used to modify it.

.. math::
   :nowrap:

   \begin{align}
   \text{seed} &=
   \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}1} & 1_{\color{red}2} & 0_{\color{red}3} & 1_{\color{red}4} & 1_{\color{red}5} &
   1_{\color{red}6} & 1_{\color{red}7}
   \end{bmatrix}\\
   T &= \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}7} & 1_{\color{red}6} & 1_{\color{red}5} & 1_{\color{red}4} & 1 \\
   1_{\color{red}1} & 0 & 1 & 1 & 1 & & 1 \\
   1_{\color{red}2} & 1 & 0 & 1 & 1 & & & 1 \\
   0_{\color{red}3} & 1 & 1 & 0 & 1 & & & & 1
   \end{bmatrix}
   \end{align}

.. math::

   \text{output} = T \times \text{input }= \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 \\
   1 & 0 & 1 & 1 & 1 & & 1 \\
   1 & 1 & 0 & 1 & 1 & & & 1 \\
   0 & 1 & 1 & 0 & 1 & & & & 1
   \end{bmatrix}
   \begin{bmatrix}
   1 \\ 0 \\ 0 \\ 1 \\ 0 \\ 1 \\ 1 \\ 1 \\ 0
   \end{bmatrix} =
   \begin{bmatrix}
   2 \\ 3 \\ 3 \\ 0
   \end{bmatrix} \mod 2 =
   \begin{bmatrix}
   0 \\ 1 \\ 1 \\ 0
   \end{bmatrix}


--------------------------------
TestModifiedToeplitzMatrixNarrow
--------------------------------

..
   .. literalinclude:: /../../tests/unit/test_modified_toeplitz_hashing.py
      :language: python
      :pyobject: TestModifiedToeplitzMatrixNarrow

.. math::
   \text{output_length} = \Big\lfloor 0.99\times 11 - 2\log_2\frac{1}{0.5} \Big\rfloor = \Big\lfloor 8.89 \Big\rfloor = 8

.. code-block:: python

   >>> import math
   >>> math.floor(0.99*11 - 2*math.log2(1/0.5))
   8

.. math::
   :nowrap:

   \begin{align}
   \text{seed} &=
   \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}1} & 1_{\color{red}2} & 0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} &
   1_{\color{red}6} & 1_{\color{red}7} & 1_{\color{red}8} & 1_{\color{red}9}
   \end{bmatrix}\\
   T &= \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}8} & 1_{\color{red}9} & 1\\
   1_{\color{red}1} & 0 & 1 & & 1 \\
   1_{\color{red}2} & 1 & 0 & & & 1 \\
   0_{\color{red}3} & 1 & 1 & & & & 1 \\
   0_{\color{red}4} & 0 & 1 & & & & & 1 \\
   0_{\color{red}5} & 0 & 0 & & & & & & 1\\
   1_{\color{red}6} & 0 & 0 & & & & & & & 1\\
   1_{\color{red}7} & 1 & 0 & & & & & & & & 1\\
   \end{bmatrix}
   \end{align}

.. math::

   T \times \text{input }&= \begin{bmatrix}
   0 & 1 & 1 & 1\\
   1 & 0 & 1 & & 1 \\
   1 & 1 & 0 & & & 1 \\
   0 & 1 & 1 & & & & 1 \\
   0 & 0 & 1 & & & & & 1 \\
   0 & 0 & 0 & & & & & & 1\\
   1 & 0 & 0 & & & & & & & 1\\
   1 & 1 & 0 & & & & & & & & 1\\
   \end{bmatrix}
   \begin{bmatrix}
   1 \\ 1 \\ 0 \\ 0 \\ 0 \\ 1 \\ 0 \\ 1 \\ 1 \\ 0 \\ 1
   \end{bmatrix} =
   \begin{bmatrix}
   1 \\ 1 \\ 3 \\ 1 \\ 1 \\ 1 \\ 1 \\ 3
   \end{bmatrix} \mod 2 =
   \begin{bmatrix}
   1 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1
   \end{bmatrix}


------------------------------
TestModifiedToeplitzMatrixWide
------------------------------

..
   .. literalinclude:: /../../tests/unit/test_modified_toeplitz_hashing.py
      :language: python
      :pyobject: TestModifiedToeplitzMatrixWide

.. math::
   \text{output_length} = \Big\lfloor 0.5\times 11 - 2\log_2\frac{1}{0.5} \Big\rfloor = \Big\lfloor 3.5 \Big\rfloor = 3

.. code-block:: python

   >>> import math
   >>> math.floor(0.5*11 - 2*math.log2(1/0.5))
   3

.. math::
   :nowrap:

   \begin{align}
   \text{seed} &=
   \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}1} & 1_{\color{red}2} & 0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} &
   1_{\color{red}6} & 1_{\color{red}7} & 1_{\color{red}8} & 1_{\color{red}9}
   \end{bmatrix}\\
   T &= \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}9} & 1_{\color{red}8} & 1_{\color{red}7} & 1_{\color{red}6} & 0_{\color{red}5} &
   0_{\color{red}4} & 0_{\color{red}3} & 1 \\
   1_{\color{red}1} & 0 & 1 & 1 & 1 & 1 & 0 & 0 & & 1 \\
   1_{\color{red}2} & 1 & 0 & 1 & 1 & 1 & 1 & 0 & & & 1 \\
   \end{bmatrix}
   \end{align}

.. math::

   T \times \text{input} = \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 0 &
   0 & 0 & 1 \\
   1 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & & 1 \\
   1 & 1 & 0 & 1 & 1 & 1 & 1 & 0 & & & 1 \\
   \end{bmatrix}
   \begin{bmatrix}
   1 \\ 1 \\ 0 \\ 0 \\ 0 \\ 1 \\ 0 \\ 1 \\ 1 \\ 1 \\ 1
   \end{bmatrix} =
   \begin{bmatrix}
   2 \\ 3 \\ 4
   \end{bmatrix} \mod 2 =
   \begin{bmatrix}
   0 \\ 1 \\ 0
   \end{bmatrix}

----------------------------------------------
TestModifiedToeplitzMatrixOneDimensionalNarrow
----------------------------------------------

..
   .. literalinclude:: /../../tests/unit/test_modified_toeplitz_hashing.py
      :language: python
      :pyobject: TestModifiedToeplitzMatrixOneDimensionalNarrow

.. math::
   \text{output_length} = \Big\lfloor 0.99\times 11 - 2\log_2\frac{1}{0.99} \Big\rfloor = \Big\lfloor 10.861 \Big\rfloor = 10

.. code-block:: python

   >>> import math
   >>> math.floor(0.99*11 - 2*math.log2(1/0.99))
   10

.. math::
   :nowrap:

   \begin{align}
   \text{seed} &=
   \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}1} & 1_{\color{red}2} & 0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} &
   1_{\color{red}6} & 1_{\color{red}7} & 1_{\color{red}8} & 1_{\color{red}9}
   \end{bmatrix}\\
   T &= \begin{bmatrix}
   0_{\color{red}0} & 1 \\
   1_{\color{red}1} & & 1 \\
   1_{\color{red}2} & & & 1\\
   0_{\color{red}3} & & & & 1\\
   0_{\color{red}4} & & & & & 1\\
   0_{\color{red}5} & & & & & & 1\\
   1_{\color{red}6} & & & & & & & 1\\
   1_{\color{red}7} & & & & & & & & 1\\
   1_{\color{red}8} & & & & & & & & & 1\\
   1_{\color{red}9} & & & & & & & & & & 1
   \end{bmatrix}
   \end{align}

.. math::

   T \times \text{input} &= \begin{bmatrix}
   0 & 1 \\
   1 & & 1 \\
   1 & & & 1\\
   0 & & & & 1\\
   0 & & & & & 1\\
   0 & & & & & & 1\\
   1 & & & & & & & 1\\
   1 & & & & & & & & 1\\
   1 & & & & & & & & & 1\\
   1 & & & & & & & & & & 1
   \end{bmatrix}
   \begin{bmatrix}
   1 \\ 1 \\ 0 \\ 1 \\ 1 \\ 1 \\ 0 \\ 0 \\ 1 \\ 0 \\ 0
   \end{bmatrix} =
   \begin{bmatrix}
   1 \\ 1 \\ 2 \\ 1 \\ 1 \\ 0 \\ 1 \\ 2 \\ 1 \\ 1
   \end{bmatrix} \mod 2 =
   \begin{bmatrix}
   1 \\ 1 \\ 0 \\ 1 \\ 1 \\ 0 \\ 1 \\ 0 \\ 1 \\ 1
   \end{bmatrix}

--------------------------------------------
TestModifiedToeplitzMatrixOneDimensionalWide
--------------------------------------------

..
   .. literalinclude:: /../../tests/unit/test_modified_toeplitz_hashing.py
      :language: python
      :pyobject: TestModifiedToeplitzMatrixOneDimensionalWide

.. math::
   \text{output_length} = \Big\lfloor 0.5\times 11 - 2\log_2\frac{1}{0.25} \Big\rfloor = \Big\lfloor 1.5 \Big\rfloor = 1

.. code-block:: python

   >>> import math
   >>> math.floor(0.5*11 - 2*math.log2(1/0.25))
   1

.. math::
   :nowrap:

   \begin{align}
   \text{seed} &=
   \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}1} & 1_{\color{red}2} & 0_{\color{red}3} & 0_{\color{red}4} & 0_{\color{red}5} &
   1_{\color{red}6} & 1_{\color{red}7} & 1_{\color{red}8} & 1_{\color{red}9}
   \end{bmatrix}\\
   T &=
   \begin{bmatrix}
   0_{\color{red}0} & 1_{\color{red}9} & 1_{\color{red}8} & 1_{\color{red}7} & 1_{\color{red}6} & 0_{\color{red}5} &
   0_{\color{red}4} & 0_{\color{red}3} & 1_{\color{red}2} & 1_{\color{red}1} & 1
   \end{bmatrix}
   \end{align}

.. math::

   T \times \text{input} =
   \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 1 & 1
   \end{bmatrix}
   \begin{bmatrix}
   0 \\ 1 \\ 0 \\ 1 \\ 1 \\ 0 \\ 1 \\ 1 \\ 1 \\ 0 \\ 0
   \end{bmatrix} =
   4 \mod 2 = 0
