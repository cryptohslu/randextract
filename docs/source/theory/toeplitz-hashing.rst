================
Toeplitz hashing
================

Even though the quantum leftover hash lemma has solved our main problem by proving that families of two-universal
functions are quantum-proof randomness extractors, we still need to construct these families. Before we mentioned that
for any integers :math:`0\leq m\leq n` it is possible to find them. In this section we will present two particular
families of two-universal functions that are quite relevant for QKD: Toeplitz hashing and modified Toeplitz
hashing.

.. admonition:: Complexity & seed length
   :class: important

   Toeplitz hashing can be computed very efficiently. Its complexity is :math:`O(n\log n)`. However, it requires seeds
   that grow *linearly* with the sum of the input and output lengths. Modified Toeplitz hashing only depends on the
   input length but the growth is still *linear*.


-----------------
Toeplitz matrices
-----------------

Toeplitz\ [#toeplitz]_ matrices are matrices fully characterized by their first row and first column. In other words,
they are diagonal-constant matrices.

.. [#toeplitz] They are named after `Otto Toeplitz`_ (1881 - 1940), a German mathematician that worked mainly in
   functional analysis.

.. _Otto Toeplitz: https://en.wikipedia.org/wiki/Otto_Toeplitz

.. note::
   Toeplitz matrices don't need to be square matrices. The only condition is that their descending diagonals, going left
   to right, are constant.

.. math::
   :nowrap:

   \begin{equation}
   T = \begin{bmatrix}
   T_{0,0}   & T_{0,1}   & T_{0,2}   & \cdots & T_{0,m-1} \\
   T_{1,0}   & T_{0,0}   & T_{0,1}   & \cdots & T_{0,m-2} \\
   T_{2,0}   & T_{1,0}   & T_{0,0}   & \cdots & T_{0,m-3} \\
   \vdots    & \vdots    & \vdots    & \ddots & \vdots    \\
   T_{n-1,0} & T_{n-2,0} & T_{n-3,0} & \cdots & T_{0,0}
   \end{bmatrix}
   \end{equation}

In other words, :math:`T` is a Toeplitz matrix if it satisfies the following condition

.. math::
   T_{i,j} = T_{i-1, j-1}\,.

Once we fix the dimensions of a Toeplitz matrix, let's say :math:`m\times n`, the set of all possible binary Toeplitz
matrices is two-universal. The mapping or *hashing* from :math:`\{0,1\}^n` to :math:`\{0,1\}^m` is done with a
matrix-vector multiplication. The input :math:`x` is shaped as :math:`n\times 1` column vector and multiplied on the
right with the Toeplitz matrix to obtain the :math:`m\times 1` column vector :math:`z`

.. math::
   z = Tx\,.

Using the language of randomness extractors, the function :math:`\text{Ext}(X, Y)` is constructed by defining a
Toeplitz matrix :math:`T` using the uniform seed :math:`Y`, which is used to populate the first row and column. The
remaining matrix elements are determined by the relation :math:`T_{i,j} = T_{i-1, j-1}`. Then, the mapping is defined
by the matrix-vector multiplication.

A convenient notation, which will be useful for later embedding this matrix in a circulant one, is to use the following
mapping to reshape the vector seed into the matrix

.. math::
   T_{i,j}(y):=y_{i-j}\,.

Using this convention, we obtain

.. math::
   :nowrap:

   \begin{equation}
   \text{Ext}(x,y):=
   \begin{bmatrix}
   y_0     & y_{-1}  & y_{-2}  & \cdots  & y_{-n+1}  \\
   y_1     & y_0     & y_{-1}  & \cdots  & y_{-n+2}  \\
   y_2     & y_1     & y_0     & \cdots  & y_{-n+3}  \\
   \vdots  & \vdots  & \vdots  & \ddots  & \vdots    \\
   y_{m-1} & y_{m-2} & y_{m-3} & \cdots  & y_0
   \end{bmatrix}
   \begin{bmatrix}
   x_0 \\ x_1 \\ x_2 \\ \vdots \\ x_{n-1}
   \end{bmatrix}\,.
   \end{equation}


---------------------------------------------
Efficient hashing with Fast Fourier Transform
---------------------------------------------

One of the main advantages of Toeplitz hashing is that it has a complexity :math:`O(n\log n)`. Note that this is more
efficient that the usual matrix-vector multiplication complexity, which is :math:`O(nm)`. The reason for this is that
the hashing can be computed using the `Fast Fourier Transform`_ (FFT).

.. _Fast Fourier Transform: https://en.wikipedia.org/wiki/Fast_Fourier_transform

In order to do this, our general Toeplitz matrix has first to be transformed into a particular kind of Toeplitz matrix:
a square `circulant matrix`_. This is a matrix in which all columns are composed of the same elements and each column is
displaced one element down relative to the preceding column. Mathematically, an :math:`n\times n` circulant matrix
:math:`C` has the form

.. math::
   :nowrap:

   \begin{equation}
   C = \begin{bmatrix}
   c_0     & c_{n-1} & c_{n-2} & \cdots  & c_2    & c_1     \\
   c_1     & c_0     & c_{n-1} & \cdots  & c_3    & c_2     \\
   c_2     & c_1     & c_0     & \cdots  & c_4    & c_3     \\
   \vdots  & \vdots  & \vdots  & \ddots  & \vdots & \vdots  \\
   c_{n-2} & c_{n-3} & c_{n-4} & \cdots  & c_0    & c_{n-1} \\
   c_{n-1} & c_{n-2} & c_{n-3} & \cdots  & c_1    & c_0
   \end{bmatrix}\,.
   \end{equation}

.. _circulant matrix: https://en.wikipedia.org/wiki/Circulant_matrix

In practice, we can convert our generic :math:`m\times n` Toeplitz matrix :math:`T` into an :math:`(m+n-1) \times (m+n-1)`
circulant matrix :math:`\hat{T}` by adding :math:`n-1` additional rows and :math:`m-1` columns to our matrix in the
following way

.. math::
   :nowrap:

   \begin{equation}
   \hat{T}(y):=\begin{bmatrix}
   y_0      & y_{-1}   & y_{-2}  & \cdots  & y_{-n+1} & y_{m-1}  & y_{m-2}  & \cdots  & y_1    \\
   y_1      & y_0      & y_{-1}  & \cdots  & y_{-n+2} & y_{-n+1} & y_{m-1}  & \cdots  & y_2    \\
   y_2      & y_1      & y_0     & \cdots  & y_{-n+3} & y_{-n+2} & y_{-n+1} & \cdots  & y_3    \\
   \vdots   & \vdots   & \vdots  & \ddots  & \vdots   & \vdots   & \vdots   & \vdots  & \vdots \\
   y_{m-1}  & y_{m-2}  & y_{m-3} & \cdots  &          &          &          &         &        \\
   y_{-n+1} & y_{m-1}  & y_{m-2} & \cdots  &          & \ddots   & \vdots   & \vdots  & \vdots \\
   y_{-n+2} & y_{-n+1} & y_{m-1} & \cdots  &          & \cdots   & y_0      & y_{-1}  & y_{-2} \\
   \vdots   & \vdots   & \vdots  & \vdots  &          & \cdots   & y_1      & y_0     & y_{-1} \\
   y_{-1}   & y_{-2}   & y_{-3}  & \cdots  &          & \cdots   & y_2      & y_1     & y_0
   \end{bmatrix}\,.
   \end{equation}

A circulant matrix can be diagonalized using the :math:`n\times n` square Fourier matrix :math:`F_n`, whose
matrix elements are determined by

.. math::
   F_{j,k} := \frac{1}{\sqrt{n}} e^{2\pi ijk/n}\,.

In particular, we can write

.. math::
   \hat{T}_q(y)=F_q^{-1}\text{diag}(F_q y)F_q\,,

where :math:`q=m+n-1` and :math:`y` is the vector with the seed, or equivalently, the first column of the circulant
matrix.

Finally, we can explicitly write the randomness extractor as

.. math::
   \text{Ext}(x,y):=\text{FFT}^{-1}(\text{FFT}(y)\odot\text{FFT}(\hat{x}))\Big|_{0\,\dots\,m-1}\,,

where the symbol :math:`\odot` is to emphasize that the multiplication is the element-wise multiplication of the two
vectors, :math:`\hat{x}` is the input vector :math:`x` from the weak randomness padded with :math:`m-1` zeros, and
:math:`\big|_{0\,\dots\,m-1}` means that we only consider the first :math:`m` bits of the vector after doing the inverse
fast fourier transform.


=========================
Modified Toeplitz hashing
=========================

We have seen that it is very efficient to compute the Toeplitz hashing. However, it requires a rather long seed. In
particular, if we want to extract :math:`m` bits from a :math:`n`-bit string coming from a weak source, we need a
uniform seed of length :math:`m+n-1`. We can reduce this requirement to only :math:`n-1` bits by using a different
family of two-universal functions: the modified Toeplitz hashing.

The trick is to define a new matrix :math:`H` as the concatenation of a smaller Toeplitz matrix :math:`T'` and the
identity matrix. The new extractor is defined as

.. math::
   :nowrap:

   \begin{align}
   \text{Ext}(x,y) &:=
   H(y)x := (T'(y) \| \mathbb{1}_m)x\\&=
   \begin{bmatrix}
   y_0     & y_{-1}  & y_{-2}  & \cdots & y_{-n+m-1} & 1      & 0      & 0      & \cdots & 0      \\
   y_1     & y_0     & y_{-1}  & \cdots & y_{-n+m-2} & 0      & 1      & 0      & \cdots & 0      \\
   y_2     & y_1     & y_0     & \cdots & y_{-n+m-3} & 0      & 0      & 1      & \cdots & 0      \\
   \vdots  & \vdots  & \vdots  & \ddots & \vdots     & \vdots & \vdots & \vdots & \ddots & \vdots \\
   y_{m-1} & y_{m-2} & y_{m-3} & \cdots & y_0        & 0      & 0      & 0      & \cdots & 1
   \end{bmatrix}
   \begin{bmatrix}
   x_0 \\ x_1 \\ x_2 \\ \vdots \\ x_{n-1}
   \end{bmatrix}\,.
   \end{align}

It can be proven that the class of functions determined by such matrices is still two-universal. However, because we are
concatenating the identity matrix :math:`\mathbb{1}_m`, the Toeplitz matrix is of size :math:`m\times(n-m)`, and
therefore we only need a seed of length :math:`n-1`.

The trick of embedding this matrix into a circulant one and use the fast fourier transform to compute the matrix-vector
multiplication still applies to this new matrix :math:`H`.
