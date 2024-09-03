=======
Testing
=======

:obj:`randextract` uses `unittest`_ for unit and integration tests. Here are the instructions to run these tests locally,
as well as detailed calculations for the relevant examples.

.. _unittest: https://docs.python.org/3/library/unittest.html

--------------------
How to run the tests
--------------------

.. code-block:: console

   python -m unittest -v

It is also possible to use pytest

.. code-block:: console

   pytest tests

----------
Unit tests
----------

.. toctree::
   :maxdepth: 1

   toeplitz-unit
   trevisan-unit
