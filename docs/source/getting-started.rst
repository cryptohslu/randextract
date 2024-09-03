===============
Getting started
===============

------------
Installation
------------

This Python package can be easily installed using ``pip``

.. code-block:: console

   pip install --user randextract

Note that the previous command installs our package and all runtime dependencies into your `local Python environment`_.
This may have unwanted effects (e.g. downgrading an already installed NumPy package for compatibility reasons).
If you want to test our package in `a virtual Python environment`_, you can run the following instead.

.. _local Python environment: https://docs.python.org/3/install/index.html#alternate-installation-the-user-scheme
.. _a virtual Python environment: https://docs.python.org/3/library/venv.html

.. tabs::

   .. tab:: POSIX (GNU/Linux, macOS, etc.)

      .. code-block:: bash

         python -m venv --upgrade-deps env-randextract
         source env-randextract/bin/activate
         pip install randextract

   .. tab:: Windows (cmd)

      .. code-block:: powershell

         python -m venv --upgrade-deps env-randextract
         env-randextract\Scripts\activate.bat
         pip install randextract

   .. tab:: Windows (PowerShell)

      .. code-block:: powershell

         python -m venv --upgrade-deps env-randextract
         env-randextract\Scripts\Activate.ps1
         pip install randextract

.. warning::

   If you decide to use a virtual environment, you will have to activate it every time you start a new console by
   running the second command from the box above.


Development installation
------------------------

`PyPI`_ only contains stable releases. If you want to try the latest code, you have to clone the source repository
first. You will need ``git`` for this, so check out the `documentation`_ if you don't already have it installed on
your system.

.. _PyPI: https://pypi.org/project/randextract/
.. _documentation: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

.. code-block:: console

   git clone https://github.com/cryptohslu/randextract.git
   cd randextract

Now you can create a virtual env and install the package from the local directory

.. tabs::

   .. tab:: POSIX (GNU/Linux, macOS, etc.)

      .. code-block:: bash

         python -m venv --upgrade-deps env-randextract
         source env-randextract/bin/activate
         pip install ".[dev,docs,examples,test]"

   .. tab:: Windows (cmd)

      .. code-block:: powershell

         python -m venv --upgrade-deps env-randextract
         env-randextract\Scripts\activate.bat
         pip install ".[dev,docs,examples,test]"

   .. tab:: Windows (PowerShell)

      .. code-block:: powershell

         python -m venv --upgrade-deps env-randextract
         env-randextract\Scripts\Activate.ps1
         pip install ".[dev,docs,examples,test]"

.. hint::

   You can also install the package in `"editable" mode`_ by adding ``-e`` to the ``pip`` command. In this way, you
   don't need to reinstall the package after doing changes in the source code.

.. _"editable" mode: https://pip.pypa.io/en/latest/topics/local-project-installs/#editable-installs

To locally build this documentation

.. code-block:: console

   cd docs
   make html

And to run the unit tests

.. code-block:: console

   pytest tests


--------------------
Testing installation
--------------------

The following code creates a modified Toeplitz hashing extractor. After generating a random input and a seed with the
correct lengths, we pass them to the extractor and save the output to a new variable.

.. code-block:: python

   from galois import GF2

   import randextract
   from randextract import ModifiedToeplitzHashing, RandomnessExtractor

   optimal_output_length = ModifiedToeplitzHashing.calculate_length(
       extractor_type="quantum",
       input_length=2**20,
       relative_source_entropy=0.8,
       error_bound=1e-3,
   )

   ext = RandomnessExtractor.create(
       extractor_type="modified_toeplitz",
       input_length=2**20,
       output_length=optimal_output_length,
   )

   input_ext = GF2.Random(ext.input_length)
   seed_ext = GF2.Random(ext.seed_length)

   output = ext.extract(input_ext, seed_ext)


If you are interested in learning more about randomness extractors continue reading :ref:`the theory section <Theory>`.
If you just want to learn how to use this Python package, you can jump directly to :ref:`the usage section <Usage>`.
