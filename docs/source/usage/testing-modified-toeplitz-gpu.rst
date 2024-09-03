.. py:module:: randextract
   :no-index:

=============================
GPU modified Toeplitz hashing
=============================

.. seealso::

   The full source code of this example is in ``examples/validation_gpu_modified_toeplitz_zeromq.py``

.. figure:: /images/GeForce3_Ti_500.jpg
   :figclass: margin
   :alt: Nvidia GeForce3 Ti 500

   A photo of an old Nvidia GeForce3 Ti 500 (source: `Wikipedia.org`_)

.. _Wikipedia.org: https://commons.wikimedia.org/wiki/File:GeForce3_Ti_500.jpg

Modern GPUs can be used to accelerate many computation tasks. Primitive graphics accelerators were developed quite early
to unburden the CPU from some computations. However, in the early 2000s, both Nvidia (with their GeForce 3 Series) and
ATI (with their Radeon R300) moved to the direction of the so called general purpose GPUs (GPGPUs). These cards
supported some mathematical functions and loops. Multiple languages and interfaces were developed to support
accelerating devices and GPUs in particular (e.g., `OpenCL`_, `CUDA`_, `OpenACC`_, ...). Nowadays, GPUs from Nvidia, AMD
and Intel have evolved to become even more general-purpose parallel processors, and they are widely use to to accelerate
many computations in different fields. In this particular example, the GPU is used to accelerate the FFT computation.

.. _OpenCL: https://www.khronos.org/opencl/
.. _CUDA: https://developer.nvidia.com/cuda-toolkit
.. _OpenACC: https://www.openacc.org/

`Nico Bosshard`_ implemented the modified Toeplitz hashing to run on GPUs using the CUDA library for supported Nvidia
cards and with Vulkan for many other compatible GPUs (AMD, Intel, etc.). This implementation :cite:p:`2021bosshard` is
open source and `it is available on GitHub`_. To the best of our knowledge, this remains the fastest modified Toeplitz
implementation, and for that reason it has been used in at least one QKD experiment :cite:p:`2023fadri`, where the
privacy amplification step could not be implemented with FPGAs for the required block lengths and throughput.

.. _Nico Bosshard: https://github.com/nicoboss
.. _it is available on GitHub: https://github.com/nicoboss/PrivacyAmplification


Here we briefly explain the setup we used to test this implementation, and show how the CUDA version was validated
with the :obj:`Validator` class using ``input_method="custom"`` and a custom implementation of
:obj:`ValidatorCustomClassAbs` that takes advantage of `ZeroMQ`_ to exchange the arrays between our reference
implementation and the CUDA application.

.. _ZeroMQ: https://zeromq.org/

------------------------
PrivacyAmplificationCuda
------------------------

The ``PrivacyAmplificationCuda`` application was compiled using `CUDA 12.3.1`_ and executed on a
`NVIDIA GeForce GTX 1070`_. The following ``config.yaml`` was used instead of the upstream one:

.. _CUDA 12.3.1: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
.. _NVIDIA GeForce GTX 1070: https://www.nvidia.com/en-gb/geforce/graphics-cards/geforce-gtx-1070/specifications/

.. code-block::

   factor_exp: 27
   reuse_seed_amount: 0
   vertical_len: 50331648
   do_xor_key_rest: true
   do_compress: true
   reduction_exp: 11
   pre_mul_reduction_exp: 5
   gpu_device_id_to_use: 0
   input_blocks_to_cache: 16
   output_blocks_to_cache: 16
   show_ampout: 8
   show_zeromq_status: true
   use_matrix_seed_server: true
   address_seed_in: 'tcp://<REDACTED>:45555'
   use_key_server: true
   address_key_in: 'tcp://<REDACTED>:47777'
   host_ampout_server: true
   address_amp_out: 'tcp://*:48888'
   store_first_ampouts_in_file: 0
   verify_ampout: false
   verify_ampout_threads: 8


This configuration chooses a particular family of :ref:`modified Toeplitz hashing <Modified Toeplitz hashing>`
functions that takes :math:`2^{27} + 1` bits (:math:`\sim 17` MB) as input and outputs 50331648 bits (:math:`\sim 6` MB).
This family requires :math:`2^{27}` uniform bits as seed to select one particular hash function. Relevant to run this
particular validation are the parameters ``address_seed_in``, ``address_key_in`` and ``address_amp_out``. The first two
are the addresses where the application will try to pull the extractor input and seed, respectively,
using ZeroMQ sockets. The last is the address where the output of the extractor will be pushed. To read a full
description of all the paramenters in the configuration file, check `the commented version upstream`_.

.. _the commented version upstream: https://github.com/nicoboss/PrivacyAmplification/blob/master/PrivacyAmplification/config.yaml

.. note::

   The IP of the server where :obj:`randextract` was running was redacted. If you try to replicate this experiment,
   change the ``address_seed_in`` and ``address_seed_in`` with a valid IP. If you are running our Python package in
   the same device as ``PrivacyAmplificationCuda``, try using the host loopback address ``127.0.0.1``.

---------------------------
Validator with custom class
---------------------------

Our :obj:`Validator` class supports the ``input_mode="custom"`` to allow the user to validate implementations that do
not use standard input and output, or files, to obtain the extractor inputs and seeds and to save the output. In this
particular case, all the arrays are communicated as raw bytes using `ZeroMQ sockets`_ in a `push/pull pipeline pattern`_.

.. _ZeroMQ sockets: https://zeromq.org/socket-api/
.. _push/pull pipeline pattern: https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/patterns/pushpull.html

In this mode, the :obj:`Validator` also expects a ``custom_class``, which is an instance of an implementation class of
the base abstract class provided in :obj:`ValidatorCustomClassAbs`. The implementation class should contain, at least,
the following three methods: :obj:`get_extractor_inputs()`, :obj:`get_extractor_seeds()` and :obj:`get_extractor_output()`.
This is enforced by the abstract class, raising a ``TypeError`` if some of these are missing. The first two methods
should produce generators (i.e., ``yield`` instead of ``return``). This is implementation class used to test
``PrivacyAmplificationCuda``.

.. literalinclude:: /../../examples/validation_gpu_modified_toeplitz_zeromq.py
   :language: python
   :pyobject: CustomValidatorGPU

Notice the following:

1. The constructor (``def __init__()``) is optional, but here it is used to keep a counter and implement something
   similar to what is done by the :obj:`Validator` class when used with ``input_method="stdio"`` and the kwarg
   ``sample_size``. It also takes an extractor as input to have access to its properties ``input_length`` and
   ``seed_length``.
2. All the communication with the CUDA application in the :obj:`get_extractor_inputs()` and :obj:`get_extractor_seeds()`
   methods is done before yielding the randomly generated arrays. This ensures that :obj:`get_extractor_output()`
   will never be called before the GPU implementation has received the two inputs.
3. Two functions are used to convert between GF2 arrays and memory buffers (raw bytes). These are defined in the example
   script and, therefore, not shown above. Alternatively, if you prefer your custom class to be self-sufficient,
   they could be added as static methods.
4. As mentioned in a code comment, if `this patch`_ is not applied, then the last bit of the input passed to the
   randomness extractor should always be zero to obtain the same results with our implementation. This is due to
   `a bug in the GPU implementation`_ previously unnoticed.

.. _this patch: https://github.com/nicoboss/PrivacyAmplification/pull/2
.. _a bug in the GPU implementation: https://github.com/nicoboss/PrivacyAmplification/issues/1


.. figure:: /images/screenshot-validation-gpu-example.png
   :width: 90%
   :figclass: margin-caption
   :alt: Screenshot validation

   Screenshot after running this example
