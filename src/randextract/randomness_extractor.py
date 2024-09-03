from abc import ABC, abstractmethod
from numbers import Integral

import numpy as np
from galois import GF2

from ._verify import verify_type


class RandomnessExtractor(ABC):
    """
    A factory class to create seeded randomness extractors. It also serves as an abstract class with the minimum methods
    and properties that any implementation class should have.
    """

    subclasses = {}

    @classmethod
    def register_subclass(cls, extractor_type: str):
        """
        Decorator to register an implementation class.

        Arguments:
            extractor_type: unique label of the extractor

        Example:
            .. code-block:: python

                @RandomnessExtractor.register_subclass('toeplitz')
                class ToeplitzHashing(RandomnessExtractor):
                    # ...
        """

        def decorator(subclass):
            cls.subclasses[extractor_type] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, extractor_type: str, *args, **kwargs):
        """
        It creates a randomness extractor object.

        Arguments:
            extractor_type: the label of the desired extractor type. For available types check
                :ref:`the usage section <Usage>`.
            *args: additional parameters that are passed to the implementation class.
            **kwargs: additional keyword arguments that are passed to the implementation class.

        Examples:
            The following code creates a Toeplitz Hashing extractor and stores it in the variable ``ext``. The output
            length is computed based on the provided bound on the min-entropy of the weak random source and the allowed
            error for the extractor.

            .. code-block:: python

                import randextract
                from randextract import RandomnessExtractor

                ext = RandomnessExtractor.create(
                        extractor_type="toeplitz",
                        input_length=1000,
                        relative_source_entropy=0.2,
                        error_bound=1e-3)

                print(f"Output length = {ext.output_length}")
                print(f"Required seed length = {ext.seed_length}")

            Similarly, we can create a Trevisan's extractor using a polynomial one-bit extractor and a finite field
            polynomial weak design. This extractor takes as input a block of 1 MiB and outputs 500 KiB consuming a seed
            of roughly 26 KiB.

            .. code-block:: python

                import randextract
                from randextract import RandomnessExtractor

                ext = RandomnessExtractor.create(
                        extractor_type="trevisan",
                        weak_design_type="finite_field",
                        one_bit_extractor_type="polynomial",
                        input_length=2**20,
                        relative_source_entropy=0.8,
                        output_length=500 * 2**10,
                        error_bound=1e-3)

                print(f"Output length = {ext.output_length}")
                print(f"Required seed length = {ext.seed_length}")

            These are examples of actual implementations. For all available parameters of the respective implementation
            we hereby refer to the actual implementation and its description, see respective docstrings.
        """
        verify_type(extractor_type, str)

        if extractor_type not in cls.subclasses:
            raise ValueError(
                f"{extractor_type} is not a valid randomness extractor type. Valid types are {cls.subclasses}"
            )

        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return cls.subclasses[extractor_type](*args, **clean_kwargs)

    #
    # Properties
    #
    @property
    @abstractmethod
    def input_length(self) -> int:
        """
        Returns the expected length of the input by the randomness extractor, i.e., the expected length of a bit string
        from a weak random source.

        Returns:
            int: Length of the input.
        """
        pass

    @property
    @abstractmethod
    def seed_length(self) -> int:
        """
        Returns the expected length of the required seed by the randomness extractor.

        Returns:
            int: Length of the seed.
        """
        pass

    @property
    @abstractmethod
    def output_length(self) -> int:
        """
        Returns the length of the randomness extractor's output.

        Returns:
            int: An integer in the range [0, ``input_length``).
        """
        pass

    #
    # Methods
    #
    @staticmethod
    @abstractmethod
    def calculate_length(extractor_type: str, input_length: Integral, **kwargs) -> int:
        """
        For a given extractor type (i.e., quantum-proof) and a set of parameters (e.g., ``error_bound`` or
        ``relative_source_entropy``), it computes the optimal output length for the extractor. In the case of one-bit
        extractors, it computes the optimal seed length instead. The exact number of accepted arguments depends on the
        particular implementation.

        Arguments:
            extractor_type: Type of side information for which the extractor should remain secure. It can take two
                values: "classical" or "quantum", for classical-proof and quantum-proof extractors, respectively.
            input_length: Length of the bit string from the weak random source.

        Returns:
            int: The optimal output or seed length for given input length and constraints.
        """
        pass

    @abstractmethod
    def extract(self, extractor_input: np.ndarray | GF2, seed: np.ndarray | GF2) -> GF2:
        """
        For a given input and a seed, it computes the output of the randomness extractor.

        Arguments:
            extractor_input: a Numpy or Galois binary array from a weak random source.
            seed: a Numpy or Galois binary array with a uniform seed.

        Returns:
            GF2: A binary array with the randomness extractor's output.
        """
        pass
