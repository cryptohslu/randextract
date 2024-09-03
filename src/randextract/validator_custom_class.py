from abc import ABC, abstractmethod

from galois import GF2

from .randomness_extractor import RandomnessExtractor


class ValidatorCustomClassAbs(ABC):
    """
    An abstract base class that serves as the skeleton for the custom class required to validate the implementation of
    an extractor using the class obj::`Validator` together with the ``input_method="custom"``. All the methods here
    represent the set of minimum functions that should be implemented to test an extractor in this mode. Feel free to
    implement any other required methods. If some of these methods are missing in the actual implementation class,
    a TypeError will be raised when trying to create an instance.
    """

    #
    # Methods
    #
    @abstractmethod
    def get_extractor_inputs(self) -> GF2:
        """
        This method should yield the next extractor input used to validate the implementation as a GF2 array.

        Yields:
            GF2: a GF2 array of length self.ref.input_length
        """
        pass

    @abstractmethod
    def get_extractor_seeds(self) -> GF2:
        """
        This method should yield the next extractor seed used to validate the implementation as a GF2 array.

        Yields:
            GF2: a GF2 array of length self.ref.seed_length
        """
        pass

    @abstractmethod
    def get_extractor_output(self, ext_input, ext_seed) -> GF2:
        """
        This method should return the output of the extractor being tested as a GF2 array.

        Returns:
            GF2: a GF2 array of length self.ref.output_length
        """
        pass
