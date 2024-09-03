from abc import ABC, abstractmethod
from numbers import Integral
from pathlib import Path

import numpy as np
from galois import FieldArray

from .._verify import verify_array, verify_type

# TODO: unit tests non-abstract class methods (register_subclass, create, get_relative_overlap_for_type)


class WeakDesign(ABC):
    """
    Abstract class with the minimum methods and properties that any weak design implementation class should have.
    """

    subclasses = {}

    @classmethod
    def register_subclass(cls, weak_design_type: str):
        """
        Decorator to register an implementation class.

        Arguments:
            weak_design_type: unique label of the weak design

        Example:
            .. code-block:: python

                @WeakDesign.register_subclass('finite_field')
                class FiniteFieldPolynomialDesign(WeakDesign):
                    # ...
        """

        def decorator(subclass):
            cls.subclasses[weak_design_type] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, weak_design_type: str, *args, **kwargs):
        r"""
        It creates a weak design object.

        Arguments:
            weak_design_type: the label of the desired weak design type.
            *args: additional parameters that are passed to the implementation class.
            **kwargs: additional keyword arguments that are passed to the implementation class.

        Example:
            The following code creates a small finite field polynomial weak design and stores it in
            the variable ``weak``.

            .. code-block:: python

                import randextract
                from randextract import WeakDesign

                weak = WeakDesign.create(weak_design_type="finite_field",
                                         number_of_sets=20, size_of_set=5)
        """
        verify_type(weak_design_type, str)

        if weak_design_type not in cls.subclasses:
            raise ValueError(
                f"{weak_design_type} is not a valid weak design type. Valid types are {cls.subclasses}"
            )

        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return cls.subclasses[weak_design_type](*args, **clean_kwargs)

    # TODO: is this necessary?
    @classmethod
    def get_relative_overlap(cls, weak_design_type: str) -> float:
        """
        Weak designs are characterized by an upper bound on their set's overlap. This method returns this bound. For the
        mathematical definition check :ref:`the corresponding section <Weak designs>` in the theory.

        Arguments:
            weak_design_type: the label of the weak design type.

        Returns:
            float: Upper bound on the weak design's overlap normalized by the number of sets.
        """
        verify_type(weak_design_type, str)

        if weak_design_type not in cls.subclasses:
            raise ValueError(
                f"{weak_design_type} is not a valid weak design type. Valid types are {cls.subclasses}"
            )

        return cls.subclasses[weak_design_type].relative_overlap()

    # TODO: add unit test
    @staticmethod
    def is_valid(
        weak_design: np.ndarray,
        number_of_sets: Integral,
        size_of_sets: Integral,
        relative_overlap: float,
    ) -> bool:
        r"""
        It checks if a NumPy array is a valid weak design.

        Arguments:
            weak_design: a NumPy array containing a weak design.
            number_of_sets: the desired number of sets that the weak design should contain.
            size_of_sets: the desired size of the sets.
            relative_overlap: an upper bound of the relative overlap between the sets.

        Returns:
            bool: True if the NumPy array is a valid weak design with the desired parameters, False otherwise.

        """
        upper_bound = number_of_sets * relative_overlap
        for index in range(number_of_sets):
            if len(np.unique(weak_design[index])) != size_of_sets:
                return False
            # Definition II.2 arXiv:1109.6147
            actual_overlap = 0
            for i in range(index):
                actual_overlap += 2 ** len(
                    np.intersect1d(
                        weak_design[index],
                        weak_design[i],
                        assume_unique=True,
                    )
                )
                if actual_overlap > upper_bound:
                    return False
        return True

    # TODO: add unit test
    @staticmethod
    def save_to_file(weak_design: np.ndarray, filename: str | Path) -> None:
        r"""
        A wrapper using ``numpy.save()`` to save a weak design to a file so that it can be read and reused later.

        Arguments:
             weak_design: the weak design to be saved.
             filename: a string with the desired filename or a ``Path`` object. If a ``str`` is passed, the file will be
                stored in the current working directory.
        """
        if not weak_design.is_computed:
            warnings.warn("Design not computed yet! Use compute_design() method.")
            return

        verify_type(filename, [str, Path])

        if isinstance(filename, str):
            filename = Path.cwd() / filename

        np.save(filename, weak_design, allow_pickle=True, fix_imports=False)
        print(f"Weak design saved to {filename}.npy")

    # TODO: add unit test
    @staticmethod
    def read_from_file(
        filename: str | Path,
        number_of_sets: Integral,
        size_of_sets: Integral,
        range_design: Integral,
    ) -> np.ndarray:
        r"""
        It reads and verifies a precomputed weak design stored in a file. If the array has the correct shape and
        contains values in the right range, the weak design is returned as a NumPy array. Note that this method does not
        check if it's a valid weak design with a certain overlap parameter. Use :obj:`is_valid()` for that.

        Arguments:
            filename: a string with the filename or a ``Path`` object to read. If a ``str`` is passed, the file is
                assumed to be in the current working directory.
            number_of_sets: the desired number of sets that the weak design should contain.
            size_of_sets: the desired size of the sets.
            range_design: the desired range of the weak design

        Returns:
            np.ndarray: A NumPy array of shape number_of_sets x size_of_sets and values in [0, range).
        """
        verify_type(filename, [np.ndarray, Path, str])
        verify_type(number_of_sets, Integral)
        verify_type(size_of_sets, Integral)
        verify_type(range_design, Integral)

        if isinstance(filename, str):
            filename = Path.cwd() / filename

        if isinstance(filename, Path):
            if not filename.is_file():
                raise ValueError(f"{filename} does not exist.")
            weak_design = np.load(filename, allow_pickle=True, fix_imports=False)
        else:
            weak_design = filename

        if weak_design.shape != (number_of_sets, size_of_sets):
            raise ValueError(
                f"Precomputed weak design must be a {number_of_sets} x {size_of_sets} array, but it is {weak_design.shape}."
            )

        verify_array(
            weak_design,
            valid_type=Integral,
            valid_range=[0, range_design, "right-open"],
        )

        return weak_design

    # This could be a classproperty if supported in future versions of Python
    @classmethod
    @abstractmethod
    def relative_overlap(cls) -> float:
        """
        Returns:
            float: Upper bound on the weak design's overlap normalized by the number of sets.
        """
        pass

    #
    # Abstract properties
    #
    @property
    @abstractmethod
    def number_of_sets(self) -> int:
        """
        Returns:
            int: Size of the weak design, i.e., the number of sets.
        """
        pass

    @property
    @abstractmethod
    def size_of_sets(self) -> int:
        """
        Returns:
             int: The size (cardinality) of all sets.
        """
        pass

    @property
    @abstractmethod
    def weak_design(self) -> np.ndarray:
        r"""
        Returns:
            FieldArray: The computed weak design, i.e., a family of ``number_of_sets`` sets of size ``size_of_set``.
        """
        pass

    @property
    @abstractmethod
    def is_computed(self) -> bool:
        """
        Returns:
            bool: True if the design has been computed already, False otherwise.
        """

    @property
    @abstractmethod
    def range_design(self) -> int:
        """
        Returns:
            int: All the sets that form the weak design have elements from [d] = [0, range_design).
        """
        pass

    #
    # Abstract methods
    #
    @abstractmethod
    def compute_design(self) -> None:
        """
        This function computes the weak design and saves it to memory.
        """

    @abstractmethod
    def get_set(self, index: Integral) -> np.ndarray:
        """
        Arguments:
            index: The index of a set from the family of sets, i.e. an integer in [0, m-1].

        Returns:
            FieldArray: The index-th set of the weak design.
        """
        pass
