import galois
import gmpy2
import numpy as np

NUMPY_INTEGRAL_TYPES = [np.dtype(_) for _ in np.typecodes["AllInteger"]]
NUMPY_REAL_TYPES = [np.dtype(_) for _ in np.typecodes["Float"]]
NUMPY_COMPLEX_TYPES = [np.dtype(_) for _ in np.typecodes["Complex"]]
INTEGERS = [1, gmpy2.mpz(1)] + [_.type(1) for _ in NUMPY_INTEGRAL_TYPES]
REALS = [0.25, gmpy2.mpfr(0.25), 2.5e-1] + [_.type(0.25) for _ in NUMPY_REAL_TYPES]
COMPLEX = [0.25 + 0j, gmpy2.mpc(0.25)] + [_.type(0.25) for _ in NUMPY_COMPLEX_TYPES]
NOT_NUMBERS_WITHOUT_NONE = [
    print,
    "forty-two",
    [4, 2],
    (4, 2),
    {"forty-two": 42},
    np.array([42]),
    galois.GF(128)(42),
]
NOT_NUMBERS = NOT_NUMBERS_WITHOUT_NONE + [None]
NOT_INTEGERS_WITHOUT_NONE = REALS + COMPLEX + NOT_NUMBERS_WITHOUT_NONE
NOT_INTEGERS = NOT_INTEGERS_WITHOUT_NONE + [None]
REALS_WITHOUT_INTEGERS = REALS
# Now that we have defined NOT_INTEGERS, we can add INTEGERS also to the REALS list
REALS = REALS_WITHOUT_INTEGERS + INTEGERS
NOT_REALS_WITHOUT_NONE = COMPLEX + NOT_NUMBERS_WITHOUT_NONE
NOT_REALS = NOT_REALS_WITHOUT_NONE + [None]
NOT_STRINGS = [42, ["string"], {"string": 42}, set("string"), True, print]
