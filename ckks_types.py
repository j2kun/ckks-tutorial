"""Type definitions for CKKS."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from polynomial import IntPolynomial, ModQPolynomial

# A cleartext is a vector of possibly complex values
Cleartext = np.ndarray

# A plaintext is a polynomial (in coefficient form) in a ring
# Z/QZ[x] / (x^N + 1) for some N (power of two) and Q.
Plaintext = ModQPolynomial


# A private key is a polynomial s(x) with ternary coefficients.
PrivateKey = IntPolynomial


@dataclass(frozen=True)
class Ciphertext:
    """A CKKS Ciphertext.

    Mathematically represented as a pair of polynomials (c_0, c_1) in the ring
    R_Q = Z_Q[X] / (X^N + 1) such that decryption is a dot product with the
    secret key basis (1, s):
        < (c_0, c_1), (1, s) >_Q = [c_0 + c_1 * s]_Q = m + e

    Attributes:
        data: A tuple (bias, sample) = (c_0, c_1) where:
            bias (c_0): The encrypted message polynomial combined with error.
            sample (c_1): A uniformly random masking polynomial.
    """

    data: Tuple[ModQPolynomial, ModQPolynomial]
