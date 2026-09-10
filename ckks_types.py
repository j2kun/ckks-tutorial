"""Type definitions for CKKS."""

import numpy as np
from polynomial import ModQPolynomial

# A cleartext is a vector of possibly complex values
Cleartext = np.ndarray

# A plaintext is a polynomial (in coefficient form) in a ring
# Z/QZ[x] / (x^N + 1) for some N (power of two) and Q.
Plaintext = ModQPolynomial
