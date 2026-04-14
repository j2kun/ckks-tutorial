import numpy as np
from typing import Iterable


def is_power_of_two(x: int) -> bool:
    return (x != 0) and (x & (x - 1) == 0)


class Polynomial:
    """A class representing a univariate polynomial with a ring modulus x^N + 1."""

    def __init__(self, coefficients: np.ndarray | Iterable[float], modulus_degree: int):
        """Construct a polynomial.

        Args:
            coefficients: the coefficients of the polynomial.
            modulus_degree: the degree N of x^N + 1. Must be a power of two.
        """
        self.coefficients = np.array(coefficients)
        self.modulus_degree = (
            modulus_degree if modulus_degree is not None else len(coefficients)
        )

        if self.modulus_degree != self.coefficients.shape[0]:
            raise ValueError(
                f"twoNodulus degree {self.modulus_degree} must be at least "
                f"the number of coefficients {self.coefficients.shape[0]}."
            )
        if not is_power_of_two(self.modulus_degree):
            raise ValueError(
                "twoNodulus degree must be a power of two, but was "
                f"{self.modulus_degree}."
            )

    def __repr__(self):
        return f"Polynomial({self.coefficients.tolist()})"

    def __eq__(self, other) -> bool:
        # Equality is nontrivial with modular polynomials, but for now we can
        # compare coefficients exactly.
        if not isinstance(other, Polynomial):
            return False
        return self.modulus_degree == other.modulus_degree and np.array_equal(
            self.coefficients, other.coefficients
        )

    def assert_close(self, other, rtol=1e-5, atol=1e-8):
        if not isinstance(other, Polynomial):
            raise AssertionError(f"Other object is not a Polynomial: {other}")
        if self.modulus_degree != other.modulus_degree:
            raise AssertionError(
                f"Modulus degrees differ: {self.modulus_degree} vs "
                f"{other.modulus_degree}"
            )
        np.testing.assert_allclose(
            self.coefficients, other.coefficients, rtol=rtol, atol=atol
        )


def vandermonde_matrix(n: int) -> np.ndarray:
    """Computes the Vandermonde matrix used for canonical embedding."""
    twoN = 2 * n

    # Primitive roots of X^N + 1 are odd powers of the first primitive root of
    # unity.
    first_root = np.exp(2j * np.pi / twoN)
    odd_powers = np.arange(1, twoN, 2)
    roots = first_root**odd_powers

    powers = np.arange(n)
    vandermonde = roots[:, np.newaxis] ** powers
    return vandermonde


# For correctness testing, compare the FFT method to a more straightforward
# Vandermonde matrix method.
def canonical_embedding_vandermonde(poly: Polynomial) -> np.ndarray:
    """Computes the canonical embedding of a polynomial.

    This evaluates the polynomial at primitive complex roots of unity,
    using a Vandermonde matrix method.
    """
    vandermonde = vandermonde_matrix(poly.modulus_degree)
    # Vandermonde matrix turns evaluation into a matrix-vector mul
    return vandermonde @ poly.coefficients

def inverse_canonical_embedding_vandermonde(values: np.ndarray) -> Polynomial:
    """Computes the inverse canonical embedding of a polynomial.

    The input `values` must be N complex values representing the evaluation of
    the result Polynomial at primitive (2N)-th roots of unity. In this case,
    the output is guaranteed to be real-valued (up to floating point roundoff
    error).
    """
    n = len(values)
    vandermonde = vandermonde_matrix(n)
    inv_vand = np.linalg.inv(vandermonde)
    return Polynomial(inv_vand @ values, n)


def canonical_embedding(poly: Polynomial) -> np.ndarray:
    """Computes the canonical embedding of a polynomial.

    This evaluates the polynomial at primitive complex roots of unity,
    using an FFT-based method.
    """
    poly_coeffs = poly.coefficients
    N = poly.modulus_degree

    # 2N-point FFT evaluates at all 2N-th roots: omega^0, omega^1, ...,
    # omega^{2N-1}. But return only the odd entries for the primitive roots.
    padded = np.concatenate([poly_coeffs, np.zeros(N)])
    fft_result = np.fft.ifft(padded) * (2 * N)
    return fft_result[np.arange(1, 2 * N, 2)]


def inverse_canonical_embedding(values: np.ndarray) -> Polynomial:
    """
    Invert the canonical embedding using FFT.

    The input `values` must be N complex values representing the evaluation of
    the result Polynomial at primitive (2N)-th roots of unity. In this case,
    the output is guaranteed to be real-valued (up to floating point roundoff
    error).
    """
    N = len(values)
    if not is_power_of_two(N):
        raise ValueError(
            f"Length of values must be a power of two, but was {len(values)}."
        )

    M = 2 * N
    ifft_result = np.fft.fft(values)
    zeta = np.exp(2j * np.pi / M)
    phase_factors = zeta ** (-np.arange(N))
    coeffs = (phase_factors * ifft_result) / N
    return Polynomial(coeffs, modulus_degree=N)


# --- OpenFHE-compatible encoding support ---
#
# OpenFHE uses a different root ordering for the canonical embedding,
# based on the Galois group of the cyclotomic ring. Slot i maps to the
# root zeta^{5^i mod M}, where M = 2N. This ordering (vs. consecutive
# odd powers) enables slot rotations: rotating by 1 slot corresponds
# to the automorphism X -> X^5.
#
# The "special FFT" algorithms (FFTSpecial / FFTSpecialInv) from
# Chen/Cheon/Kim/Kim (eprint 2018/1043) exploit this structure to
# work directly on N/2 complex slot values without an explicit
# Hermitian extension to N values.


def rotation_group(n: int) -> list[int]:
    """Compute root indices for OpenFHE-compatible CKKS encoding.

    For the cyclotomic ring Z[x]/(x^N + 1) with M = 2N, returns
    [5^i mod M for i in range(N/2)]. The generator 5 generates
    the subgroup of (Z/MZ)* that acts on roots without complex
    conjugation, enabling slot rotations via Galois automorphisms.
    """
    M = 2 * n
    group = []
    power = 1
    for _ in range(n // 2):
        group.append(power)
        power = (power * 5) % M
    return group


def fft_special(vals: np.ndarray, n: int) -> np.ndarray:
    """Forward special FFT matching OpenFHE's canonical embedding.

    fft_special evaluates a polynomial at the same N roots as the
    standard canonical_embedding (the odd powers of zeta_{2N}), just
    indexed via the rotation group rather than in consecutive odd order.
    So it's canonical_embedding + a gather by rot[].

    Input: N/2 complex values packing N real polynomial coefficients as
    coeffs[:N/2] + 1j * coeffs[N/2:]. Returns N/2 slot values.
    """
    # Unpack to N real polynomial coefficients, then evaluate at all N
    # odd-powered roots in standard order (zeta^1, zeta^3, ..., zeta^{2N-1}).
    coeffs = np.concatenate([np.real(vals), np.imag(vals)])
    std_emb = canonical_embedding(Polynomial(coeffs, n))

    # Slot i corresponds to zeta^{rot[i]}, which is at standard index (rot[i]-1)/2.
    rot = np.array(rotation_group(n))
    return std_emb[(rot - 1) // 2]


def fft_special_inv(vals: np.ndarray, n: int) -> np.ndarray:
    """Inverse special FFT matching OpenFHE's inverse canonical embedding.

    Inverse of fft_special: scatter slot values (and their conjugates)
    into the N-element standard-ordered Hermitian vector, then reuse
    the standard inverse_canonical_embedding.

    Input: N/2 complex slot values. Returns N/2 complex values packing
    N real coefficients as coeffs[:N/2] + 1j * coeffs[N/2:].
    """
    rot = np.array(rotation_group(n))

    # Build the length-N Hermitian vector in standard order. Slot i
    # (root zeta^{rot[i]}) sits at standard index (rot[i]-1)/2; its
    # conjugate (root zeta^{M-rot[i]}) sits at (M-rot[i]-1)/2.
    std_values = np.zeros(n, dtype=complex)
    std_values[(rot - 1) // 2] = vals
    std_values[(2 * n - rot - 1) // 2] = np.conj(vals)

    real_coeffs = np.real(inverse_canonical_embedding(std_values).coefficients)
    num_slots = n // 2
    return real_coeffs[:num_slots] + 1j * real_coeffs[num_slots:]
