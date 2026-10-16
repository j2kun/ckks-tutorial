"""Implementation of gadget decomposition."""

import numpy as np

from polynomial import ModQPolynomial

DecomposedInt = np.ndarray
TOTAL_BIT_LENGTH = 64


def _shifts(num_digits: int, base_log: int) -> np.ndarray:
    return np.uint64(TOTAL_BIT_LENGTH) - (
        np.arange(1, num_digits + 1, dtype=np.uint64) * np.uint64(base_log)
    )


def decompose(x: np.uint64, base_log: int) -> DecomposedInt:
    """Decompose a number in a given power-of-2 base, most significant first."""
    if TOTAL_BIT_LENGTH % base_log != 0:
        raise ValueError(f"base_log={base_log} must divide {TOTAL_BIT_LENGTH}")
    num_digits = TOTAL_BIT_LENGTH // base_log
    digit_mask = np.uint64((1 << base_log) - 1)
    return (np.uint64(x) >> _shifts(num_digits, base_log)) & digit_mask


def recompose(digits: DecomposedInt, base_log: int = 4) -> np.uint64:
    """The inverse of decompose."""
    digits = np.asarray(digits, dtype=np.uint64)
    return np.sum(digits << _shifts(digits.shape[0], base_log), dtype=np.uint64)


def decompose_poly(poly: ModQPolynomial, base_log: int) -> list[ModQPolynomial]:
    """Decompose a polynomial's coefficients.

    Returns a list of polynomials, one for each digit place, ordered from most
    significant to least significant. This must match the corresponding
    generation of the key-switching key.

    The i-th output polynomial's j-th coefficient is the i-th digit of the
    j-th coefficient of the input.
    """
    if TOTAL_BIT_LENGTH % base_log != 0:
        raise ValueError(f"base_log={base_log} must divide {TOTAL_BIT_LENGTH}")
    num_digits = TOTAL_BIT_LENGTH // base_log
    digit_mask = np.uint64((1 << base_log) - 1)

    # Coefficients are in [0, Q), and for now Q fits in a 64-bit integer.
    coeffs = poly.coefficients.astype(np.uint64)

    # Shape (num_digits, N): row i holds digit i of every coefficient.
    digits = (
        coeffs[np.newaxis, :] >> _shifts(num_digits, base_log)[:, np.newaxis]
    ) & digit_mask

    return [
        ModQPolynomial(
            row.astype(np.int64),
            modulus_degree=poly.modulus_degree,
            coefficient_modulus=poly.coefficient_modulus,
            ntt_params=poly.ntt_params,
        )
        for row in digits
    ]
