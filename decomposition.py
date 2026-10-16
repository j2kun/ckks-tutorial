"""Implementation of gadget decomposition."""

import numpy as np

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
