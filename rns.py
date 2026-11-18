"""Implementation of rns decomposition and recomposition."""

import numpy as np


def decompose(x: int, moduli: list[int]) -> np.ndarray:
    return np.array([x % p for p in moduli])


def recompose(residues: np.ndarray, moduli: list[int]) -> int:
    """Recompose an integer from RNS residues via the textbook formula.

    x = sum_i residues[i] * (Q/q_i) * [(Q/q_i)^{-1}]_{q_i} mod Q
    """
    result = 0

    # do arithmetic in python integers for bigint capabilities
    Q = np.prod(moduli, dtype=object)

    for residue, modulus in zip(residues, moduli):
        Q_div_qi = Q // modulus
        Q_div_qi_inv = pow(Q_div_qi, -1, modulus)
        result += (int(residue) * Q_div_qi * Q_div_qi_inv) % Q

    return result % Q
