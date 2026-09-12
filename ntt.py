from dataclasses import dataclass
import functools

import numpy as np
import galois

NTT_32_BIT_PRIME = 0x7FFFD801
NTT_64_BIT_PRIME = 0x1FFFFFFFFFE00001


@dataclass
class PrecomputedNTT:
    psi_powers: np.ndarray
    psi_inv_powers: np.ndarray


@functools.lru_cache(maxsize=10)
def precompute_ntt(q: int, n: int) -> PrecomputedNTT:
    psi = get_primitive_root(q, 2 * n)
    psi_inv = pow(psi, -1, q)

    psi_powers = np.array([pow(psi, i, q) for i in range(n)], dtype=np.int64)
    psi_inv_powers = np.array([pow(psi_inv, i, q) for i in range(n)], dtype=np.int64)

    return PrecomputedNTT(
        psi_powers=psi_powers,
        psi_inv_powers=psi_inv_powers,
    )


def get_primitive_root(q: int, n: int) -> int:
    """Find a primitive n-th root of unity modulo q."""
    if (q - 1) % n != 0:
        raise ValueError(
            f"Modulus {q} does not support a primitive {n}-th root of unity."
        )

    k = (q - 1) // n
    for i in range(2, q):
        r = pow(i, k, q)
        if pow(r, n // 2, q) != 1:
            return r
    raise ValueError(f"Failed to find primitive {n}-th root of unity for modulus {q}.")


@functools.lru_cache(maxsize=10)
def _get_gf(q: int):
    return galois.GF(q)


def ntt(x: np.ndarray, q: int) -> np.ndarray:
    """Forward NTT."""
    GF = _get_gf(q)
    return np.asarray(galois.ntt(GF(x % q)), dtype=np.int64)


def intt(x: np.ndarray, q: int) -> np.ndarray:
    """Inverse NTT."""
    GF = _get_gf(q)
    return np.asarray(galois.intt(GF(x % q)), dtype=np.int64)


def negacyclic_polymul_ntt(
    p1: np.ndarray, p2: np.ndarray, q: int, precomputed: PrecomputedNTT
) -> np.ndarray:
    """NTT-based multiplication modulo X^N + 1 and q."""
    GF = _get_gf(q)

    p1_gf = GF(p1 % q)
    p2_gf = GF(p2 % q)
    psi_powers_gf = GF(precomputed.psi_powers)
    psi_inv_powers_gf = GF(precomputed.psi_inv_powers)

    p1_pre = p1_gf * psi_powers_gf
    p2_pre = p2_gf * psi_powers_gf

    p1_ntt = galois.ntt(p1_pre)
    p2_ntt = galois.ntt(p2_pre)

    prod_ntt = p1_ntt * p2_ntt

    prod_pre = galois.intt(prod_ntt)
    prod = prod_pre * psi_inv_powers_gf

    return np.asarray(prod, dtype=np.int64)
