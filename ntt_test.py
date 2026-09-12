import numpy as np
from hypothesis import given
import hypothesis.strategies as st

from ntt import get_primitive_root, negacyclic_polymul_ntt, NTT_32_BIT_PRIME
from params import NTTParams


def test_primitive_root():
    # degree 8 needs 16-th root
    root = get_primitive_root(NTT_32_BIT_PRIME, 16)
    assert pow(root, 16, NTT_32_BIT_PRIME) == 1
    assert pow(root, 8, NTT_32_BIT_PRIME) != 1


def test_ntt_multiplication_simple():
    # x^2 + 2x + 3
    p1 = np.array([3, 2, 1, 0, 0, 0, 0, 0], dtype=np.int64)
    # 4x + 5
    p2 = np.array([5, 4, 0, 0, 0, 0, 0, 0], dtype=np.int64)

    # expected: (x^2 + 2x + 3)(4x+5) = 4x^3 + 13x^2 + 22x + 15
    prod = negacyclic_polymul_ntt(
        p1, p2, NTT_32_BIT_PRIME, NTTParams(8, NTT_32_BIT_PRIME).precomputed
    )
    expected = np.array([15, 22, 13, 4, 0, 0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(prod, expected)


def test_ntt_multiplication_negacyclic_wrap():
    # x^7 * x = x^8 = -1 mod X^8 + 1
    p1 = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)
    p2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    prod = negacyclic_polymul_ntt(
        p1, p2, NTT_32_BIT_PRIME, NTTParams(8, NTT_32_BIT_PRIME).precomputed
    )
    # -1 mod NTT_32_BIT_PRIME is NTT_32_BIT_PRIME - 1
    expected = np.array([NTT_32_BIT_PRIME - 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(prod, expected)


# The reference implementation below is from
# https://www.jeremykun.com/2022/12/09/negacyclic-polynomial-multiplication/
def cylic_matrix(c: np.array) -> np.ndarray:
    """Generates a cyclic matrix with each row of the input shifted.

    For input: [1, 2, 3], generates the following matrix:

        [[1 2 3]
         [2 3 1]
         [3 1 2]]
    """
    c = np.asarray(c).ravel()
    a, b = np.ogrid[0 : len(c), 0 : -len(c) : -1]
    indx = a + b
    return c[indx]


def negacyclic_polymul_toeplitz(p1, p2):
    n = len(p1)

    # Generates a sign matrix with 1s below the diagonal and -1 above.
    up_tri = np.tril(np.ones((n, n), dtype=int), 0)
    low_tri = np.triu(np.ones((n, n), dtype=int), 1) * -1
    sign_matrix = up_tri + low_tri

    cyclic_matrix = cylic_matrix(p1)
    toeplitz_p1 = sign_matrix * cyclic_matrix
    return np.matmul(toeplitz_p1, p2)


@given(
    a=st.lists(st.integers(min_value=0, max_value=1_000_000), min_size=8, max_size=8),
    b=st.lists(st.integers(min_value=0, max_value=1_000_000), min_size=8, max_size=8),
)
def test_ntt_multiplication_hypothesis(a, b):
    # Reference implementation using numpy convolve and negacyclic wrap
    p1 = np.array(a, dtype=np.int64)
    p2 = np.array(b, dtype=np.int64)

    expected = negacyclic_polymul_toeplitz(p1, p2)
    expected = expected % NTT_32_BIT_PRIME
    actual = negacyclic_polymul_ntt(
        p1,
        p2,
        NTT_32_BIT_PRIME,
        NTTParams(8, NTT_32_BIT_PRIME).precomputed,
    )
    np.testing.assert_array_equal(actual, expected)
