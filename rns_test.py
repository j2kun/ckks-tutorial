import math

import numpy as np
from hypothesis import given
import hypothesis.strategies as st

from rns import decompose, recompose
from primes import generate_ntt_primes

PRIMES = generate_ntt_primes(bit_width=60, poly_modulus_degree=1024, count=20)


def test_decompose():
    primes = [3, 4, 5]
    x = 39
    expected = np.array([0, 3, 4])
    np.testing.assert_equal(expected, decompose(x, primes))


def test_recompose():
    primes = [3, 4, 5]
    residues = np.array([0, 3, 4])
    expected = 39
    assert expected == recompose(residues, primes)


def prime_subsets(
    min_size: int = 1, max_size: int = len(PRIMES)
) -> st.SearchStrategy[list[int]]:
    """A strategy for a list of distinct primes drawn from PRIMES."""
    return st.lists(
        st.sampled_from(PRIMES), min_size=min_size, max_size=max_size, unique=True
    )


@st.composite
def primes_and_value(draw: st.DrawFn) -> tuple[list[int], int]:
    """A list of primes with product Q, and an integer x in [0, Q)."""
    primes = draw(prime_subsets())
    Q = math.prod(primes)
    x = draw(st.integers(min_value=0, max_value=Q - 1))
    return primes, x


@st.composite
def primes_and_residues(draw: st.DrawFn) -> tuple[list[int], np.ndarray]:
    """A list of primes, and one residue in [0, q_i) for each prime q_i."""
    primes = draw(prime_subsets())
    residues = np.array(
        [draw(st.integers(min_value=0, max_value=p - 1)) for p in primes],
        dtype=np.int64,
    )
    return primes, residues


@given(primes_and_value())
def test_decompose_recompose(primes_and_x):
    primes, x = primes_and_x
    residues = decompose(x, primes)
    assert x == recompose(residues, primes)


@given(primes_and_residues())
def test_recompose_decompose(primes_and_res):
    primes, residues = primes_and_res
    x = recompose(residues, primes)
    np.testing.assert_equal(residues, decompose(x, primes))
