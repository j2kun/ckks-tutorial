import numpy as np
import pytest
from hypothesis import given
import hypothesis.strategies as st

from decomposition import decompose, decompose_poly, recompose
from ntt import NTT_64_BIT_PRIME
from params import NTTParams
from polynomial import ModQPolynomial

MAX_UINT64 = 2**64 - 1
# Bases that evenly divide the 64-bit word.
BASE_LOGS = [1, 2, 4, 8, 16, 32]


def test_decompose_base_16():
    x = np.uint64(0x0123_4567_89AB_CDEF)
    expected = np.arange(16, dtype=np.uint64)
    np.testing.assert_array_equal(decompose(x, base_log=4), expected)


def test_decompose_base_256():
    x = np.uint64(0xFF00_0000_0000_0001)
    expected = np.array([0xFF, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint64)
    np.testing.assert_array_equal(decompose(x, base_log=8), expected)


def test_decompose_output_shape_and_dtype():
    digits = decompose(np.uint64(12345), base_log=4)
    assert digits.shape == (16,)
    assert digits.dtype == np.uint64


def test_decompose_zero():
    np.testing.assert_array_equal(
        decompose(np.uint64(0), base_log=4), np.zeros(16, dtype=np.uint64)
    )


def test_decompose_max_value():
    np.testing.assert_array_equal(
        decompose(np.uint64(MAX_UINT64), base_log=4),
        np.full(16, 15, dtype=np.uint64),
    )


def test_decompose_top_bit_set():
    # The top bit is where a signed/unsigned mixup would show up.
    digits = decompose(np.uint64(1 << 63), base_log=4)
    assert digits[0] == 8
    assert np.all(digits[1:] == 0)


def test_recompose_known_digits():
    digits = np.array([1, 2, 3], dtype=np.uint64)
    # Digits are most-significant first, aligned to the top of the 64-bit word.
    expected = (1 << 60) + (2 << 56) + (3 << 52)
    assert int(recompose(digits, base_log=4)) == expected


def test_recompose_full_word():
    digits = np.full(16, 15, dtype=np.uint64)
    assert int(recompose(digits, base_log=4)) == MAX_UINT64


@pytest.mark.parametrize("base_log", BASE_LOGS)
@given(x=st.integers(min_value=0, max_value=MAX_UINT64))
def test_decompose_recompose_roundtrip(base_log, x):
    digits = decompose(np.uint64(x), base_log=base_log)
    assert int(recompose(digits, base_log=base_log)) == x


@pytest.mark.parametrize("base_log", BASE_LOGS)
@given(x=st.integers(min_value=0, max_value=MAX_UINT64))
def test_digits_are_bounded_by_base(base_log, x):
    digits = decompose(np.uint64(x), base_log=base_log)
    assert digits.shape == (64 // base_log,)
    assert np.all(digits < (1 << base_log))


@pytest.mark.parametrize("base_log", BASE_LOGS)
@given(
    coeffs=st.lists(
        st.integers(min_value=0, max_value=NTT_64_BIT_PRIME - 1),
        min_size=8,
        max_size=8,
    )
)
def test_decompose_poly_matches_coefficientwise_decompose(base_log, coeffs):
    ntt_params = _NTT_PARAMS
    poly = ModQPolynomial(coeffs, 8, NTT_64_BIT_PRIME, ntt_params)
    digit_polys = decompose_poly(poly, base_log)

    assert len(digit_polys) == 64 // base_log
    for j, c in enumerate(coeffs):
        expected = decompose(np.uint64(c), base_log)
        actual = np.array([p.coefficients[j] for p in digit_polys], dtype=np.uint64)
        np.testing.assert_array_equal(actual, expected)


_NTT_PARAMS = NTTParams(degree=8, modulus=NTT_64_BIT_PRIME)
