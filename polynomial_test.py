import math
import numpy as np
from hypothesis import given
import hypothesis.strategies as st

from polynomial import Polynomial
from polynomial import canonical_embedding
from polynomial import canonical_embedding_vandermonde
from polynomial import inverse_canonical_embedding
from polynomial import inverse_canonical_embedding_vandermonde
from polynomial import rotation_group
from polynomial import fft_special
from polynomial import fft_special_inv


def test_canonical_embedding():
    polynomial = Polynomial(
        [5 / 2, math.sqrt(2), 5 / 2, math.sqrt(2) / 2], modulus_degree=4
    )
    # This vector would be the message (3+4j, 2-1j), but here it has been
    # extended to include its complex conjugates.
    expected = (3 + 4j, 2 - 1j, 2 + 1j, 3 - 4j)
    np.testing.assert_allclose(canonical_embedding(polynomial), expected)


def test_inverse_canonical_embedding():
    vector = np.array([3 + 4j, 2 - 1j, 2 + 1j, 3 - 4j])
    expected = Polynomial(
        [5 / 2, math.sqrt(2), 5 / 2, math.sqrt(2) / 2], modulus_degree=4
    )
    actual = inverse_canonical_embedding(vector)
    actual.assert_close(expected)


@given(
    coeffs=st.lists(
        st.floats(min_value=0, max_value=1e6, allow_infinity=False, allow_nan=False),
        min_size=1,
        max_size=16,
    )
)
def test_canonical_embedding_equivalence(coeffs):
    # zero-extend coeffs to next power of two
    modulus_degree = 1 << (len(coeffs) - 1).bit_length()
    coeffs += [0] * (modulus_degree - len(coeffs))
    poly = Polynomial(coeffs, modulus_degree=modulus_degree)

    result_fft = canonical_embedding(poly)
    result_vandermonde = canonical_embedding_vandermonde(poly)
    np.testing.assert_allclose(result_fft, result_vandermonde, rtol=0, atol=1e-7)


@given(
    values=st.lists(
        st.integers(min_value=0, max_value=1_000_000),
        min_size=1,
        max_size=16,
    )
)
def test_inverse_canonical_embedding_equivalence(values):
    # zero-extend values to next power of two
    modulus_degree = 1 << (len(values) - 1).bit_length()
    values += [0] * (modulus_degree - len(values))

    result_fft = inverse_canonical_embedding(values).coefficients
    result_vandermonde = inverse_canonical_embedding_vandermonde(values).coefficients
    np.testing.assert_allclose(result_fft, result_vandermonde, rtol=0, atol=1e-06)

@given(
    coeffs=st.lists(
        st.floats(min_value=0, max_value=1e6, allow_infinity=False, allow_nan=False),
        min_size=1,
        max_size=16,
    )
)
def test_embed_then_inverse(coeffs):
    # zero-extend coeffs to next power of two
    modulus_degree = 1 << (len(coeffs) - 1).bit_length()
    coeffs += [0] * (modulus_degree - len(coeffs))
    poly = Polynomial(coeffs, modulus_degree=modulus_degree)

    embedded = canonical_embedding(poly)
    recovered_poly = inverse_canonical_embedding(embedded)
    recovered_poly.assert_close(poly, rtol=0, atol=1e-7)


@given(
    vector=st.lists(
        st.complex_numbers(
            min_magnitude=0, max_magnitude=1e6, allow_infinity=False, allow_nan=False
        ),
        min_size=1,
        max_size=16,
    )
)
def test_inverse_then_embed(vector):
    next_power_of_two = 1 << (len(vector) - 1).bit_length()
    vector += [0] * (next_power_of_two - len(vector))

    # Ensure vector has the conjugate symmetry property. In this test we just
    # double the vector length by appending conjugates in reverse.
    vector += [np.conj(x) for x in reversed(vector)]

    recovered_poly = inverse_canonical_embedding(np.array(vector))
    reembedded = canonical_embedding(recovered_poly)
    np.testing.assert_allclose(reembedded, vector, rtol=0, atol=1e-7)


# --- OpenFHE-compatible special FFT tests ---


def test_rotation_group_n4():
    # 5^0 mod 8 = 1, 5^1 mod 8 = 5
    assert rotation_group(4) == [1, 5]


def test_rotation_group_n8():
    # 5^0=1, 5^1=5, 5^2=25%16=9, 5^3=45%16=13
    assert rotation_group(8) == [1, 5, 9, 13]


def test_rotation_group_n16():
    expected = [1, 5, 25, 29, 17, 21, 9, 13]
    assert rotation_group(16) == expected


def test_rotation_group_covers_all_roots():
    """The rotation group indices and their conjugates should cover
    all odd numbers in [1, 2N-1], i.e., all roots of X^N+1."""
    for n in [4, 8, 16, 32]:
        M = 2 * n
        rot = rotation_group(n)
        all_roots = set(rot) | {M - r for r in rot}
        expected = set(range(1, M, 2))
        assert all_roots == expected, f"Failed for n={n}"


def test_fft_special_inv_n4_known_values():
    """Verify fft_special_inv on a known input for N=4."""
    msg = np.array([1 + 0j, 2 + 0j])
    packed = fft_special_inv(msg, 4)
    # From manual calculation: packed = [(m0+m1)/2, (m0-m1)*zeta^7/2]
    zeta7 = np.exp(2j * np.pi * 7 / 8)
    expected = np.array([1.5, (-1) * zeta7 / 2])
    np.testing.assert_allclose(packed, expected, atol=1e-12)


def test_fft_special_roundtrip_n4():
    """fft_special(fft_special_inv(msg)) == msg for N=4."""
    msg = np.array([3 + 4j, 2 - 1j])
    packed = fft_special_inv(msg, 4)
    recovered = fft_special(packed, 4)
    np.testing.assert_allclose(recovered, msg, atol=1e-12)


@given(
    values=st.lists(
        st.complex_numbers(
            min_magnitude=0, max_magnitude=1e6,
            allow_infinity=False, allow_nan=False,
        ),
        min_size=1,
        max_size=16,
    )
)
def test_fft_special_roundtrip(values):
    """fft_special(fft_special_inv(msg)) == msg for arbitrary inputs."""
    num_slots = 1 << (len(values) - 1).bit_length()
    values += [0] * (num_slots - len(values))
    n = num_slots * 2  # ring dimension N = 2 * num_slots

    msg = np.array(values, dtype=complex)
    packed = fft_special_inv(msg, n)
    recovered = fft_special(packed, n)
    np.testing.assert_allclose(recovered, msg, rtol=0, atol=1e-7)


