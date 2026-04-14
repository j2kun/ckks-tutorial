from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from encoding import encode, EncodingParams, decode
from polynomial import (
    Polynomial,
    inverse_canonical_embedding_vandermonde,
    rotation_group,
)


# Example D-3.1.1 from
# https://fhetextbook.github.io/EncodingandDecoding.html#encoding-and-decoding
#
# Nb., this is commented out because in the reference above they use a
# different encoding method that does not flip the order of the second half of
# vector values when preparing a Hermitian input to the inverse canonical
# embedding. As a result, the inverse canonical embedding matrix they use is
# slightly different.
#
# def test_fhetextbook_d_3_1_1():
#     message = (1.1 + 4.3j, 3.5 - 1.4j)
#     params = EncodingParams(scale=1024, poly_modulus_degree=4)
#     plaintext = encode(message, params)
#     expected = np.array([2355, 1195, 1485, 2933], dtype=np.int64)
#     np.testing.assert_equal(plaintext.coefficients, expected)


def test_encode():
    message = np.array([1, 2])
    params = EncodingParams(scale=1024, poly_modulus_degree=4)
    plaintext = encode(message, params)
    expected = np.array([1536, -362, 0, 362], dtype=np.int64)
    np.testing.assert_equal(plaintext.coefficients, expected)


def test_encode_decode_exact():
    message = np.array([3 + 4j, 2 - 1j, 1 + 0j, 0 + 2j])
    params = EncodingParams(scale=2**20, poly_modulus_degree=8)
    plaintext = encode(message, params)
    decoded_message = decode(plaintext, params)
    np.testing.assert_allclose(decoded_message, message, atol=1e-06)


@given(
    message=st.lists(
        st.integers(min_value=0, max_value=1_000_000),
        min_size=16,
        max_size=16,
    )
)
def test_encode_decode_approx_equality(message):
    # encode + decode inherently loses precision due to the rounding step
    # of encoding. The amount of precision lost is decreased as the scale
    # is increased.
    params = EncodingParams(scale=2**40, poly_modulus_degree=32)
    encoded = encode(message, params)
    decoded = decode(encoded, params)
    np.testing.assert_allclose(message, decoded, rtol=0, atol=1e-06)


def test_encode_decode_with_ciphertext_modulus():
    """Encode and decode with modular reduction, verifying round-trip.

    Q must be small enough for all coefficients (including Q - |c|
    for negative c) to be exactly representable in float64 (< 2^53).
    """
    message = np.array([3 + 4j, 2 - 1j, 1 + 0j, 0 + 2j])
    Q = 2**40
    params = EncodingParams(
        scale=2**20, poly_modulus_degree=8, ciphertext_modulus=Q
    )
    plaintext = encode(message, params)

    # Coefficients should be in [0, Q)
    assert np.all(plaintext.coefficients >= 0)
    assert np.all(plaintext.coefficients < Q)

    decoded_message = decode(plaintext, params)
    np.testing.assert_allclose(decoded_message, message, atol=1e-06)


@given(
    message=st.lists(
        st.integers(min_value=0, max_value=1_000),
        min_size=16,
        max_size=16,
    )
)
def test_encode_decode_with_modulus_approx(message):
    """Round-trip with ciphertext modulus and integer messages.

    Parameters are chosen so max scaled coefficient (~ max_val * scale * sqrt(N))
    stays well below Q, and Q stays below 2^53 for float64 exactness.
    """
    params = EncodingParams(
        scale=2**30, poly_modulus_degree=32, ciphertext_modulus=2**50
    )
    encoded = encode(message, params)
    decoded = decode(encoded, params)
    np.testing.assert_allclose(message, decoded, rtol=0, atol=1e-06)


def test_encode_matches_vandermonde_reference():
    """The encoding should produce the same polynomial as an independent
    Vandermonde-matrix reference: scatter slot values into a standard-order
    Hermitian vector (using the rotation group), then invert via the
    standard Vandermonde matrix."""
    message = np.array([3 + 4j, 2 - 1j, 1 + 0j, 0 + 2j])
    n = 8
    params = EncodingParams(scale=2**20, poly_modulus_degree=n)

    # Encode via the main encode() function
    plaintext = encode(message, params)

    # Reference: place slot i's value at standard index (rot[i]-1)/2, and its
    # conjugate at (M-rot[i]-1)/2, then use standard Vandermonde inverse.
    rot = np.array(rotation_group(n))
    std_values = np.zeros(n, dtype=complex)
    std_values[(rot - 1) // 2] = message
    std_values[(2 * n - rot - 1) // 2] = np.conjugate(message)
    poly_ref = inverse_canonical_embedding_vandermonde(std_values)

    scaled = np.real(poly_ref.coefficients) * params.scale
    # Use same rounding as encode (half away from zero)
    rounded = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
    expected = Polynomial(rounded, n)

    np.testing.assert_equal(plaintext.coefficients, expected.coefficients)


def test_negative_coefficient_modular_reduction():
    """Verify that negative coefficients are properly reduced mod Q
    and recovered via center-lift."""
    message = np.array([-5.0, -10.0])
    Q = 2**30
    params = EncodingParams(scale=1024, poly_modulus_degree=4, ciphertext_modulus=Q)
    plaintext = encode(message, params)

    # All coefficients should be non-negative (mod Q)
    assert np.all(plaintext.coefficients >= 0)

    decoded = decode(plaintext, params)
    np.testing.assert_allclose(decoded, message, atol=1e-02)
