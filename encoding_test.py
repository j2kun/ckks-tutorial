from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from encoding import encode, EncodingParams, decode


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
