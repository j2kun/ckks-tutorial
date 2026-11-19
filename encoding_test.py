from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from encoding import encode, decode
from params import EncodingParams, NTTParams
from ntt import NTT_32_BIT_PRIME, NTT_64_BIT_PRIME
from test_utils import single_limb_encoding_and_ntt_params

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


# Encoding single-limb plaintexts, preserving the test cases from
# before RNS was introduced.


def test_encode():
    message = np.array([1, 2])
    params, ntt_params = single_limb_encoding_and_ntt_params(
        scale=1024,
        poly_modulus_degree=4,
        modulus=NTT_32_BIT_PRIME,
    )
    plaintext = encode(message, params, ntt_params)
    expected = [1536, -362, 0, 362]
    np.testing.assert_equal(
        plaintext.lift_to_signed_representative().coefficients, expected
    )


def test_encode_decode_exact():
    message = np.array([3 + 4j, 2 - 1j, 1 + 0j, 0 + 2j])
    params, ntt_params = single_limb_encoding_and_ntt_params(
        scale=2**20,
        poly_modulus_degree=8,
        modulus=NTT_32_BIT_PRIME,
    )
    plaintext = encode(message, params, ntt_params)
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
    params, ntt_params = single_limb_encoding_and_ntt_params(
        scale=2**40,
        poly_modulus_degree=32,
        modulus=NTT_64_BIT_PRIME,
    )
    encoded = encode(message, params, ntt_params)
    decoded = decode(encoded, params)
    np.testing.assert_allclose(decoded, message, rtol=0, atol=1e-06)


# Multi-limb RNS tests
from primes import generate_ntt_primes


def test_encode_rns():
    num_limbs = 3
    degree = 4
    primes = generate_ntt_primes(
        bit_width=60, poly_modulus_degree=degree, count=num_limbs
    )
    message = np.array([1, 2])
    params = EncodingParams(
        scale=1024,
        poly_modulus_degree=degree,
        moduli=primes,
    )
    ntt_params = [
        NTTParams(degree=params.poly_modulus_degree, modulus=prime) for prime in primes
    ]
    plaintext = encode(message, params, ntt_params)

    expected = [1536, -362, 0, 362]

    # In polynomial_test.py we test that going to/from RNS form is safe, so we
    # can simplify the assertion to check the lifted coefficients, matching the
    # single-limb test above.
    np.testing.assert_equal(
        plaintext.lift_to_signed_representative().coefficients, expected
    )


def test_encode_decode_rns_exact():
    num_limbs = 3
    degree = 8
    primes = generate_ntt_primes(
        bit_width=60, poly_modulus_degree=degree, count=num_limbs
    )
    message = np.array([3 + 4j, 2 - 1j, 1 + 0j, 0 + 2j])
    params = EncodingParams(
        scale=2**20,
        poly_modulus_degree=degree,
        moduli=primes,
    )
    ntt_params = [
        NTTParams(
            degree=params.poly_modulus_degree,
            modulus=prime,
        )
        for prime in primes
    ]
    plaintext = encode(message, params, ntt_params)
    decoded_message = decode(plaintext, params)
    np.testing.assert_allclose(decoded_message, message, atol=1e-06)


@given(
    message=st.lists(
        st.integers(min_value=0, max_value=1_000_000),
        min_size=16,
        max_size=16,
    )
)
def test_encode_decode_rns_approx_equality(message):
    num_limbs = 3
    degree = 32
    primes = generate_ntt_primes(
        bit_width=60, poly_modulus_degree=degree, count=num_limbs
    )
    # encode + decode inherently loses precision due to the rounding step
    # of encoding. The amount of precision lost is decreased as the scale
    # is increased.
    params = EncodingParams(
        scale=2**40,
        poly_modulus_degree=degree,
        moduli=primes,
    )
    ntt_params = [
        NTTParams(
            degree=params.poly_modulus_degree,
            modulus=prime,
        )
        for prime in primes
    ]
    encoded = encode(message, params, ntt_params)
    decoded = decode(encoded, params)
    np.testing.assert_allclose(decoded, message, rtol=0, atol=1e-06)


@given(
    num_limbs=st.integers(min_value=2, max_value=20),
    log_degree=st.integers(min_value=2, max_value=8),
)
def test_encode_decode_rns_limb_sizes(num_limbs, log_degree):
    degree = 2**log_degree
    primes = generate_ntt_primes(
        bit_width=60, poly_modulus_degree=degree, count=num_limbs
    )
    message = [complex(k) / 10 for k in range(degree // 2)]
    params = EncodingParams(
        scale=2**40,
        poly_modulus_degree=degree,
        moduli=primes,
    )
    ntt_params = [
        NTTParams(
            degree=params.poly_modulus_degree,
            modulus=prime,
        )
        for prime in primes
    ]
    encoded = encode(message, params, ntt_params)
    decoded = decode(encoded, params)
    np.testing.assert_allclose(decoded, message, rtol=0, atol=1e-06)
