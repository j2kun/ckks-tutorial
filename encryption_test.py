"""Tests for CKKS encrypt/decrypt routines."""

import random as std_random

import hypothesis
from hypothesis import strategies as st
import numpy as np

from ckks_types import Plaintext
from encoding import encode, decode
from encryption import (
    decrypt_symmetric,
    encrypt_symmetric,
    encrypt_asymmetric,
    decrypt_asymmetric,
)
from key_gen import generate_symmetric_private_key, generate_asymmetric_keypair
from params import EncryptionParams, EncodingParams, NTTParams
from ntt import NTT_32_BIT_PRIME, NTT_64_BIT_PRIME
from rng import SeededRandomSource, ZeroNoiseRandomSource

DEFAULT_ENCODING_PARAMS = EncodingParams(
    scale=2**20,
    poly_modulus_degree=8,
    coefficient_modulus=NTT_32_BIT_PRIME,
)


@hypothesis.given(
    st.lists(
        st.complex_numbers(min_magnitude=1, max_magnitude=10),
        min_size=4,
        max_size=4,
    ),
    st.floats(min_value=2**20, max_value=2**25),
    st.integers(min_value=0, max_value=10),
    st.sampled_from([NTT_32_BIT_PRIME, NTT_64_BIT_PRIME]),
)
@hypothesis.settings(deadline=3000)
def test_encrypt_decrypt_hypothesis(message, scale, seed, modulus):
    degree = DEFAULT_ENCODING_PARAMS.poly_modulus_degree
    encoding_params = EncodingParams(
        scale=scale,
        poly_modulus_degree=degree,
        coefficient_modulus=modulus,
    )
    params = EncryptionParams(degree=degree, modulus=modulus)
    ntt_params = NTTParams(degree=params.degree, modulus=params.modulus)
    random_source = SeededRandomSource(seed)

    sk = generate_symmetric_private_key(params, random_source)
    pt = encode(message, encoding_params, ntt_params)
    ct = encrypt_symmetric(pt, sk, params, random_source)
    decrypted_pt = decrypt_symmetric(ct, sk, params)
    decoded = decode(decrypted_pt, encoding_params)

    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)


@hypothesis.given(st.integers(min_value=0, max_value=2**32 - 1))
def test_exact_encrypt_decrypt(seed):
    # This tests encrypt/decrypt without cryptographic noise, which allows for
    # exactness checking to verify the underlying algebra.
    degree = 8
    modulus = NTT_32_BIT_PRIME
    params = EncryptionParams(degree=degree, modulus=modulus)
    rng = std_random.Random(seed)
    ntt_params = NTTParams(degree=params.degree, modulus=params.modulus)

    # Generate arbitrary plaintext exactly, to avoid encoding roundoff causing
    # equality comparison failures.
    pt_data = np.array(
        [rng.randint(-modulus // 2, modulus // 2) for _ in range(degree)],
        dtype=np.int64,
    )
    pt = Plaintext(pt_data, degree, coefficient_modulus=modulus, ntt_params=ntt_params)

    random_source = ZeroNoiseRandomSource()
    sk = generate_symmetric_private_key(params, random_source)
    ct = encrypt_symmetric(pt, sk, params, random_source)
    decrypted_pt = decrypt_symmetric(ct, sk, params)

    assert (decrypted_pt % params.modulus) == pt


def test_encrypt_decrypt():
    message = [(1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j)]
    seed = 0
    params = EncryptionParams(
        degree=DEFAULT_ENCODING_PARAMS.poly_modulus_degree, modulus=NTT_32_BIT_PRIME
    )
    ntt_params = NTTParams(degree=params.degree, modulus=params.modulus)
    random_source = SeededRandomSource(seed)

    sk = generate_symmetric_private_key(params, random_source)
    pt = encode(message, DEFAULT_ENCODING_PARAMS, ntt_params)
    ct = encrypt_symmetric(pt, sk, params, random_source)
    decrypted_pt = decrypt_symmetric(ct, sk, params)
    decoded = decode(decrypted_pt, DEFAULT_ENCODING_PARAMS)

    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)


def test_encrypt_decrypt_64_bit_overflow():
    message = [(1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j)]
    scale = 1048576.0
    seed = 0
    modulus = NTT_64_BIT_PRIME
    degree = DEFAULT_ENCODING_PARAMS.poly_modulus_degree

    encoding_params = EncodingParams(
        scale=scale,
        poly_modulus_degree=degree,
        coefficient_modulus=modulus,
    )
    params = EncryptionParams(degree=degree, modulus=modulus)
    ntt_params = NTTParams(degree=params.degree, modulus=params.modulus)
    random_source = SeededRandomSource(seed)

    sk = generate_symmetric_private_key(params, random_source)
    pt = encode(message, encoding_params, ntt_params)
    ct = encrypt_symmetric(pt, sk, params, random_source)
    decrypted_pt = decrypt_symmetric(ct, sk, params)
    decoded = decode(decrypted_pt, encoding_params)

    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)


@hypothesis.given(
    st.lists(
        st.complex_numbers(min_magnitude=1, max_magnitude=10),
        min_size=4,
        max_size=4,
    ),
    st.floats(min_value=2**20, max_value=2**25),
    st.integers(min_value=0, max_value=10),
    st.sampled_from([NTT_32_BIT_PRIME, NTT_64_BIT_PRIME]),
)
def test_encrypt_decrypt_asymmetric_hypothesis(message, scale, seed, modulus):
    degree = DEFAULT_ENCODING_PARAMS.poly_modulus_degree
    encoding_params = EncodingParams(
        scale=scale,
        poly_modulus_degree=degree,
        coefficient_modulus=modulus,
    )
    params = EncryptionParams(degree=degree, modulus=modulus)
    ntt_params = NTTParams(degree=params.degree, modulus=modulus)
    random_source = SeededRandomSource(seed)

    sk, pk = generate_asymmetric_keypair(params, random_source)
    pt = encode(message, encoding_params, ntt_params)
    ct = encrypt_asymmetric(pt, pk, params, random_source)
    decrypted_pt = decrypt_asymmetric(ct, sk, params)
    decoded = decode(decrypted_pt, encoding_params)

    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)


@hypothesis.given(st.integers(min_value=0, max_value=2**32 - 1))
def test_exact_encrypt_decrypt_asymmetric(seed):
    degree = 8
    modulus = NTT_32_BIT_PRIME
    params = EncryptionParams(degree=degree, modulus=modulus)
    ntt_params = NTTParams(degree=degree, modulus=modulus)
    rng = std_random.Random(seed)

    pt_data = np.array(
        [rng.randint(-modulus // 2, modulus // 2) for _ in range(degree)],
        dtype=np.int64,
    )
    pt = Plaintext(
        pt_data,
        degree,
        coefficient_modulus=modulus,
        ntt_params=ntt_params,
    )

    random_source = ZeroNoiseRandomSource()
    sk, pk = generate_asymmetric_keypair(params, random_source)
    ct = encrypt_asymmetric(pt, pk, params, random_source)
    decrypted_pt = decrypt_asymmetric(ct, sk, params)
    decrypted_pt = decrypted_pt % params.modulus

    assert decrypted_pt == pt


def test_encrypt_decrypt_asymmetric():
    message = [(1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j)]
    seed = 0
    modulus = NTT_32_BIT_PRIME
    degree = DEFAULT_ENCODING_PARAMS.poly_modulus_degree
    params = EncryptionParams(degree=degree, modulus=modulus)
    ntt_params = NTTParams(degree=degree, modulus=modulus)
    random_source = SeededRandomSource(seed)

    sk, pk = generate_asymmetric_keypair(params, random_source)
    pt = encode(message, DEFAULT_ENCODING_PARAMS, ntt_params)
    ct = encrypt_asymmetric(pt, pk, params, random_source)
    decrypted_pt = decrypt_asymmetric(ct, sk, params)
    decoded = decode(decrypted_pt, DEFAULT_ENCODING_PARAMS)

    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)


def test_encrypt_decrypt_asymmetric_64_bit_overflow():
    message = [(1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j)]
    scale = 1048576.0
    seed = 0
    modulus = NTT_64_BIT_PRIME
    degree = DEFAULT_ENCODING_PARAMS.poly_modulus_degree

    encoding_params = EncodingParams(
        scale=scale,
        poly_modulus_degree=degree,
        coefficient_modulus=modulus,
    )
    params = EncryptionParams(degree=degree, modulus=modulus)
    ntt_params = NTTParams(degree=params.degree, modulus=params.modulus)
    random_source = SeededRandomSource(seed)

    sk, pk = generate_asymmetric_keypair(params, random_source)
    pt = encode(message, encoding_params, ntt_params)
    ct = encrypt_asymmetric(pt, pk, params, random_source)
    decrypted_pt = decrypt_asymmetric(ct, sk, params)
    decoded = decode(decrypted_pt, encoding_params)

    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)
