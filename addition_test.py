"""Tests for ciphertext addition."""

from hypothesis import strategies as st
import hypothesis
import numpy as np

from context import CKKSContext
from ntt import NTT_64_BIT_PRIME
from params import EncodingParams
from rng import SeededRandomSource

DEFAULT_ENCODING_PARAMS = EncodingParams(
    scale=2**20,
    poly_modulus_degree=8,
    coefficient_modulus=NTT_64_BIT_PRIME,
)


def test_add_two_symmetric():
    seed = 837465
    message1 = np.array([1, 2, 3, 4], dtype=np.complex64)
    message2 = np.array([2, 3, 1, 2], dtype=np.complex64)
    expected = np.array([3, 5, 4, 6], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    sk = context.generate_symmetric_private_key()
    pt1 = context.encode(message1)
    ct1 = context.encrypt_symmetric(pt1, sk)
    pt2 = context.encode(message2)
    ct2 = context.encrypt_symmetric(pt2, sk)

    ct_sum = context.add(ct1, ct2)

    decrypted_sum = context.decrypt_symmetric(ct_sum, sk)
    decoded_sum = context.decode(decrypted_sum)

    np.testing.assert_allclose(decoded_sum, expected, rtol=0, atol=0.2)


def test_add_two_asymmetric():
    seed = 837465
    message1 = np.array([1, 2, 3, 4], dtype=np.complex64)
    message2 = np.array([2, 3, 1, 2], dtype=np.complex64)
    expected = np.array([3, 5, 4, 6], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    sk, pk = context.generate_asymmetric_keypair()
    pt1 = context.encode(message1)
    ct1 = context.encrypt_asymmetric(pt1, pk)
    pt2 = context.encode(message2)
    ct2 = context.encrypt_asymmetric(pt2, pk)

    ct_sum = context.add(ct1, ct2)

    decrypted_sum = context.decrypt_asymmetric(ct_sum, sk)
    decoded_sum = context.decode(decrypted_sum)

    np.testing.assert_allclose(decoded_sum, expected, rtol=0, atol=0.2)


@hypothesis.given(
    st.lists(
        st.complex_numbers(min_magnitude=1, max_magnitude=10),
        min_size=4,
        max_size=4,
    ),
    st.lists(
        st.complex_numbers(min_magnitude=1, max_magnitude=10),
        min_size=4,
        max_size=4,
    ),
    st.floats(min_value=2**20, max_value=2**25),
    st.integers(min_value=0, max_value=10),
)
def test_add_asymmetric_hypothesis(message1, message2, scale, seed):
    encoding_params = EncodingParams(
        scale=scale,
        poly_modulus_degree=8,
        coefficient_modulus=NTT_64_BIT_PRIME,
    )

    message1 = np.array(message1, dtype=np.complex64)
    message2 = np.array(message2, dtype=np.complex64)
    expected = message1 + message2
    context = CKKSContext(encoding_params)
    context.rng = SeededRandomSource(seed)

    sk, pk = context.generate_asymmetric_keypair()
    pt1 = context.encode(message1)
    ct1 = context.encrypt_asymmetric(pt1, pk)
    pt2 = context.encode(message2)
    ct2 = context.encrypt_asymmetric(pt2, pk)

    ct_sum = context.add(ct1, ct2)

    decrypted_sum = context.decrypt_asymmetric(ct_sum, sk)
    decoded_sum = context.decode(decrypted_sum)

    np.testing.assert_allclose(decoded_sum, expected, rtol=0, atol=0.2)
