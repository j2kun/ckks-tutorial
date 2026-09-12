"""Tests for e2e CKKSContext."""

import numpy as np

from context import CKKSContext
from ntt import NTT_64_BIT_PRIME
from params import EncodingParams
from rng import SeededRandomSource


def test_symmetric_encrypt_decrypt():
    scale = 2**20
    seed = 837465
    message = np.array([1, 2, 3, 4], dtype=np.complex64)
    encoding_params = EncodingParams(
        scale=scale,
        poly_modulus_degree=8,
        coefficient_modulus=NTT_64_BIT_PRIME,
    )
    context = CKKSContext(encoding_params)
    context.rng = SeededRandomSource(seed)

    sk = context.generate_symmetric_private_key()
    pt = context.encode(message)
    ct = context.encrypt_symmetric(pt, sk)
    decrypted_pt = context.decrypt_symmetric(ct, sk)
    decoded = context.decode(decrypted_pt)

    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)


def test_asymmetric_encrypt_decrypt():
    scale = 2**20
    seed = 837465
    message = np.array([1, 2, 3, 4], dtype=np.complex64)
    encoding_params = EncodingParams(
        scale=scale,
        poly_modulus_degree=8,
        coefficient_modulus=NTT_64_BIT_PRIME,
    )
    context = CKKSContext(encoding_params)
    context.rng = SeededRandomSource(seed)

    sk, pk = context.generate_asymmetric_keypair()
    pt = context.encode(message)
    ct = context.encrypt_asymmetric(pt, pk)
    decrypted_pt = context.decrypt_asymmetric(ct, sk)
    decoded = context.decode(decrypted_pt)

    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)
