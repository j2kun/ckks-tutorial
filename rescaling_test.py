"""Tests for rescaling."""

import numpy as np

from context import CKKSContext
from encoding import decode
from ntt import NTT_64_BIT_PRIME
from params import EncodingParams
from rng import SeededRandomSource
from encoding import encode
from rescaling import rescale


def test_rescale():
    rescaled_encoding_params = EncodingParams(
        scale=2**20,
        poly_modulus_degree=8,
        # This prime is close to NTT_64_BIT_PRIME / 2**20
        coefficient_modulus=2199023255521,
    )
    squared_encoding_params = EncodingParams(
        scale=2**40,
        poly_modulus_degree=8,
        coefficient_modulus=NTT_64_BIT_PRIME,
    )

    seed = 837465
    message = np.array([1, 2, 3, 4], dtype=np.complex64)
    context = CKKSContext(rescaled_encoding_params)
    context.rng = SeededRandomSource(seed)

    # need a special context for the squared encoding parameters
    sq_context = CKKSContext(squared_encoding_params)
    sq_context.rng = SeededRandomSource(seed)
    sk = sq_context.generate_symmetric_private_key()
    pt = encode(message, squared_encoding_params, sq_context.ntt_params)
    ct = sq_context.encrypt_symmetric(pt, sk)

    ct_rescaled = rescale(
        ct,
        rescaled_encoding_params.coefficient_modulus,
        context.ntt_params,
    )

    decrypted_product = context.decrypt_symmetric(ct_rescaled, sk)
    decoded_product = decode(decrypted_product, rescaled_encoding_params)

    np.testing.assert_allclose(decoded_product, message, rtol=0, atol=0.1)


def test_mul_relin_rescale_symmetric():
    rescaled_encoding_params = EncodingParams(
        scale=2**20,
        poly_modulus_degree=8,
        # This prime is close to NTT_64_BIT_PRIME / 2**20
        coefficient_modulus=2199023255521,
    )
    # unlike test_rescale, we have to use the same scale for encoding as for
    # the rescaled params
    input_encoding_params = EncodingParams(
        scale=2**20,
        poly_modulus_degree=8,
        coefficient_modulus=NTT_64_BIT_PRIME,
    )

    seed = 837465
    message1 = np.array([1, 2, 3, 4], dtype=np.complex64)
    message2 = np.array([2, 3, 1, 2], dtype=np.complex64)
    expected = np.array([2, 6, 3, 8], dtype=np.complex64)
    context = CKKSContext(input_encoding_params)
    context.rng = SeededRandomSource(seed)
    rescaled_context = CKKSContext(rescaled_encoding_params)
    rescaled_context.rng = SeededRandomSource(seed)

    sk = context.generate_symmetric_private_key()
    context.initialize_relinearization_key(sk)
    pt1 = context.encode(message1)
    ct1 = context.encrypt_symmetric(pt1, sk)
    pt2 = context.encode(message2)
    ct2 = context.encrypt_symmetric(pt2, sk)

    ct_product = context.mul(ct1, ct2)
    ct_product = rescale(
        ct_product,
        rescaled_encoding_params.coefficient_modulus,
        rescaled_context.ntt_params,
    )

    decrypted_product = rescaled_context.decrypt_symmetric(ct_product, sk)
    decoded_product = decode(decrypted_product, rescaled_encoding_params)

    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)
