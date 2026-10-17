"""Tests for key switching."""

import numpy as np
import pytest

from context import CKKSContext
from encoding import decode
from key_switch import key_switch
from ntt import NTT_64_BIT_PRIME
from params import EncodingParams, KeySwitchingKeyParams
from rng import SeededRandomSource

DEFAULT_ENCODING_PARAMS = EncodingParams(
    scale=2**20,
    poly_modulus_degree=8,
    coefficient_modulus=NTT_64_BIT_PRIME,
)


def test_key_switch_symmetric():
    seed = 837465
    message = np.array([1, 2, 3, 4], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    source_key = context.generate_symmetric_private_key()
    target_key = context.generate_symmetric_private_key()
    pt = context.encode(message)
    ct = context.encrypt_symmetric(pt, source_key)
    ksk = context.generate_symmetric_key_switching_key(source_key, target_key)

    switched_ct = key_switch(ct, ksk)
    decrypted = context.decrypt_symmetric(switched_ct, target_key)
    decoded = decode(decrypted, DEFAULT_ENCODING_PARAMS)

    # Should get back the original message, since no computation was performed
    # beyond key switching.
    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)


@pytest.mark.parametrize("log_base", [1, 2, 4, 8])
def test_key_switch_symmetric_various_bases(log_base):
    seed = 9182
    message = np.array([1.5, -2, 3j, 4], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS, KeySwitchingKeyParams(log_base))
    context.rng = SeededRandomSource(seed)

    source_key = context.generate_symmetric_private_key()
    target_key = context.generate_symmetric_private_key()
    ct = context.encrypt_symmetric(context.encode(message), source_key)
    ksk = context.generate_symmetric_key_switching_key(source_key, target_key)

    switched_ct = key_switch(ct, ksk)
    decoded = context.decode(context.decrypt_symmetric(switched_ct, target_key))
    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)


def test_key_switch_asymmetric():
    seed = 4321
    message = np.array([1, 2, 3, 4], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    source_sk, source_pk = context.generate_asymmetric_keypair()
    target_sk, target_pk = context.generate_asymmetric_keypair()
    ct = context.encrypt_asymmetric(context.encode(message), source_pk)
    ksk = context.generate_asymmetric_key_switching_key(source_sk, target_pk)

    switched_ct = key_switch(ct, ksk)
    decoded = context.decode(context.decrypt_asymmetric(switched_ct, target_sk))
    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)
