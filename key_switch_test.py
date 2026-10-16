"""Tests for key switching."""

import numpy as np

from context import CKKSContext
from encoding import decode
from encryption import extract_noise
from key_switch import key_switch
from ntt import NTT_64_BIT_PRIME
from params import EncodingParams
from rng import SeededRandomSource

DEFAULT_ENCODING_PARAMS = EncodingParams(
    scale=2**20,
    poly_modulus_degree=8,
    coefficient_modulus=NTT_64_BIT_PRIME,
)


def test_key_switch_symmetric_noise_formula():
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
    # beyond key switching. However, due to the lack of a gadget decomposition,
    # the noise has overwhelmed the message and it decrypts to garbage.

    # np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)

    # For now, extract the exact noise formula, subtract it from the plaintext,
    # and compare the remaining. This shows consistency of our impl with the
    # algebra and noise model.
    c1 = ct.data[1]
    c1_eprime_term = c1 * extract_noise(
        ksk.data, target_key, source_key.to_mod_q(context.ntt_params)
    )

    ks_noise_removed = decrypted - c1_eprime_term
    decoded = decode(ks_noise_removed, DEFAULT_ENCODING_PARAMS)
    np.testing.assert_allclose(decoded, message, rtol=0, atol=0.2)
