from ckks_types import KeySwitchingKey, PrivateKey, PublicKey
from decomposition import TOTAL_BIT_LENGTH
from encryption import encrypt_symmetric, encrypt_asymmetric
from encryption import sample_uniform_rns_poly
from params import EncryptionParams, NTTParams, KeySwitchingKeyParams
from polynomial import IntPolynomial, RNSPolynomial, ModQPolynomial
from typing import Tuple
import rng


def generate_symmetric_private_key(
    params: EncryptionParams,
    random_source: rng.RandomSource,
) -> PrivateKey:
    return PrivateKey(
        random_source.gen_ternary_poly(params.degree),
        modulus_degree=params.degree,
    )


def generate_asymmetric_keypair(
    params: EncryptionParams,
    ntt_params: list[NTTParams],
    random_source: rng.RandomSource,
) -> Tuple[PrivateKey, PublicKey]:
    sk = generate_symmetric_private_key(params, random_source)

    a = sample_uniform_rns_poly(random_source, params.degree, ntt_params)
    e = IntPolynomial(
        random_source.gen_gaussian_poly(params.degree),
        modulus_degree=params.degree,
    ).to_rns(
        ntt_params=ntt_params,
    )

    b = -(a * sk) + e
    pk = PublicKey(data=(b, a))
    return sk, pk


def _scaled_source_keys(
    source_key: PrivateKey,
    ks_params: KeySwitchingKeyParams,
    ntt_params: NTTParams,
) -> list[ModQPolynomial]:
    """Compute B^i * s mod Q for each digit place i = k-1, ..., 0."""
    B = 1 << ks_params.log_base
    num_digits = TOTAL_BIT_LENGTH // ks_params.log_base
    # Because the source_key may not be ternary (e.g., for relinearization
    # it is s^2 for some ternary s), we need to ensure we don't overflow
    # int64, which can happen if N is large.
    Q = ntt_params.modulus
    python_int_coeffs = source_key.coefficients.astype(object)
    return [
        IntPolynomial(
            (pow(B, i, Q) * python_int_coeffs) % Q,
            source_key.modulus_degree,
        ).to_mod_q(ntt_params)
        for i in reversed(range(num_digits))
    ]


def generate_symmetric_key_switching_key(
    source_key: PrivateKey,
    target_key: PrivateKey,
    params: EncryptionParams,
    ks_params: KeySwitchingKeyParams,
    ntt_params: NTTParams,
    random_source: rng.RandomSource,
) -> KeySwitchingKey:
    """Generate a key switching key from source_key to target_key.

    This key can be used by key_switch to convert a ciphertext
    encrypted under source_key to be encrypted under target_key.
    """
    # Encrypt B^i * s for all i
    cts = [
        encrypt_symmetric(scaled_key, target_key, params, random_source)
        for scaled_key in _scaled_source_keys(source_key, ks_params, ntt_params)
    ]
    return KeySwitchingKey(data=cts, params=ks_params)


def generate_asymmetric_key_switching_key(
    source_key: PrivateKey,
    target_key: PublicKey,
    params: EncryptionParams,
    ks_params: KeySwitchingKeyParams,
    ntt_params: NTTParams,
    random_source: rng.RandomSource,
) -> KeySwitchingKey:
    """Generate a key switching key from source_key to target_key.

    This key can be used by key_switch to convert a ciphertext
    encrypted under source_key to be encrypted under target_key.
    """
    # Encrypt B^i * s for all i
    cts = [
        encrypt_asymmetric(scaled_key, target_key, params, random_source)
        for scaled_key in _scaled_source_keys(source_key, ks_params, ntt_params)
    ]
    return KeySwitchingKey(data=cts, params=ks_params)
