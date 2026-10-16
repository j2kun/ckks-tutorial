from ckks_types import KeySwitchingKey, PrivateKey, PublicKey
from decomposition import TOTAL_BIT_LENGTH
from encryption import encrypt_symmetric, encrypt_asymmetric
from params import EncryptionParams, NTTParams, KeySwitchingKeyParams
from polynomial import IntPolynomial, ModQPolynomial
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
    random_source: rng.RandomSource,
) -> Tuple[PrivateKey, PublicKey]:
    sk = generate_symmetric_private_key(params, random_source)
    ntt_params = NTTParams(degree=params.degree, modulus=params.modulus)

    a = ModQPolynomial(
        random_source.gen_uniform_poly(params.degree, params.modulus),
        modulus_degree=params.degree,
        coefficient_modulus=params.modulus,
        ntt_params=ntt_params,
    )
    e = ModQPolynomial(
        random_source.gen_gaussian_poly(params.degree, params.modulus),
        modulus_degree=params.degree,
        coefficient_modulus=params.modulus,
        ntt_params=ntt_params,
    )

    b = (-(a * sk) + e) % params.modulus
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
    # Scale before reducing s mod Q: s is ternary, so (B^i mod Q) * s has
    # coefficients in {-(B^i mod Q), 0, B^i mod Q}, which fit in an int64,
    # whereas scaling the reduced coefficients (up to Q-1) would overflow.
    return [
        IntPolynomial(
            pow(B, i, ntt_params.modulus) * source_key.coefficients,
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
