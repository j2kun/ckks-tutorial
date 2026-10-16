from ckks_types import KeySwitchingKey, PrivateKey, PublicKey
from encryption import encrypt_symmetric, encrypt_asymmetric
from params import EncryptionParams, NTTParams
from polynomial import ModQPolynomial
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


def generate_symmetric_key_switching_key(
    source_key: PrivateKey,
    target_key: PrivateKey,
    params: EncryptionParams,
    ntt_params: NTTParams,
    random_source: rng.RandomSource,
) -> KeySwitchingKey:
    """Generate a key switching key from source_key to target_key.

    This key can be used by key_switch to convert a ciphertext
    encrypted under source_key to be encrypted under target_key.
    """
    ct = encrypt_symmetric(
        source_key.to_mod_q(ntt_params), target_key, params, random_source
    )
    return KeySwitchingKey(data=ct)


def generate_asymmetric_key_switching_key(
    source_key: PrivateKey,
    target_key: PublicKey,
    params: EncryptionParams,
    ntt_params: NTTParams,
    random_source: rng.RandomSource,
) -> KeySwitchingKey:
    """Generate a key switching key from source_key to target_key.

    This key can be used by key_switch to convert a ciphertext
    encrypted under source_key to be encrypted under target_key.
    """
    ct = encrypt_asymmetric(
        source_key.to_mod_q(ntt_params), target_key, params, random_source
    )
    return KeySwitchingKey(data=ct)
