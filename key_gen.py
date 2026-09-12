from ckks_types import PrivateKey, PublicKey
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
