from ckks_types import PrivateKey
from params import EncryptionParams
import rng


def generate_symmetric_private_key(
    params: EncryptionParams,
    random_source: rng.RandomSource,
) -> PrivateKey:
    return PrivateKey(
        random_source.gen_ternary_poly(params.degree),
        modulus_degree=params.degree,
    )
