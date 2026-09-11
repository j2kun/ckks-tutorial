from ckks_types import Ciphertext, Plaintext, PrivateKey
from params import EncryptionParams
from polynomial import ModQPolynomial
import rng


def encrypt_symmetric(
    plaintext: Plaintext,
    private_key: PrivateKey,
    params: EncryptionParams,
    random_source: rng.RandomSource,
) -> Ciphertext:
    sample = ModQPolynomial(
        random_source.gen_uniform_poly(degree=params.degree, modulus=params.modulus),
        modulus_degree=params.degree,
        coefficient_modulus=params.modulus,
        ntt_params=plaintext.ntt_params,
    )
    error = ModQPolynomial(
        random_source.gen_gaussian_poly(degree=params.degree, modulus=params.modulus),
        modulus_degree=params.degree,
        coefficient_modulus=params.modulus,
        ntt_params=plaintext.ntt_params,
    )

    bias = -(sample * private_key) + plaintext + error
    return Ciphertext(data=(bias, sample))


def decrypt_symmetric(
    ciphertext: Ciphertext,
    secret_key: PrivateKey,
    params: EncryptionParams,
) -> Plaintext:
    bias, sample = ciphertext.data
    return bias + sample * secret_key
