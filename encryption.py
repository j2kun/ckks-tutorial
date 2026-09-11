from ckks_types import Ciphertext, Plaintext, PrivateKey, PublicKey
from params import EncryptionParams
from polynomial import ModQPolynomial, IntPolynomial
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


def encrypt_asymmetric(
    plaintext: Plaintext,
    public_key: PublicKey,
    params: EncryptionParams,
    random_source: rng.RandomSource,
) -> Ciphertext:
    b, a = public_key.data
    u = IntPolynomial(
        random_source.gen_ternary_poly(params.degree),
        modulus_degree=params.degree,
    )
    e0 = ModQPolynomial(
        random_source.gen_gaussian_poly(params.degree, params.modulus),
        modulus_degree=params.degree,
        coefficient_modulus=params.modulus,
        ntt_params=plaintext.ntt_params,
    )
    e1 = ModQPolynomial(
        random_source.gen_gaussian_poly(params.degree, params.modulus),
        modulus_degree=params.degree,
        coefficient_modulus=params.modulus,
        ntt_params=plaintext.ntt_params,
    )

    c0 = (b * u + plaintext + e0) % params.modulus
    c1 = (a * u + e1) % params.modulus

    return Ciphertext(data=(c0, c1))


def decrypt_asymmetric(
    ciphertext: Ciphertext,
    secret_key: PrivateKey,
    params: EncryptionParams,
) -> Plaintext:
    return decrypt_symmetric(ciphertext, secret_key, params)
