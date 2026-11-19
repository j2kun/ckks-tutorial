from typing import Iterable

from ckks_types import Ciphertext, Deg2Ciphertext, Plaintext, PrivateKey, PublicKey
from params import EncryptionParams, NTTParams
from polynomial import IntPolynomial, ModQPolynomial, RNSPolynomial
import rng


def sample_uniform_rns_poly(
    random_source: rng.RandomSource, degree: int, ntt_params: Iterable[NTTParams]
) -> RNSPolynomial:
    """Sample a uniform polynomial limbwise."""
    # Note we can't sample the error polynomial limbwise, because the error of
    # each limb relates in a complex way to the error of the reconstructed
    # polynomial.
    samples = []
    for ntt_param in ntt_params:
        modulus = ntt_param.modulus
        samples.append(
            ModQPolynomial(
                random_source.gen_uniform_poly(degree, modulus),
                modulus_degree=degree,
                coefficient_modulus=modulus,
                ntt_params=ntt_param,
            )
        )

    return RNSPolynomial(limbs=samples)


def encrypt_symmetric(
    plaintext: Plaintext,
    private_key: PrivateKey,
    params: EncryptionParams,
    random_source: rng.RandomSource,
) -> Ciphertext:
    sample = sample_uniform_rns_poly(random_source, params.degree, plaintext.ntt_params)
    error = IntPolynomial(
        random_source.gen_gaussian_poly(degree=params.degree),
        modulus_degree=params.degree,
        dtype=object,
    ).to_rns(ntt_params=plaintext.ntt_params)

    bias = -(sample * private_key) + plaintext + error
    return Ciphertext(data=(bias, sample))


def decrypt_symmetric(
    ciphertext: Ciphertext,
    secret_key: PrivateKey,
    params: EncryptionParams,
) -> Plaintext:
    bias, sample = ciphertext.data
    return bias + sample * secret_key


def extract_noise(
    ciphertext: Ciphertext,
    secret_key: PrivateKey,
    plaintext: Plaintext,
) -> IntPolynomial:
    """A debug helper to extract the exact noise polynomial from a ciphertext."""
    bias, sample = ciphertext.data
    return ((bias + sample * secret_key) - plaintext).lift_to_signed_representative()


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
    e0 = IntPolynomial(
        random_source.gen_gaussian_poly(params.degree),
        modulus_degree=params.degree,
        dtype=object,
    ).to_rns(ntt_params=plaintext.ntt_params)
    e1 = IntPolynomial(
        random_source.gen_gaussian_poly(params.degree),
        modulus_degree=params.degree,
        dtype=object,
    ).to_rns(ntt_params=plaintext.ntt_params)

    c0 = b * u + plaintext + e0
    c1 = a * u + e1

    return Ciphertext(data=(c0, c1))


def decrypt_asymmetric(
    ciphertext: Ciphertext,
    secret_key: PrivateKey,
    params: EncryptionParams,
) -> Plaintext:
    return decrypt_symmetric(ciphertext, secret_key, params)


def decrypt_deg2(
    ciphertext: Deg2Ciphertext,
    secret_key: PrivateKey,
) -> Plaintext:
    t0, t1, t2 = ciphertext.data
    return t0 + t1 * secret_key + t2 * (secret_key * secret_key)
