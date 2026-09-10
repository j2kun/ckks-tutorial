from dataclasses import dataclass
import numpy as np

from polynomial import canonical_embedding
from polynomial import inverse_canonical_embedding
from ckks_types import Cleartext, Plaintext


@dataclass(frozen=True)
class EncodingParams:
    scale: float
    poly_modulus_degree: int
    coefficient_modulus: int


def encode(message: Cleartext, params: EncodingParams) -> Plaintext:
    """Encode a vector of complex numbers into a plaintext polynomial.

    First converts the message to its Hermitian form, then computes
    round(params.scale * sigma^{-1}(message)), where sigma^{-1} is the inverse
    canonical embedding.
    """
    message = np.array(message)
    if message.shape[0] > params.poly_modulus_degree / 2:
        raise ValueError(
            "Message length exceeds maximum for given poly_modulus_degree. "
            f"Got message length {len(message)} with "
            f"poly_modulus_degree {params.poly_modulus_degree}."
        )

    # Pad with zeros up to N / 2
    num_zeros = params.poly_modulus_degree // 2 - message.shape[0]
    if num_zeros:
        message = np.concatenate([message, np.zeros(num_zeros, dtype=message.dtype)])

    # Flip and concat with conjugate to make Hermitian
    hermitian_msg = np.concatenate([message, np.flip(np.conjugate(message))])

    # result of inverse canonical_embedding is guaranteed to be real-valued.
    polynomial = inverse_canonical_embedding(hermitian_msg)
    rounded_scaled_coeffs = np.round(np.real(polynomial) * params.scale)
    return Plaintext(
        rounded_scaled_coeffs, params.poly_modulus_degree, params.coefficient_modulus
    )


def decode(plaintext: Plaintext, params: EncodingParams) -> Cleartext:
    """Decode a CKKS plaintext into a vector of complex numbers.

    Computes sigma(message / params.scale), where sigma is the canonical
    embedding, then extracts the first N/2 entries.
    """
    centered_plaintext = plaintext.lift_to_signed_representative()
    scale_removed_coeffs = centered_plaintext.coefficients / params.scale
    unembedded = canonical_embedding(scale_removed_coeffs)
    return unembedded[: params.poly_modulus_degree // 2]
