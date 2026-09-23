"""Data classes representing parameters."""

from dataclasses import dataclass
from dataclasses import field

import ntt


@dataclass
class NTTParams:
    """NTT parameters and pre-computed twiddle factors."""

    degree: int
    modulus: int
    precomputed: ntt.PrecomputedNTT = field(init=False)

    def __post_init__(self):
        self.precomputed = ntt.precompute_ntt(self.modulus, self.degree)


@dataclass(frozen=True)
class EncodingParams:
    """Parameters needed for plaintext encoding."""

    scale: float
    poly_modulus_degree: int
    coefficient_modulus: int


@dataclass(frozen=True)
class EncryptionParams:
    """Parameters needed for encryption and key generation."""

    degree: int
    modulus: int
