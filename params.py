"""Data classes representing parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EncodingParams:
    scale: float
    poly_modulus_degree: int
    coefficient_modulus: int
