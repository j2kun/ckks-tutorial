"""Shared helpers for tests."""

from typing import Tuple
from params import EncodingParams, NTTParams


def single_limb_encoding_and_ntt_params(
    scale: float, poly_modulus_degree: int, modulus: int
) -> Tuple[EncodingParams, list[NTTParams]]:
    params = EncodingParams(
        scale=scale,
        poly_modulus_degree=poly_modulus_degree,
        moduli=[modulus],
    )
    ntt_params = [
        NTTParams(degree=params.poly_modulus_degree, modulus=params.moduli[0])
    ]
    return params, ntt_params
