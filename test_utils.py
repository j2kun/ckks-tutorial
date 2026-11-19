"""Shared helpers for tests."""

import functools
import math

import hypothesis.strategies as st

from params import EncodingParams, NTTParams
from polynomial import IntPolynomial, RNSPolynomial
from polynomial import make_rns_polynomial
from primes import generate_ntt_primes
from typing import Tuple


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


@functools.lru_cache
def _prime_pool(bit_width: int, poly_modulus_degree: int, count: int) -> list[int]:
    return generate_ntt_primes(
        bit_width=bit_width, poly_modulus_degree=poly_modulus_degree, count=count
    )


@st.composite
def rns_encoding_and_ntt_params(
    draw: st.DrawFn,
    poly_modulus_degree: int = 8,
    min_limbs: int = 1,
    max_limbs: int = 4,
    min_scale: float = 2**20,
    max_scale: float = 2**25,
    bit_width: int = 60,
) -> Tuple[EncodingParams, list[NTTParams]]:
    """Draw multi-limb encoding params and matching per-limb NTT params."""
    pool = _prime_pool(bit_width, poly_modulus_degree, 2 * max_limbs)
    moduli = draw(
        st.lists(
            st.sampled_from(pool),
            min_size=min_limbs,
            max_size=max_limbs,
            unique=True,
        )
    )
    scale = draw(st.floats(min_value=min_scale, max_value=max_scale))
    params = EncodingParams(
        scale=scale, poly_modulus_degree=poly_modulus_degree, moduli=moduli
    )
    ntt_params = [NTTParams(degree=poly_modulus_degree, modulus=q) for q in moduli]
    return params, ntt_params


@st.composite
def rns_params_and_plaintext(
    draw: st.DrawFn, **params_kwargs
) -> Tuple[EncodingParams, list[NTTParams], RNSPolynomial]:
    """Draw multi-limb params and an RNS plaintext with coefficients in [-Q/2, Q/2]."""
    params, ntt_params = draw(rns_encoding_and_ntt_params(**params_kwargs))
    Q = math.prod(params.moduli)
    coeffs = draw(
        st.lists(
            st.integers(min_value=-(Q // 2), max_value=Q // 2),
            min_size=params.poly_modulus_degree,
            max_size=params.poly_modulus_degree,
        )
    )
    plaintext = IntPolynomial(coeffs, params.poly_modulus_degree, dtype=object).to_rns(
        ntt_params=ntt_params
    )
    return params, ntt_params, plaintext


@st.composite
def rns_polynomial(
    draw: st.DrawFn, num_limbs: int = 3, poly_modulus_degree: int = 16
) -> RNSPolynomial:
    """Draw a random RNS polynomial."""

    def coeff_picker(deg, modulus):
        return draw(st.integers(min_value=0, max_value=modulus - 1))

    return make_rns_polynomial(
        num_limbs=num_limbs,
        poly_modulus_degree=poly_modulus_degree,
        coeff_picker=coeff_picker,
        bit_width=60,
    )
