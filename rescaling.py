import numpy as np

from ckks_types import Ciphertext
from polynomial import ModQPolynomial
from params import NTTParams


def rescale_poly(
    poly: ModQPolynomial, new_modulus: int, new_ntt_params: NTTParams
) -> ModQPolynomial:
    rescale_factor = np.float64(poly.coefficient_modulus) / np.float64(new_modulus)
    new_coeffs = np.round(poly.coefficients / rescale_factor)
    return ModQPolynomial(
        coefficients=new_coeffs,
        coefficient_modulus=new_modulus,
        modulus_degree=poly.modulus_degree,
        ntt_params=new_ntt_params,
    )


def rescale(ct: Ciphertext, new_modulus: int, new_ntt_params: NTTParams) -> Ciphertext:
    d0 = rescale_poly(ct.data[0], new_modulus, new_ntt_params)
    d1 = rescale_poly(ct.data[1], new_modulus, new_ntt_params)
    return Ciphertext(data=(d0, d1))
