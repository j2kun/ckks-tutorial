from ckks_types import Ciphertext, KeySwitchingKey
from decomposition import decompose_poly
from polynomial import ModQPolynomial


def key_switch(ct: Ciphertext, ksk: KeySwitchingKey) -> Ciphertext:
    """Switch `ct` to encrypt the same plaintext under a new key."""
    c0, c1 = ct.data
    ks_cts = ksk.data

    # Digit-decompose c1 with the same decomposition used to
    # generate the key switching key.
    c1_digits = decompose_poly(c1, ksk.params.log_base)

    # Sum the plaintext-ciphertext products d_i * (b_i, a_i),
    # with an initializer of (c0, 0).
    switched_c0 = c0
    switched_c1 = ModQPolynomial.zeros_like(c1)
    for c1_digit, ks_ct in zip(c1_digits, ks_cts):
        ks_b, ks_a = ks_ct.data
        switched_c0 += c1_digit * ks_b
        switched_c1 += c1_digit * ks_a

    return Ciphertext(data=(switched_c0, switched_c1))
