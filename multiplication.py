from ckks_types import Ciphertext, Deg2Ciphertext, Plaintext, PtOrCt, KeySwitchingKey
from key_switch import key_switch
from polynomial import ModQPolynomial


def mul(ct1: Ciphertext, ct2: Ciphertext) -> Deg2Ciphertext:
    c0, c1 = ct1.data
    d0, d1 = ct2.data

    t0 = c0 * d0
    t1 = c0 * d1 + c1 * d0
    t2 = c1 * d1
    return Deg2Ciphertext(data=(t0, t1, t2))


def relinearize(ct: Deg2Ciphertext, ksk: KeySwitchingKey) -> Ciphertext:
    """Convert a degree-2 ciphertext to a degree-1 ciphertext."""
    t0, t1, t2 = ct.data
    ks_input = Ciphertext(data=(ModQPolynomial.zeros_like(t2), t2))
    k0, k1 = key_switch(ks_input, ksk).data
    return Ciphertext(data=(t0 + k0, t1 + k1))


def mul_plain(v1: PtOrCt, v2: PtOrCt) -> Ciphertext:
    match v1, v2:
        case Ciphertext((c0, c1)), Plaintext() as p:
            pass
        case Plaintext() as p, Ciphertext((c0, c1)):
            pass
        case _:
            raise ValueError(
                "mul_plain must take exactly one ciphertext and one plaintext as input."
            )

    return Ciphertext(data=(p * c0, p * c1))
