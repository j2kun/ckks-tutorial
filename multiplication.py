from ckks_types import Ciphertext, Deg2Ciphertext, Plaintext, PtOrCt


def mul(ct1: Ciphertext, ct2: Ciphertext) -> Deg2Ciphertext:
    c0, c1 = ct1.data
    d0, d1 = ct2.data

    t0 = c0 * d0
    t1 = c0 * d1 + c1 * d0
    t2 = c1 * d1
    return Deg2Ciphertext(data=(t0, t1, t2))


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
