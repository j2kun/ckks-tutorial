from ckks_types import Ciphertext, Deg2Ciphertext


def mul(ct1: Ciphertext, ct2: Ciphertext) -> Deg2Ciphertext:
    c0, c1 = ct1.data
    d0, d1 = ct2.data

    t0 = c0 * d0
    t1 = c0 * d1 + c1 * d0
    t2 = c1 * d1
    return Deg2Ciphertext(data=(t0, t1, t2))
