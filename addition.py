from ckks_types import Ciphertext


def add(ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
    c0, c1 = ct1.data
    d0, d1 = ct2.data
    return Ciphertext(data=(c0 + d0, c1 + d1))
