from ckks_types import Ciphertext, Plaintext, PtOrCt


def add(v1: PtOrCt, v2: PtOrCt) -> Ciphertext:
    match v1, v2:
        case Ciphertext((c0, c1)), Ciphertext((d0, d1)):
            return Ciphertext(data=(c0 + d0, c1 + d1))
        case Ciphertext((c0, c1)), Plaintext() as p:
            return Ciphertext(data=(c0 + p, c1))
        case Plaintext() as p, Ciphertext((c0, c1)):
            return Ciphertext(data=(c0 + p, c1))
        case _:
            raise ValueError("Calling add on two plaintexts is weird!")
