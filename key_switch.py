from ckks_types import Ciphertext, KeySwitchingKey


def key_switch(ct: Ciphertext, ksk: KeySwitchingKey) -> Ciphertext:
    """Switch `ct` to encrypt the same plaintext under a new key."""
    # TODO: enhance naive key-switching to reduce error growth
    c0, c1 = ct.data
    b, a = ksk.data.data
    return Ciphertext(data=(c0 + c1 * b, c1 * a))
