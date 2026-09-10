"""A 'context' class wrapping various parameters and precomputed values."""

from ckks_types import Cleartext, Plaintext
from encoding import encode, decode
from params import EncodingParams, NTTParams


class CKKSContext:

    def __init__(self, params: EncodingParams):
        self.params = params
        self.nttParams = NTTParams(
            degree=params.poly_modulus_degree, modulus=params.coefficient_modulus
        )

    def encode(self, message: Cleartext) -> Plaintext:
        return encode(self, message, self.params)

    def decode(self, encoded: Plaintext) -> Cleartext:
        return decode(self, encoded, self.params)

    def encrypt_symmetric(self):
        ...

    def decrypt_symmetric(self):
        ...

    def encrypt_asymmetric(self):
        ...

    def decrypt_asymmetric(self):
        ...

    def generate_symmetric_private_key(self):
        ...

    def generate_asymmetric_keypair(self):
        ...
