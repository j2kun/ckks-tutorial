"""A 'context' class wrapping various parameters and precomputed values."""

from typing import Tuple

from ckks_types import Cleartext, Plaintext, Ciphertext, PublicKey, PrivateKey
from encoding import encode, decode
from encryption import (
    encrypt_symmetric,
    decrypt_symmetric,
    encrypt_asymmetric,
    decrypt_asymmetric,
)
from key_gen import generate_symmetric_private_key, generate_asymmetric_keypair
from params import EncodingParams, EncryptionParams, NTTParams
from rng import SecureRandomSource


class CKKSContext:

    def __init__(self, params: EncodingParams):
        self.encoding_params = params
        self.ntt_params = NTTParams(
            degree=params.poly_modulus_degree,
            modulus=params.coefficient_modulus,
        )
        self.encryption_params = EncryptionParams(
            degree=params.poly_modulus_degree,
            modulus=params.coefficient_modulus,
        )
        # TODO: add proper selection of sigma parameter
        self.rng = SecureRandomSource()

    def encode(self, message: Cleartext) -> Plaintext:
        return encode(message, self.encoding_params, self.ntt_params)

    def decode(self, encoded: Plaintext) -> Cleartext:
        return decode(encoded, self.encoding_params)

    def encrypt_symmetric(
        self, plaintext: Plaintext, private_key: PrivateKey
    ) -> Ciphertext:
        return encrypt_symmetric(
            plaintext, private_key, self.encryption_params, self.rng
        )

    def decrypt_symmetric(
        self, ciphertext: Ciphertext, private_key: PrivateKey
    ) -> Plaintext:
        return decrypt_symmetric(ciphertext, private_key, self.encryption_params)

    def encrypt_asymmetric(
        self, plaintext: Plaintext, public_key: PublicKey
    ) -> Ciphertext:
        return encrypt_asymmetric(
            plaintext, public_key, self.encryption_params, self.rng
        )

    def decrypt_asymmetric(
        self, ciphertext: Ciphertext, private_key: PrivateKey
    ) -> Plaintext:
        return decrypt_asymmetric(ciphertext, private_key, self.encryption_params)

    def generate_symmetric_private_key(self) -> PrivateKey:
        return generate_symmetric_private_key(self.encryption_params, self.rng)

    def generate_asymmetric_keypair(self) -> Tuple[PrivateKey, PublicKey]:
        return generate_asymmetric_keypair(self.encryption_params, self.rng)
