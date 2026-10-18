"""A 'context' class wrapping various parameters and precomputed values."""

from typing import Tuple

from ckks_types import (
    Cleartext,
    Plaintext,
    Ciphertext,
    PtOrCt,
    Deg2Ciphertext,
    PublicKey,
    PrivateKey,
    KeySwitchingKey,
)
from encoding import encode, decode
from encryption import (
    encrypt_symmetric,
    decrypt_symmetric,
    encrypt_asymmetric,
    decrypt_asymmetric,
)
import key_gen
from params import EncodingParams, EncryptionParams, KeySwitchingKeyParams, NTTParams
from rng import SecureRandomSource
from addition import add
from multiplication import mul, mul_plain, relinearize


class CKKSContext:

    def __init__(
        self,
        params: EncodingParams,
        ks_params: KeySwitchingKeyParams = KeySwitchingKeyParams(log_base=4),
    ):
        self.encoding_params = params
        self.ks_params = ks_params
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
        return key_gen.generate_symmetric_private_key(self.encryption_params, self.rng)

    def generate_asymmetric_keypair(self) -> Tuple[PrivateKey, PublicKey]:
        return key_gen.generate_asymmetric_keypair(self.encryption_params, self.rng)

    def initialize_relinearization_key(self, sk: PrivateKey):
        self.relinearization_key = self.generate_symmetric_key_switching_key(
            sk * sk, sk
        )

    def generate_symmetric_key_switching_key(
        self, source_key: PrivateKey, target_key: PrivateKey
    ) -> KeySwitchingKey:
        return key_gen.generate_symmetric_key_switching_key(
            source_key,
            target_key,
            self.encryption_params,
            self.ks_params,
            self.ntt_params,
            self.rng,
        )

    def generate_asymmetric_key_switching_key(
        self,
        source_key: PrivateKey,
        target_key: PublicKey,
    ) -> KeySwitchingKey:
        return key_gen.generate_asymmetric_key_switching_key(
            source_key,
            target_key,
            self.encryption_params,
            self.ks_params,
            self.ntt_params,
            self.rng,
        )

    def add(self, v1: PtOrCt, v2: PtOrCt) -> Ciphertext:
        return add(v1, v2)

    def mul_no_relin(self, ct1: Ciphertext, ct2: Ciphertext) -> Deg2Ciphertext:
        return mul(ct1, ct2)

    def mul(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        if not hasattr(self, "relinearization_key"):
            raise ValueError("Must generate relinearization key first")
        unrelinearized = mul(ct1, ct2)
        return relinearize(unrelinearized, self.relinearization_key)

    def mul_plain(self, v1: PtOrCt, v2: PtOrCt) -> Ciphertext:
        return mul_plain(v1, v2)
