"""Tests for ciphertext multiplication."""

from hypothesis import strategies as st
import hypothesis
import numpy as np

from context import CKKSContext
from encryption import decrypt_deg2
from encoding import decode
from ntt import NTT_64_BIT_PRIME
from params import EncodingParams
from rng import SeededRandomSource

DEFAULT_ENCODING_PARAMS = EncodingParams(
    scale=2**20,
    poly_modulus_degree=8,
    coefficient_modulus=NTT_64_BIT_PRIME,
)

# Without relinearizing the ciphertext yet, we need a separate encoding params
# for the product two scale-D ciphertexts, which has a scale of D**2
PRODUCT_ENCODING_PARAMS = EncodingParams(
    scale=2**40,
    poly_modulus_degree=DEFAULT_ENCODING_PARAMS.poly_modulus_degree,
    coefficient_modulus=DEFAULT_ENCODING_PARAMS.coefficient_modulus,
)


def test_mul_no_relin_two_symmetric():
    seed = 837465
    message1 = np.array([1, 2, 3, 4], dtype=np.complex64)
    message2 = np.array([2, 3, 1, 2], dtype=np.complex64)
    expected = np.array([2, 6, 3, 8], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    sk = context.generate_symmetric_private_key()
    pt1 = context.encode(message1)
    ct1 = context.encrypt_symmetric(pt1, sk)
    pt2 = context.encode(message2)
    ct2 = context.encrypt_symmetric(pt2, sk)

    ct_product = context.mul_no_relin(ct1, ct2)

    decrypted_product = decrypt_deg2(ct_product, sk)
    decoded_product = decode(decrypted_product, PRODUCT_ENCODING_PARAMS)

    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)


def test_mul_no_relin_two_asymmetric():
    seed = 837465
    message1 = np.array([1, 2, 3, 4], dtype=np.complex64)
    message2 = np.array([2, 3, 1, 2], dtype=np.complex64)
    expected = np.array([2, 6, 3, 8], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    sk, pk = context.generate_asymmetric_keypair()
    pt1 = context.encode(message1)
    ct1 = context.encrypt_asymmetric(pt1, pk)
    pt2 = context.encode(message2)
    ct2 = context.encrypt_asymmetric(pt2, pk)

    ct_product = context.mul_no_relin(ct1, ct2)

    decrypted_product = decrypt_deg2(ct_product, sk)
    decoded_product = decode(decrypted_product, PRODUCT_ENCODING_PARAMS)

    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)


def test_mul_no_relin_plain_symmetric():
    seed = 837465
    message1 = np.array([1, 2, 3, 4], dtype=np.complex64)
    message2 = np.array([2, 3, 1, 2], dtype=np.complex64)
    expected = np.array([2, 6, 3, 8], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    sk = context.generate_symmetric_private_key()
    pt1 = context.encode(message1)
    ct1 = context.encrypt_symmetric(pt1, sk)
    pt2 = context.encode(message2)

    ct_product = context.mul_plain(ct1, pt2)
    decrypted_product = context.decrypt_symmetric(ct_product, sk)
    decoded_product = decode(decrypted_product, PRODUCT_ENCODING_PARAMS)
    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)

    ct_product = context.mul_plain(pt2, ct1)
    decrypted_product = context.decrypt_symmetric(ct_product, sk)
    decoded_product = decode(decrypted_product, PRODUCT_ENCODING_PARAMS)
    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)


def test_mul_no_relin_plain_asymmetric():
    seed = 837465
    message1 = np.array([1, 2, 3, 4], dtype=np.complex64)
    message2 = np.array([2, 3, 1, 2], dtype=np.complex64)
    expected = np.array([2, 6, 3, 8], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    sk, pk = context.generate_asymmetric_keypair()
    pt1 = context.encode(message1)
    ct1 = context.encrypt_asymmetric(pt1, pk)
    pt2 = context.encode(message2)

    ct_product = context.mul_plain(ct1, pt2)
    decrypted_product = context.decrypt_asymmetric(ct_product, sk)
    decoded_product = decode(decrypted_product, PRODUCT_ENCODING_PARAMS)
    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)

    ct_product = context.mul_plain(pt2, ct1)
    decrypted_product = context.decrypt_asymmetric(ct_product, sk)
    decoded_product = decode(decrypted_product, PRODUCT_ENCODING_PARAMS)
    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)


@hypothesis.given(
    st.lists(
        st.complex_numbers(min_magnitude=1, max_magnitude=10),
        min_size=4,
        max_size=4,
    ),
    st.lists(
        st.complex_numbers(min_magnitude=1, max_magnitude=10),
        min_size=4,
        max_size=4,
    ),
    st.floats(min_value=2**20, max_value=2**25),
    st.integers(min_value=0, max_value=10),
)
def test_mul_no_relin_symmetric_hypothesis(message1, message2, scale, seed):
    encoding_params = EncodingParams(
        scale=scale,
        poly_modulus_degree=8,
        coefficient_modulus=NTT_64_BIT_PRIME,
    )
    decoding_params = EncodingParams(
        scale=scale**2,
        poly_modulus_degree=8,
        coefficient_modulus=NTT_64_BIT_PRIME,
    )

    message1 = np.array(message1, dtype=np.complex64)
    message2 = np.array(message2, dtype=np.complex64)
    expected = message1 * message2
    context = CKKSContext(encoding_params)
    context.rng = SeededRandomSource(seed)

    sk = context.generate_symmetric_private_key()
    pt1 = context.encode(message1)
    ct1 = context.encrypt_symmetric(pt1, sk)
    pt2 = context.encode(message2)
    ct2 = context.encrypt_symmetric(pt2, sk)

    ct_product = context.mul_no_relin(ct1, ct2)

    decrypted_product = decrypt_deg2(ct_product, sk)
    decoded_product = decode(decrypted_product, decoding_params)

    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)


@hypothesis.given(
    st.lists(
        st.complex_numbers(min_magnitude=1, max_magnitude=10),
        min_size=4,
        max_size=4,
    ),
    st.lists(
        st.complex_numbers(min_magnitude=1, max_magnitude=10),
        min_size=4,
        max_size=4,
    ),
    st.floats(min_value=2**20, max_value=2**25),
    st.integers(min_value=0, max_value=10),
)
def test_mul_no_relin_asymmetric_hypothesis(message1, message2, scale, seed):
    encoding_params = EncodingParams(
        scale=scale,
        poly_modulus_degree=8,
        coefficient_modulus=NTT_64_BIT_PRIME,
    )
    decoding_params = EncodingParams(
        scale=scale**2,
        poly_modulus_degree=8,
        coefficient_modulus=NTT_64_BIT_PRIME,
    )

    message1 = np.array(message1, dtype=np.complex64)
    message2 = np.array(message2, dtype=np.complex64)
    expected = message1 * message2
    context = CKKSContext(encoding_params)
    context.rng = SeededRandomSource(seed)

    sk, pk = context.generate_asymmetric_keypair()
    pt1 = context.encode(message1)
    ct1 = context.encrypt_asymmetric(pt1, pk)
    pt2 = context.encode(message2)
    ct2 = context.encrypt_asymmetric(pt2, pk)

    ct_product = context.mul_no_relin(ct1, ct2)

    decrypted_product = decrypt_deg2(ct_product, sk)
    decoded_product = decode(decrypted_product, decoding_params)

    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)


def test_mul_no_relin_error_growth_formula_symmetric():
    """Test that the error of a product of ciphertexts encrypting plaintexts
    pt_c and pt_d with errors e_c and e_d matches the formula:

        pt_c * e_d + e_c * pt_d + e_c * e_d
    """
    seed = 384765
    message_c = np.array([1, 2, 3, 4], dtype=np.complex64)
    message_d = np.array([2, 3, 1, 2], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    sk = context.generate_symmetric_private_key()

    pt_c = context.encode(message_c)
    ct_c = context.encrypt_symmetric(pt_c, sk)
    e_c = context.decrypt_symmetric(ct_c, sk) - pt_c

    pt_d = context.encode(message_d)
    ct_d = context.encrypt_symmetric(pt_d, sk)
    e_d = context.decrypt_symmetric(ct_d, sk) - pt_d

    ct_product = context.mul_no_relin(ct_c, ct_d)

    pt_product = pt_c * pt_d
    e_product = decrypt_deg2(ct_product, sk) - pt_product

    expected_error = pt_c * e_d + e_c * pt_d + e_c * e_d
    np.testing.assert_allclose(
        e_product.coefficients, expected_error.coefficients, rtol=0, atol=1e-10
    )


def test_mul_no_relin_error_growth_formula_asymmetric():
    """Test error formula for asymmetric mul."""
    seed = 384765
    message_c = np.array([1, 2, 3, 4], dtype=np.complex64)
    message_d = np.array([2, 3, 1, 2], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    sk, pk = context.generate_asymmetric_keypair()

    pt_c = context.encode(message_c)
    ct_c = context.encrypt_asymmetric(pt_c, pk)
    e_c = context.decrypt_symmetric(ct_c, sk) - pt_c

    pt_d = context.encode(message_d)
    ct_d = context.encrypt_asymmetric(pt_d, pk)
    e_d = context.decrypt_symmetric(ct_d, sk) - pt_d

    ct_product = context.mul_no_relin(ct_c, ct_d)

    pt_product = pt_c * pt_d
    e_product = decrypt_deg2(ct_product, sk) - pt_product

    expected_error = pt_c * e_d + e_c * pt_d + e_c * e_d
    np.testing.assert_allclose(
        e_product.coefficients, expected_error.coefficients, rtol=0, atol=1e-10
    )


def test_mul_two_symmetric():
    seed = 837465
    message1 = np.array([1, 2, 3, 4], dtype=np.complex64)
    message2 = np.array([2, 3, 1, 2], dtype=np.complex64)
    expected = np.array([2, 6, 3, 8], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    sk = context.generate_symmetric_private_key()
    context.initialize_relinearization_key(sk)
    pt1 = context.encode(message1)
    ct1 = context.encrypt_symmetric(pt1, sk)
    pt2 = context.encode(message2)
    ct2 = context.encrypt_symmetric(pt2, sk)

    ct_product = context.mul(ct1, ct2)

    decrypted_product = context.decrypt_symmetric(ct_product, sk)
    decoded_product = decode(decrypted_product, PRODUCT_ENCODING_PARAMS)

    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)


def test_mul_two_asymmetric():
    seed = 837465
    message1 = np.array([1, 2, 3, 4], dtype=np.complex64)
    message2 = np.array([2, 3, 1, 2], dtype=np.complex64)
    expected = np.array([2, 6, 3, 8], dtype=np.complex64)
    context = CKKSContext(DEFAULT_ENCODING_PARAMS)
    context.rng = SeededRandomSource(seed)

    sk, pk = context.generate_asymmetric_keypair()
    context.initialize_relinearization_key(sk)
    pt1 = context.encode(message1)
    ct1 = context.encrypt_asymmetric(pt1, pk)
    pt2 = context.encode(message2)
    ct2 = context.encrypt_asymmetric(pt2, pk)

    ct_product = context.mul(ct1, ct2)

    decrypted_product = context.decrypt_asymmetric(ct_product, sk)
    decoded_product = decode(decrypted_product, PRODUCT_ENCODING_PARAMS)

    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)


@hypothesis.given(
    st.lists(
        st.complex_numbers(min_magnitude=1, max_magnitude=10),
        min_size=4,
        max_size=4,
    ),
    st.lists(
        st.complex_numbers(min_magnitude=1, max_magnitude=10),
        min_size=4,
        max_size=4,
    ),
    st.floats(min_value=2**20, max_value=2**25),
    st.integers(min_value=0, max_value=10),
)
def test_mul_symmetric_hypothesis(message1, message2, scale, seed):
    encoding_params = EncodingParams(
        scale=scale,
        poly_modulus_degree=8,
        coefficient_modulus=NTT_64_BIT_PRIME,
    )
    decoding_params = EncodingParams(
        scale=scale**2,
        poly_modulus_degree=8,
        coefficient_modulus=NTT_64_BIT_PRIME,
    )

    message1 = np.array(message1, dtype=np.complex64)
    message2 = np.array(message2, dtype=np.complex64)
    expected = message1 * message2
    context = CKKSContext(encoding_params)
    context.rng = SeededRandomSource(seed)

    sk = context.generate_symmetric_private_key()
    context.initialize_relinearization_key(sk)
    pt1 = context.encode(message1)
    ct1 = context.encrypt_symmetric(pt1, sk)
    pt2 = context.encode(message2)
    ct2 = context.encrypt_symmetric(pt2, sk)

    ct_product = context.mul_no_relin(ct1, ct2)

    decrypted_product = decrypt_deg2(ct_product, sk)
    decoded_product = decode(decrypted_product, decoding_params)

    np.testing.assert_allclose(decoded_product, expected, rtol=0, atol=0.2)
