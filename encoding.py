from dataclasses import dataclass
import numpy as np

from polynomial import Polynomial
from polynomial import fft_special
from polynomial import fft_special_inv

# A cleartext is a vector of possibly complex values
Cleartext = np.ndarray

# A plaintext is a polynomial (in coefficient form) in a ring
# Z/QZ[x] / (x^N + 1) for some N (power of two) and Q.
Plaintext = Polynomial


@dataclass(frozen=True)
class EncodingParams:
    scale: float
    poly_modulus_degree: int
    ciphertext_modulus: int | None = None


def _llround(x: np.ndarray) -> np.ndarray:
    """Round half away from zero, matching C++ std::llround."""
    return np.copysign(np.floor(np.abs(x) + 0.5), x)


def encode(message: Cleartext, params: EncodingParams) -> Plaintext:
    """Encode a vector of complex numbers into a plaintext polynomial.

    Uses the OpenFHE-compatible encoding pipeline:
    1. Pad message to N/2 slots.
    2. Inverse special FFT (implicitly handles Hermitian extension,
       uses 5^i mod M root ordering for slot-rotation compatibility).
    3. Scale and round (half away from zero, matching std::llround).
    4. Optionally reduce modulo ciphertext_modulus Q.
    """
    message = np.array(message, dtype=complex)
    N = params.poly_modulus_degree
    num_slots = N // 2

    if message.shape[0] > num_slots:
        raise ValueError(
            "Message length exceeds maximum for given poly_modulus_degree. "
            f"Got message length {len(message)} with "
            f"poly_modulus_degree {N}."
        )

    # Pad with zeros up to N/2 slots
    if message.shape[0] < num_slots:
        message = np.concatenate(
            [message, np.zeros(num_slots - message.shape[0], dtype=complex)]
        )

    # Inverse special FFT: N/2 complex slots → N/2 complex packed coefficients.
    # No explicit Hermitian extension needed — the special FFT implicitly
    # exploits the conjugate symmetry to produce real polynomial coefficients.
    packed = fft_special_inv(message, N)

    # Unpack to N real coefficients:
    #   coeffs[0:N/2]  = Re(packed)  (low-degree terms)
    #   coeffs[N/2:N]  = Im(packed)  (high-degree terms)
    coeffs = np.concatenate([np.real(packed), np.imag(packed)])

    # Scale and round (half away from zero, matching C++ std::llround)
    scaled = coeffs * params.scale
    rounded = _llround(scaled)

    # Optionally reduce modulo ciphertext modulus Q
    if params.ciphertext_modulus is not None:
        Q = params.ciphertext_modulus
        rounded = np.mod(rounded, Q)

    return Polynomial(rounded, N)


def decode(plaintext: Plaintext, params: EncodingParams) -> Cleartext:
    """Decode a CKKS plaintext into a vector of complex numbers.

    Uses the OpenFHE-compatible decoding pipeline:
    1. Center-lift coefficients modulo Q (if modular reduction was used).
    2. Remove scale factor.
    3. Pack N real coefficients into N/2 complex values.
    4. Forward special FFT → N/2 complex slot values.
    """
    N = plaintext.modulus_degree
    num_slots = N // 2
    coeffs = plaintext.coefficients.copy().astype(float)

    # Center-lift: map from [0, Q) to (-Q/2, Q/2]
    if params.ciphertext_modulus is not None:
        Q = params.ciphertext_modulus
        coeffs = np.where(coeffs > Q / 2, coeffs - Q, coeffs)

    # Remove scale
    coeffs = coeffs / params.scale

    # Pack into N/2 complex values: packed[i] = coeffs[i] + j * coeffs[i + N/2]
    packed = coeffs[:num_slots] + 1j * coeffs[num_slots:]

    # Forward special FFT: N/2 complex packed → N/2 complex slots
    return fft_special(packed, N)
