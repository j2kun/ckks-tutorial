from typing import Callable, Iterable, Sequence, TypeVar
import math

from numpy.fft import fft, ifft
import numpy as np

from ntt import negacyclic_polymul_ntt
from params import NTTParams
import rns

# A callable (degree, modulus) -> coefficient
# to help generating RNS polynomials
CoeffPicker = Callable[[int, int], int]

T = TypeVar("T")


def primitive_nth_root(n):
    """Return a primitive nth root of unity."""
    return math.cos(2 * math.pi / n) + 1.0j * math.sin(2 * math.pi / n)


def is_power_of_two(x: int) -> bool:
    return (x != 0) and (x & (x - 1) == 0)


def negacyclic_polymul_complex_twist(p1, p2):
    """Computes a poly multiplication mod (X^N + 1) where N = len(p1).

    Uses the idea on page 332 (pdf page 8) of "Fast multiplication and its
    applications" by Daniel Bernstein.
    http://cr.yp.to/lineartime/multapps-20080515.pdf
    """
    n = p2.shape[0]
    primitive_root = primitive_nth_root(2 * n)
    root_powers = primitive_root ** np.arange(n // 2)

    p1_preprocessed = (p1[: n // 2] + 1j * p1[n // 2 :]) * root_powers
    p2_preprocessed = (p2[: n // 2] + 1j * p2[n // 2 :]) * root_powers

    p1_ft = fft(p1_preprocessed)
    p2_ft = fft(p2_preprocessed)
    prod = p1_ft * p2_ft
    ifft_prod = ifft(prod)
    ifft_rotated = ifft_prod * primitive_root ** np.arange(0, -n // 2, -1)

    return np.round(
        np.concatenate([np.real(ifft_rotated), np.imag(ifft_rotated)])
    ).astype(p1.dtype)


class IntPolynomial:
    """An integer polynomial with a ring modulus x^N + 1."""

    def __init__(
        self,
        coefficients: np.ndarray | Iterable[float],
        modulus_degree: int,
        dtype=np.int64,
    ):
        """Construct a polynomial.

        Args:
            coefficients: the coefficients of the polynomial.
            modulus_degree: the degree N of x^N + 1. Must be a power of two.
        """
        self.coefficients = np.array(coefficients, dtype=dtype)
        self.modulus_degree = (
            modulus_degree if modulus_degree is not None else len(coefficients)
        )

        if self.modulus_degree != self.coefficients.shape[0]:
            raise ValueError(
                f"Modulus degree {self.modulus_degree} must be exactly "
                f"the number of coefficients {self.coefficients.shape[0]}."
            )
        if not is_power_of_two(self.modulus_degree):
            raise ValueError(
                "Modulus degree must be a power of two, but was "
                f"{self.modulus_degree}."
            )

    def __repr__(self):
        return f"IntPolynomial({self.coefficients.tolist()})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, IntPolynomial):
            return False
        return self.modulus_degree == other.modulus_degree and np.array_equal(
            self.coefficients, other.coefficients
        )

    def _assert_compatible(self, other):
        if self.modulus_degree != other.modulus_degree:
            raise ValueError(
                "Cannot compute two polynomials with differing "
                f"mod degrees {self.modulus_degree} and {other.modulus_degree}"
            )

    def __mul__(self, other):
        self._assert_compatible(other)
        if hasattr(other, "coefficient_modulus"):
            return other.__mul__(self)
        coeffs = negacyclic_polymul_complex_twist(self.coefficients, other.coefficients)
        return IntPolynomial(coeffs, self.modulus_degree)

    def __add__(self, other):
        self._assert_compatible(other)
        coeffs = self.coefficients + other.coefficients
        if hasattr(other, "coefficient_modulus"):
            return ModQPolynomial(
                coeffs,
                self.modulus_degree,
                getattr(other, "coefficient_modulus"),
                getattr(other, "ntt_params"),
            )
        return IntPolynomial(coeffs, self.modulus_degree)

    def __sub__(self, other):
        self._assert_compatible(other)
        coeffs = self.coefficients - other.coefficients
        if hasattr(other, "coefficient_modulus"):
            return ModQPolynomial(
                coeffs,
                self.modulus_degree,
                getattr(other, "coefficient_modulus"),
                getattr(other, "ntt_params"),
            )
        return IntPolynomial(coeffs, self.modulus_degree)

    def __neg__(self):
        return IntPolynomial(-self.coefficients, self.modulus_degree)

    def to_mod_q(self, ntt_params: NTTParams) -> "ModQPolynomial":
        """Reduce this polynomial's coefficients mod Q."""
        return ModQPolynomial(
            self.coefficients,
            modulus_degree=self.modulus_degree,
            coefficient_modulus=ntt_params.modulus,
            ntt_params=ntt_params,
        )

    def to_rns(self, ntt_params: Sequence[NTTParams]) -> "RNSPolynomial":
        """Reduce this polynomial's coefficients mod a list of primes."""
        # Reduce with Python ints before constructing each limb, since the
        # coefficients may be bigints (dtype=object) that overflow int64.
        limbs = tuple(
            ModQPolynomial(
                [int(c) % ntt_param.modulus for c in self.coefficients],
                modulus_degree=self.modulus_degree,
                coefficient_modulus=ntt_param.modulus,
                ntt_params=ntt_param,
            )
            for ntt_param in ntt_params
        )
        return RNSPolynomial(limbs)


class ModQPolynomial:
    """A polynomial with a ring modulus x^N + 1 and coeff modulus Q."""

    def __init__(
        self,
        coefficients: np.ndarray | Iterable[float],
        modulus_degree: int,
        coefficient_modulus: int,
        ntt_params: NTTParams,
    ):
        """Construct a polynomial.

        Args:
            coefficients: the coefficients of the polynomial.
            modulus_degree: the degree N of x^N + 1. Must be a power of two.
            coefficient_modulus: the modulus Q of the coefficient ring.
            ntt_params: the pre-computed NTT parameters needed to compute mul.
        """
        self.coefficients = np.array(coefficients, dtype=np.int64)
        self.modulus_degree = modulus_degree
        self.coefficient_modulus = coefficient_modulus
        self.ntt_params = ntt_params

        if self.modulus_degree != self.coefficients.shape[0]:
            raise ValueError(
                f"Modulus degree {self.modulus_degree} must be exactly "
                f"the number of coefficients {self.coefficients.shape[0]}."
            )
        if not is_power_of_two(self.modulus_degree):
            raise ValueError(
                "Modulus degree must be a power of two, but was "
                f"{self.modulus_degree}."
            )
        self.reduce()

    def reduce(self):
        self.coefficients = self.coefficients % self.coefficient_modulus

    def lift_to_signed_representative(self) -> IntPolynomial:
        coeffs = np.where(
            self.coefficients >= self.coefficient_modulus / 2.0,
            self.coefficients - self.coefficient_modulus,
            self.coefficients,
        )
        return IntPolynomial(coeffs, self.modulus_degree)

    def __repr__(self):
        return f"ModQPolynomial({self.coefficients.tolist()})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModQPolynomial):
            return False
        return (
            self.modulus_degree == other.modulus_degree
            and self.coefficient_modulus == other.coefficient_modulus
            and np.array_equal(self.coefficients, other.coefficients)
        )

    def assert_close(self, other, rtol=1e-5, atol=1e-8):
        if not hasattr(other, "coefficients"):
            raise AssertionError(f"Other object is not a polynomial-like: {other}")
        np.testing.assert_allclose(
            self.coefficients, other.coefficients, rtol=rtol, atol=atol
        )

    def _assert_compatible(self, other):
        if self.modulus_degree != other.modulus_degree:
            raise ValueError(
                "Cannot compute two polynomials with differing "
                f"mod degrees {self.modulus_degree} and {other.modulus_degree}"
            )
        if isinstance(other, ModQPolynomial):
            if self.coefficient_modulus != other.coefficient_modulus:
                raise ValueError(
                    "Cannot compute two polynomials with differing "
                    f"mod degrees {self.coefficient_modulus} and "
                    f"{other.coefficient_modulus}"
                )

    def __mul__(self, other):
        self._assert_compatible(other)
        coeffs = negacyclic_polymul_ntt(
            self.coefficients,
            other.coefficients,
            self.coefficient_modulus,
            self.ntt_params.precomputed,
        )
        return ModQPolynomial(
            coeffs, self.modulus_degree, self.coefficient_modulus, self.ntt_params
        )

    def __add__(self, other):
        self._assert_compatible(other)
        coeffs = self.coefficients + other.coefficients
        return ModQPolynomial(
            coeffs, self.modulus_degree, self.coefficient_modulus, self.ntt_params
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        self._assert_compatible(other)
        coeffs = self.coefficients - other.coefficients
        return ModQPolynomial(
            coeffs, self.modulus_degree, self.coefficient_modulus, self.ntt_params
        )

    def __rsub__(self, other):
        self._assert_compatible(other)
        coeffs = other.coefficients - self.coefficients
        return ModQPolynomial(
            coeffs, self.modulus_degree, self.coefficient_modulus, self.ntt_params
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return ModQPolynomial(
            -self.coefficients,
            self.modulus_degree,
            self.coefficient_modulus,
            self.ntt_params,
        )

    def __mod__(self, modulus: int):
        return ModQPolynomial(
            self.coefficients, self.modulus_degree, modulus, self.ntt_params
        )

    @staticmethod
    def zeros_like(other: "ModQPolynomial") -> "ModQPolynomial":
        return ModQPolynomial(
            np.zeros_like(other.coefficients),
            other.modulus_degree,
            other.coefficient_modulus,
            other.ntt_params,
        )


class RNSPolynomial:
    """A ModQPolynomial with a large Q, represented in the residue number system."""

    # A function mapping (limb index, polynomial) ->  polynomial
    LimbProcessor = Callable[[int, ModQPolynomial], ModQPolynomial]

    # A function mapping (limb index, polynomial, polynomial) ->  polynomial
    LimbZipper = Callable[[int, ModQPolynomial, ModQPolynomial], ModQPolynomial]

    # A function mapping (coefficient index, residues) -> T
    CoefficientProcessor = Callable[[int, np.ndarray], T]

    def __init__(self, limbs: Iterable[ModQPolynomial]):
        """Construct an RNS polynomial.

        Args:
            limbs: the RNS limbs of the polynomial
        """
        limbs = list(limbs)
        self.limbs = limbs

        if len(limbs) == 0:
            raise ValueError("Must have at least one limb")

        if not all(limb.modulus_degree == limbs[0].modulus_degree for limb in limbs):
            raise ValueError("All limbs must have the same modulus degree!")

    @property
    def ntt_params(self) -> tuple[NTTParams, ...]:
        return tuple(limb.ntt_params for limb in self.limbs)

    @property
    def moduli(self) -> list[int]:
        return [limb.coefficient_modulus for limb in self.limbs]

    @property
    def modulus_degree(self) -> int:
        return self.limbs[0].modulus_degree

    def __len__(self) -> int:
        return len(self.limbs)

    def __eq__(self, other) -> bool:
        return isinstance(other, RNSPolynomial) and self.limbs == other.limbs

    def apply_limbwise(self, func: LimbProcessor) -> "RNSPolynomial":
        """Apply func to each polynomial limb."""
        return RNSPolynomial(
            tuple(func(i, limb) for (i, limb) in enumerate(self.limbs))
        )

    @staticmethod
    def zip_limbwise(
        p1: "RNSPolynomial", p2: "RNSPolynomial", func: LimbZipper
    ) -> "RNSPolynomial":
        """Apply func to each polynomial limb."""
        return RNSPolynomial(
            tuple(
                func(i, l1, l2) for (i, (l1, l2)) in enumerate(zip(p1.limbs, p2.limbs))
            )
        )

    def apply_coefficientwise(self, func: CoefficientProcessor) -> list[T]:
        """Apply func to each coefficient's vector of residues."""
        # Shape (num_limbs, N): row j holds the coefficients of limb j, so
        # column i holds the residues of coefficient i.
        residues = np.stack([limb.coefficients for limb in self.limbs])
        return [func(i, residues[:, i]) for i in range(self.modulus_degree)]

    def lift_to_signed_representative(self) -> IntPolynomial:
        """RNS-reconstructs the polynomial to have bigint coefficients.

        If this RNSPolynomial has one limb, then no work is done and the
        returned IntPolynomial has dtype=np.int64, otherwise the RNS
        reconstruction is applied and the resulting IntPolynomial has
        dtype=object storing Python integers.
        """
        if len(self) == 1:
            return self.limbs[0].lift_to_signed_representative()

        moduli = self.moduli
        Q = math.prod(moduli)

        def recompose_centered(coeff_index: int, residues: np.ndarray) -> int:
            c = rns.recompose(residues, moduli)
            # recompose returns a value in [0, Q); center it in (-Q/2, Q/2].
            return c - Q if c > Q // 2 else c

        coefficients: list[int] = self.apply_coefficientwise(recompose_centered)
        return IntPolynomial(
            coefficients, modulus_degree=self.modulus_degree, dtype=object
        )

    def _assert_compatible(self, other):
        if isinstance(other, IntPolynomial):
            return True  # IntPolynomial is dynamically converted via to_rns

        if isinstance(other, RNSPolynomial):
            if len(self.limbs) != len(other.limbs):
                raise ValueError(
                    "Incompatible RNSPolynomial operands with differing limb counts"
                    f"{len(self.limbs)} and {len(other.limbs)}"
                )
            for l1, l2 in zip(self.limbs, other.limbs):
                l1._assert_compatible(l2)

        else:
            raise ValueError(
                f"Invalid operand of type {type(other)} "
                "to RNSPolynomial arithmetic routine."
            )

    def _maybe_cast_other(self, other):
        if isinstance(other, IntPolynomial):
            return other.to_rns(self.ntt_params)
        return other

    def __mul__(self, other):
        self._assert_compatible(other)
        other = self._maybe_cast_other(other)
        return RNSPolynomial.zip_limbwise(self, other, lambda i, l1, l2: l1 * l2)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        self._assert_compatible(other)
        other = self._maybe_cast_other(other)
        return RNSPolynomial.zip_limbwise(self, other, lambda i, l1, l2: l1 + l2)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        self._assert_compatible(other)
        other = self._maybe_cast_other(other)
        return RNSPolynomial.zip_limbwise(self, other, lambda i, l1, l2: l1 - l2)

    def __rsub__(self, other):
        self._assert_compatible(other)
        other = self._maybe_cast_other(other)
        return RNSPolynomial.zip_limbwise(other, self, lambda i, l1, l2: l1 - l2)

    def __neg__(self):
        return self.apply_limbwise(lambda i, limb: -limb)


def make_rns_polynomial(
    num_limbs: int,
    poly_modulus_degree: int,
    coeff_picker: CoeffPicker,
    bit_width: int = 60,
):
    """Generate an RNSPolynomial procedurally."""
    from primes import generate_ntt_primes

    primes = generate_ntt_primes(
        bit_width=bit_width,
        poly_modulus_degree=poly_modulus_degree,
        count=num_limbs,
    )

    limbs = []
    for prime in primes:
        coeffs = [coeff_picker(i, prime) for i in range(poly_modulus_degree)]
        ntt_params = NTTParams(degree=poly_modulus_degree, modulus=prime)
        limbs.append(ModQPolynomial(coeffs, poly_modulus_degree, prime, ntt_params))

    return RNSPolynomial(tuple(limbs))


def vandermonde_matrix(n: int) -> np.ndarray:
    """Computes the Vandermonde matrix used for canonical embedding."""
    twoN = 2 * n

    # Primitive roots of X^N + 1 are odd powers of the first primitive root of
    # unity.
    first_root = np.exp(2j * np.pi / twoN)
    odd_powers = np.arange(1, twoN, 2)
    roots = first_root**odd_powers

    powers = np.arange(n)
    vandermonde = roots[:, np.newaxis] ** powers
    return vandermonde


# For correctness testing, compare the FFT method to a more straightforward
# Vandermonde matrix method.
def canonical_embedding_vandermonde(poly_coeffs: np.ndarray) -> np.ndarray:
    """Computes the canonical embedding of a polynomial.

    This evaluates the polynomial at primitive complex roots of unity,
    using a Vandermonde matrix method.
    """
    n = len(poly_coeffs)
    vandermonde = vandermonde_matrix(n)
    # Vandermonde matrix turns evaluation into a matrix-vector mul
    return vandermonde @ poly_coeffs


def inverse_canonical_embedding_vandermonde(values: np.ndarray) -> np.ndarray:
    """Computes the inverse canonical embedding of a polynomial.

    The input `values` must be N complex values representing the evaluation of
    the result Polynomial at primitive (2N)-th roots of unity. In this case,
    the output is guaranteed to be real-valued (up to floating point roundoff
    error).
    """
    n = len(values)
    vandermonde = vandermonde_matrix(n)
    inv_vand = np.linalg.inv(vandermonde)
    return inv_vand @ values


def canonical_embedding(poly_coeffs: np.ndarray) -> np.ndarray:
    """Computes the canonical embedding of a polynomial.

    This evaluates the polynomial at primitive complex roots of unity,
    using an FFT-based method.
    """
    N = len(poly_coeffs)

    # 2N-point FFT evaluates at all 2N-th roots: omega^0, omega^1, ...,
    # omega^{2N-1}. But return only the odd entries for the primitive roots.
    padded = np.concatenate([poly_coeffs, np.zeros(N)])
    fft_result = np.fft.ifft(padded) * (2 * N)
    return fft_result[np.arange(1, 2 * N, 2)]


def inverse_canonical_embedding(values: np.ndarray) -> np.ndarray:
    """
    Invert the canonical embedding using FFT.

    The input `values` must be N complex values representing the evaluation of
    the result Polynomial at primitive (2N)-th roots of unity. In this case,
    the output is guaranteed to be real-valued (up to floating point roundoff
    error).
    """
    N = len(values)
    if not is_power_of_two(N):
        raise ValueError(
            f"Length of values must be a power of two, but was {len(values)}."
        )

    M = 2 * N
    ifft_result = np.fft.fft(values)
    zeta = np.exp(2j * np.pi / M)
    phase_factors = zeta ** (-np.arange(N))
    coeffs = (phase_factors * ifft_result) / N
    return coeffs
