"""Deterministic Miller-Rabin primality testing.

Ported from https://www.jeremykun.com/2026/04/07/deterministic-miller-rabin/
"""

# Bases to test. Using the first 12 prime bases makes the test deterministic
# for all n < 318665857834031151167461 (> 2^64). See https://oeis.org/A014233.
BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
DETERMINISTIC_BOUND = 318665857834031151167461


def count_trailing_zeros(n: int) -> int:
    if n == 0:
        return 64
    return (n & -n).bit_length() - 1


def is_prime(n: int) -> bool:
    """Return True if n is prime, deterministically for n < DETERMINISTIC_BOUND."""
    if n >= DETERMINISTIC_BOUND:
        raise ValueError(
            f"is_prime is only deterministic for n < {DETERMINISTIC_BOUND}, got {n}"
        )
    if n < 2:
        return False
    if n < 4:
        return True
    if not n & 1:
        return False

    # Write n - 1 = 2^s * d with d odd.
    d = n - 1
    s = count_trailing_zeros(d)
    d >>= s

    for a in BASES:
        if n <= a:
            break
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(1, s):
            x = (x * x) % n
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True


def generate_ntt_primes(
    bit_width: int, poly_modulus_degree: int, count: int
) -> list[int]:
    """Generate `count` distinct NTT-friendly primes with exactly `bit_width` bits.

    A prime q is NTT-friendly for the ring Z_q[x] / (x^N + 1) if q = 1 mod 2N,
    since then Z_q contains a primitive 2N-th root of unity.

    The search starts at the largest such candidate below 2^bit_width and walks
    downward in steps of 2N, so the returned primes are in decreasing order and
    as close to 2^bit_width as possible.
    """
    m = 2 * poly_modulus_degree
    lower = 1 << (bit_width - 1)
    upper = 1 << bit_width

    # Largest candidate q < 2^bit_width with q = 1 mod 2N.
    candidate = ((upper - 2) // m) * m + 1

    primes: list[int] = []
    while len(primes) < count and candidate >= lower:
        if is_prime(candidate):
            primes.append(candidate)
        candidate -= m

    if len(primes) < count:
        raise ValueError(
            f"Only found {len(primes)} of {count} requested {bit_width}-bit "
            f"primes congruent to 1 mod {m}."
        )
    return primes
