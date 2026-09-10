"""Random generation utilities."""

import abc
import secrets
import numpy as np


class RandomSource(abc.ABC):
    """An interface for random number generation in CKKS."""

    @abc.abstractmethod
    def gen_gaussian_poly(
        self, degree: int, modulus: int, sigma: float = 3.19
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def gen_ternary_poly(self, degree: int) -> np.ndarray:
        pass

    @abc.abstractmethod
    def gen_uniform_poly(self, degree: int, modulus: int) -> np.ndarray:
        pass


class SecureRandomSource(RandomSource):
    """A random source that uses secrets for secure RNG."""

    def __init__(self):
        self.rng = secrets.SystemRandom()

    def gen_gaussian_poly(
        self, degree: int, modulus: int, sigma: float = 3.19
    ) -> np.ndarray:
        return np.array(
            [round(self.rng.normalvariate(0, sigma)) % modulus for _ in range(degree)],
            dtype=np.int64,
        )

    def gen_ternary_poly(self, degree: int) -> np.ndarray:
        return np.array(
            [self.rng.choice([-1, 0, 1]) for _ in range(degree)], dtype=np.int64
        )

    def gen_uniform_poly(self, degree: int, modulus: int) -> np.ndarray:
        return np.array(
            [self.rng.randrange(0, modulus) for _ in range(degree)], dtype=np.int64
        )


class ZeroNoiseRandomSource(SecureRandomSource):
    """A random source that zeros out the gaussian noise.

    This is meant to be used only in testing, when the test wants to verify exact
    algebraic correctness of operations that normally involve CKKS noise and
    preclude an exact comparison.
    """

    def gen_gaussian_poly(
        self, degree: int, modulus: int, sigma: float = 3.19
    ) -> np.ndarray:
        return np.zeros((degree,), dtype=np.int64)


class TestRandomSource(SecureRandomSource):
    """A random source that can be seeded, for testing."""

    def __init__(self, seed: int):
        import random as std_random

        self.rng = std_random.Random(seed)
