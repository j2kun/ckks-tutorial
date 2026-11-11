"""
Plot the noise growth from CKKS key switching with gadget decomposition.

Key switching from s to s' decomposes c1 into k = 64 / log(B) digits d_i in
[0, B), and the key switching key holds encryptions (b_i, a_i) of B^i * s under
s', each with its own fresh noise e_i. The switched ciphertext decrypts to

    c0 + sum_i d_i * (b_i + a_i * s') = c0 + c1 * s + sum_i d_i * e_i

so key switching adds exactly the noise polynomial sum_i d_i * e_i.

Each coefficient of d_i * e_i is a sum of N products of a digit and an
independent zero-mean noise coefficient, so its variance is N * E[d^2] * Var(e).
Summing over the digits gives

    Var(key switching noise) = N * Var(e) * sum_i E[d_i^2]

where, for a digit uniform in [0, 2^w), E[d^2] = (2^w - 1)(2^{w+1} - 1) / 6.
Since Q < 2^64, the most significant digits only see the bits of Q, and the
digits above the bit length of Q are always zero.

For symmetric key switching keys, e_i is a fresh rounded Gaussian with variance
Var(e) ~= sigma^2 + 1/12. For asymmetric keys, e_i = e_pk * u + e0 + e1 * s'
with ternary u and s', so Var(e_i) ~= Var(e) * (1 + 4N/3).
"""

import numpy as np
import matplotlib.pyplot as plt

from context import CKKSContext
from encryption import extract_noise
from key_switch import key_switch
from ntt import NTT_64_BIT_PRIME
from params import EncodingParams, KeySwitchingKeyParams
from rng import SeededRandomSource

SIGMA = 3.19
# Variance of round(N(0, sigma^2)), for sigma well above 1.
GAUSSIAN_VARIANCE = SIGMA**2 + 1 / 12
# Variance of a uniform ternary coefficient in {-1, 0, 1}.
TERNARY_VARIANCE = 2 / 3
DIGIT_BIT_LENGTH = 64


def digit_second_moment_sum(log_base: int, modulus: int) -> float:
    """Compute sum_i E[d_i^2] for the digits of a uniform value in [0, modulus)."""
    q_bits = modulus.bit_length()
    total = 0.0
    for low_bit in range(0, DIGIT_BIT_LENGTH, log_base):
        width = max(0, min(log_base, q_bits - low_bit))
        B = 2**width
        total += (B - 1) * (2 * B - 1) / 6
    return total


def theoretical_noise(log_base: int, N: int, asymmetric: bool) -> float:
    """The theoretical RMS of the noise added by key switching."""
    ksk_noise_variance = GAUSSIAN_VARIANCE
    if asymmetric:
        ksk_noise_variance *= 1 + 2 * N * TERNARY_VARIANCE
    moment_sum = digit_second_moment_sum(log_base, NTT_64_BIT_PRIME)
    return np.sqrt(N * ksk_noise_variance * moment_sum)


def measure_key_switch_noise(context: CKKSContext, asymmetric: bool) -> float:
    """Key switch a fresh encryption of a random message, return the RMS of the
    noise added by key switching."""
    num_slots = context.encoding_params.poly_modulus_degree // 2
    message = np.random.randn(num_slots) + 1j * np.random.randn(num_slots)
    pt = context.encode(message)

    if asymmetric:
        source_sk, source_pk = context.generate_asymmetric_keypair()
        target_sk, target_pk = context.generate_asymmetric_keypair()
        ct = context.encrypt_asymmetric(pt, source_pk)
        ksk = context.generate_asymmetric_key_switching_key(source_sk, target_pk)
    else:
        source_sk = context.generate_symmetric_private_key()
        target_sk = context.generate_symmetric_private_key()
        ct = context.encrypt_symmetric(pt, source_sk)
        ksk = context.generate_symmetric_key_switching_key(source_sk, target_sk)

    switched_ct = key_switch(ct, ksk)

    noise_before = extract_noise(ct, source_sk, pt).lift_to_signed_representative()
    noise_after = extract_noise(
        switched_ct, target_sk, pt
    ).lift_to_signed_representative()
    added_noise = (noise_after - noise_before).coefficients.astype(np.float64)
    return np.sqrt(np.mean(added_noise**2))


def make_context(N: int, log_base: int, seed: int) -> CKKSContext:
    params = EncodingParams(
        scale=2**20, poly_modulus_degree=N, coefficient_modulus=NTT_64_BIT_PRIME
    )
    context = CKKSContext(params, KeySwitchingKeyParams(log_base))
    context.rng = SeededRandomSource(seed)
    return context


def plot_noise(
    ax_abs,
    ax_rel,
    x_values,
    all_noises,
    theoretical_noises,
    label: str,
    scatter_color: str,
    line_color: str,
):
    """Plot measured vs theoretical noise, and the relative deviation."""
    mean_noises = [np.mean(noises) for noises in all_noises]

    for x, noises in zip(x_values, all_noises):
        ax_abs.scatter([x] * len(noises), noises, alpha=0.3, s=10, color=scatter_color)
    ax_abs.plot(
        x_values,
        mean_noises,
        "o-",
        label=f"Mean Measured Noise ({label})",
        linewidth=2,
        markersize=6,
        color=line_color,
    )
    ax_abs.plot(
        x_values,
        theoretical_noises,
        "--",
        label=f"Theoretical ({label})",
        linewidth=2,
        alpha=0.7,
        color="red",
    )

    for x, noises, theoretical in zip(x_values, all_noises, theoretical_noises):
        relative_errors = [(noise - theoretical) / theoretical for noise in noises]
        ax_rel.scatter(
            [x] * len(relative_errors),
            relative_errors,
            alpha=0.3,
            s=10,
            color=scatter_color,
        )
    mean_relative_errors = [
        (mean - theo) / theo for mean, theo in zip(mean_noises, theoretical_noises)
    ]
    ax_rel.plot(
        x_values,
        mean_relative_errors,
        "o-",
        color=line_color,
        label=f"Mean Relative Error ({label})",
        linewidth=2,
        markersize=6,
    )


def main():
    np.random.seed(29598475)

    # Key switching noise vs decomposition base for fixed N
    N = 8192
    log_bases = [1, 2, 4, 8, 16, 32]
    num_trials = 30

    print(f"Testing with N = {N}, trials = {num_trials}")
    print("\nKSK\t\tlog(B)\tDigits\tMean Noise\tStd Dev\t\tTheoretical")
    print("-" * 80)

    results = {}
    for asymmetric in [False, True]:
        kind = "asymmetric" if asymmetric else "symmetric"
        all_noises = []
        theoretical_noises = []
        for log_base in log_bases:
            context = make_context(N, log_base, seed=log_base)
            noises = [
                measure_key_switch_noise(context, asymmetric) for _ in range(num_trials)
            ]
            theoretical = theoretical_noise(log_base, N, asymmetric)
            all_noises.append(noises)
            theoretical_noises.append(theoretical)
            print(
                f"{kind}\t{log_base}\t{DIGIT_BIT_LENGTH // log_base}\t"
                f"{np.mean(noises):.2e}\t{np.std(noises):.2e}\t{theoretical:.2e}"
            )
        results[kind] = (all_noises, theoretical_noises)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax2.set_xscale("log", base=2)
    for ax in (ax1, ax2):
        ax.set_xticks(log_bases, [str(b) for b in log_bases])
    styles = {
        "symmetric": ("blue", "darkblue"),
        "asymmetric": ("green", "darkgreen"),
    }
    for kind, (all_noises, theoretical_noises) in results.items():
        scatter_color, line_color = styles[kind]
        plot_noise(
            ax1,
            ax2,
            log_bases,
            all_noises,
            theoretical_noises,
            kind,
            scatter_color,
            line_color,
        )

    ax1.set_xlabel("Decomposition Base Bits log(B)", fontsize=12)
    ax1.set_ylabel("Key Switching Noise (RMS of coefficients)", fontsize=12)
    ax1.set_title(
        f"CKKS Key Switching Noise vs Decomposition Base (N={N}, "
        f"{num_trials} trials per base)",
        fontsize=14,
    )
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(fontsize=11)

    ax2.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Perfect Match",
    )
    ax2.set_xlabel("Decomposition Base Bits log(B)", fontsize=12)
    ax2.set_ylabel(
        "Relative Error: (Measured - Theoretical) / Theoretical", fontsize=12
    )
    ax2.set_title("Relative Deviation from Theoretical Noise", fontsize=14)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("key_switch_noise_vs_base.png", dpi=300, bbox_inches="tight")
    print("Plot saved to key_switch_noise_vs_base.png")
    plt.close()

    # Key switching noise vs N for fixed decomposition base
    fixed_log_base = 4
    N_values = [2**i for i in range(8, 16)]  # 256, 512, ..., 32768
    num_trials_N = 20

    print(
        f"\n\nTesting symmetric key switching with fixed log(B) = {fixed_log_base}, "
        f"trials = {num_trials_N}"
    )
    print("\nN\tMean Noise\tStd Dev\t\tTheoretical")
    print("-" * 80)

    all_noises_by_N = []
    theoretical_noises_by_N = []
    for N_test in N_values:
        context = make_context(N_test, fixed_log_base, seed=N_test)
        noises = [
            measure_key_switch_noise(context, asymmetric=False)
            for _ in range(num_trials_N)
        ]
        theoretical = theoretical_noise(fixed_log_base, N_test, asymmetric=False)
        all_noises_by_N.append(noises)
        theoretical_noises_by_N.append(theoretical)
        print(
            f"{N_test}\t{np.mean(noises):.2e}\t{np.std(noises):.2e}\t"
            f"{theoretical:.2e}"
        )

    _, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 12))
    ax3.set_xscale("log", base=2)
    ax3.set_yscale("log", base=2)
    ax4.set_xscale("log", base=2)
    plot_noise(
        ax3,
        ax4,
        N_values,
        all_noises_by_N,
        theoretical_noises_by_N,
        "symmetric",
        "orange",
        "darkorange",
    )

    ax3.set_xlabel("Polynomial Modulus Degree (N)", fontsize=12)
    ax3.set_ylabel("Key Switching Noise (RMS of coefficients)", fontsize=12)
    ax3.set_title(
        f"CKKS Key Switching Noise vs N (log(B)={fixed_log_base}, "
        f"{num_trials_N} trials per N)",
        fontsize=14,
    )
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    ax3.legend(fontsize=11)

    ax4.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Perfect Match",
    )
    ax4.set_xlabel("Polynomial Modulus Degree (N)", fontsize=12)
    ax4.set_ylabel(
        "Relative Error: (Measured - Theoretical) / Theoretical", fontsize=12
    )
    ax4.set_title("Relative Deviation from Theoretical Noise", fontsize=14)
    ax4.grid(True, which="both", ls="-", alpha=0.2)
    ax4.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("key_switch_noise_vs_N.png", dpi=300, bbox_inches="tight")
    print("\nPlot saved to key_switch_noise_vs_N.png")
    plt.close()


if __name__ == "__main__":
    main()
