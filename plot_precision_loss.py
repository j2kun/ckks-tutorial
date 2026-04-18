"""
Plot precision loss during CKKS encoding as a function of the scaling factor.

The precision loss comes from the rounding step in encoding: round(scale * coefficients).
Rounding introduces error uniformly distributed in [-0.5, 0.5] for each of the N coefficients.
The variance of this uniform distribution is 1/12, giving RMS = 1/(2*sqrt(3)).
After decoding, the canonical embedding combines these N independent errors, amplifying
the final error by approximately sqrt(N) (random walk behavior).

The theoretical RMS precision error is: sqrt(N / 12) / scale = sqrt(N) / (2 * sqrt(3) * scale)
For a scaling factor of 2^k and modulus degree N, the bound is sqrt(N/3) * 2^(-k).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from encoding import encode, decode, EncodingParams


def measure_precision_loss(message: np.ndarray, scale: float, N: int) -> float:
    """
    Encode and decode a message with given scale and N, return the RMS error.
    """
    params = EncodingParams(scale=scale, poly_modulus_degree=N)
    encoded = encode(message, params)
    decoded = decode(encoded, params)
    error = np.sqrt(np.mean(np.abs(message - decoded[: len(message)]) ** 2))
    return error


def main():
    N = 32  # polynomial modulus degree
    k_values = range(1, 41)  # scaling factors will be 2^k for k in this range
    num_trials = 100  # number of random messages to test at each scale

    np.random.seed(29598475)
    num_slots = N // 2

    # Measure precision loss for different scaling factors
    scaling_factors = []
    all_precision_losses = []  # list of lists
    theoretical_losses = []

    print(f"Testing with N = {N}, message length = {num_slots}, trials = {num_trials}")
    print("\nk\tScale (2^k)\tMean Loss\tStd Dev\t\tTheoretical")
    print("-" * 80)

    for k in k_values:
        scale = 2.0**k
        scaling_factors.append(scale)

        # Run multiple trials with different random messages
        losses_for_this_scale = []
        for _ in range(num_trials):
            message = np.random.randn(num_slots) + 1j * np.random.randn(num_slots)
            message = message * 100
            loss = measure_precision_loss(message, scale, N)
            losses_for_this_scale.append(loss)

        all_precision_losses.append(losses_for_this_scale)
        # RMS error from N independent uniform[-0.5, 0.5] rounding errors
        # Variance of uniform[-0.5, 0.5] is 1/12, so RMS = 1/(2*sqrt(3))
        theoretical = np.sqrt(N / 12) / scale
        theoretical_losses.append(theoretical)

        mean_loss = np.mean(losses_for_this_scale)
        std_loss = np.std(losses_for_this_scale)
        print(f"{k}\t2^{k}\t\t{mean_loss:.2e}\t{std_loss:.2e}\t{theoretical:.2e}")

    mean_precision_losses = [np.mean(losses) for losses in all_precision_losses]
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # First plot: Absolute precision loss compared to theoretical loss
    # Plot all individual measurements as scatter points
    for scale, losses in zip(scaling_factors, all_precision_losses):
        ax1.scatter([scale] * len(losses), losses, alpha=0.3, s=10, color="blue")

    # Plot mean values
    ax1.loglog(
        scaling_factors,
        mean_precision_losses,
        "o-",
        label="Mean Measured Loss",
        linewidth=2,
        markersize=6,
        color="darkblue",
    )

    # Plot theoretical bound
    ax1.loglog(
        scaling_factors,
        theoretical_losses,
        "--",
        label=f"Theoretical Bound (√({N}/12) / scale)",
        linewidth=2,
        alpha=0.7,
        color="red",
    )

    ax1.set_xlabel("Scaling Factor (2^k)", fontsize=12)
    ax1.set_ylabel("Precision Loss (RMS Error)", fontsize=12)
    ax1.set_title(
        f"CKKS Encoding Precision Loss vs Scaling Factor (N={N}, {num_trials} trials per scale)",
        fontsize=14,
    )
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(fontsize=11)

    # Second plot: Relative error (normalized by theoretical)
    # Calculate relative errors for each trial
    for scale, losses, theoretical in zip(
        scaling_factors, all_precision_losses, theoretical_losses
    ):
        relative_errors = [(loss - theoretical) / theoretical for loss in losses]
        ax2.scatter(
            [scale] * len(relative_errors),
            relative_errors,
            alpha=0.3,
            s=10,
            color="purple",
        )

    # Plot mean relative error
    mean_relative_errors = [
        (mean - theo) / theo
        for mean, theo in zip(mean_precision_losses, theoretical_losses)
    ]
    ax2.semilogx(
        scaling_factors,
        mean_relative_errors,
        "o-",
        color="darkviolet",
        label="Mean Relative Error",
        linewidth=2,
        markersize=6,
    )

    # Add a horizontal line at 0
    ax2.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Perfect Match",
    )

    ax2.set_xlabel("Scaling Factor (2^k)", fontsize=12)
    ax2.set_ylabel(
        "Relative Error: (Measured - Theoretical) / Theoretical", fontsize=12
    )
    ax2.set_title("Relative Deviation from Theoretical Bound", fontsize=14)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("precision_loss_vs_scale.png", dpi=300, bbox_inches="tight")
    print("Plot saved to precision_loss_vs_scale.png")
    plt.show()

    # Precision loss vs N for fixed scale
    fixed_scale = 2.0**40
    N_values = [2**i for i in range(5, 16)]  # 32, 64, 128, ..., 32768
    num_trials_N = 1000
    log_scale = int(math.log2(fixed_scale))

    print(f"\n\nTesting with fixed scale = 2^{log_scale}, trials = {num_trials_N}")
    print("\nN\tNum Slots\tMean Loss\tStd Dev\t\tTheoretical")
    print("-" * 80)

    all_losses_by_N = []
    theoretical_losses_by_N = []

    for N_test in N_values:
        num_slots_test = N_test // 2

        # Run multiple trials
        losses_for_this_N = []
        for _ in range(num_trials_N):
            message = np.random.randn(num_slots_test) + 1j * np.random.randn(
                num_slots_test
            )
            message = message * 100
            loss = measure_precision_loss(message, fixed_scale, N_test)
            losses_for_this_N.append(loss)

        all_losses_by_N.append(losses_for_this_N)
        theoretical = np.sqrt(N_test / 12) / fixed_scale
        theoretical_losses_by_N.append(theoretical)

        mean_loss = np.mean(losses_for_this_N)
        std_loss = np.std(losses_for_this_N)
        print(
            f"{N_test}\t{num_slots_test}\t\t{mean_loss:.2e}\t{std_loss:.2e}\t{theoretical:.2e}"
        )

    mean_losses_by_N = [np.mean(losses) for losses in all_losses_by_N]

    # Create new figure for N variation plot
    _, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 12))

    # First plot: Absolute precision loss vs N
    for N_test, losses in zip(N_values, all_losses_by_N):
        ax3.scatter([N_test] * len(losses), losses, alpha=0.3, s=10, color="green")

    # Plot mean values
    ax3.loglog(
        N_values,
        mean_losses_by_N,
        "o-",
        label="Mean Measured Loss",
        linewidth=2,
        markersize=6,
        color="darkgreen",
    )

    # Plot theoretical bound
    ax3.loglog(
        N_values,
        theoretical_losses_by_N,
        "--",
        label=f"Theoretical Bound (√(N/12) / 2^{log_scale})",
        linewidth=2,
        alpha=0.7,
        color="red",
    )

    ax3.set_xlabel("Polynomial Modulus Degree (N)", fontsize=12)
    ax3.set_ylabel("Precision Loss (RMS Error)", fontsize=12)
    ax3.set_title(
        f"CKKS Encoding Precision Loss vs N (Scale=2^{log_scale}, {num_trials_N} trials per N)",
        fontsize=14,
    )
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    ax3.legend(fontsize=11)

    # Second plot: Relative error
    for N_test, losses, theoretical in zip(
        N_values, all_losses_by_N, theoretical_losses_by_N
    ):
        relative_errors = [(loss - theoretical) / theoretical for loss in losses]
        ax4.scatter(
            [N_test] * len(relative_errors),
            relative_errors,
            alpha=0.3,
            s=10,
            color="orange",
        )

    # Plot mean relative error
    mean_relative_errors_N = [
        (mean - theo) / theo
        for mean, theo in zip(mean_losses_by_N, theoretical_losses_by_N)
    ]
    ax4.semilogx(
        N_values,
        mean_relative_errors_N,
        "o-",
        color="darkorange",
        label="Mean Relative Error",
        linewidth=2,
        markersize=6,
    )

    # Add a horizontal line at 0
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
    ax4.set_title("Relative Deviation from Theoretical Bound", fontsize=14)
    ax4.grid(True, which="both", ls="-", alpha=0.2)
    ax4.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("precision_loss_vs_N.png", dpi=300, bbox_inches="tight")
    print("\nPlot saved to precision_loss_vs_N.png")
    plt.show()


if __name__ == "__main__":
    main()
