"""Solve noise_multiplier z such that DP-FedAvg achieves a target epsilon
after R rounds (post_training mode, fixed sampling_rate q).

Usage:
    python scripts/solve_noise_mult.py [target_eps] [n_rounds] [q] [delta]

Defaults: target_eps=16, n_rounds=50, q=1.0, delta=1e-5.
"""

import sys

from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

ALPHAS = [
    1.01, 1.05, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20,
    32, 50, 64, 100, 128, 256, 512, 1000,
]


def eps_for_z(z: float, n_rounds: int, q: float = 1.0, delta: float = 1e-5) -> float:
    """RDP cost of n_rounds Sampled Gaussian Mechanism applications, in (ε,δ)-DP."""
    rdp = compute_rdp(q=q, noise_multiplier=z, steps=n_rounds, orders=ALPHAS)
    eps, _ = get_privacy_spent(orders=ALPHAS, rdp=rdp, delta=delta)
    return float(eps)


def find_z(target_eps: float, n_rounds: int, q: float = 1.0,
           delta: float = 1e-5, lo: float = 0.1, hi: float = 50.0,
           tol: float = 1e-3) -> float:
    """Binary search smallest z whose accumulated eps <= target_eps."""
    for _ in range(80):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if eps_for_z(mid, n_rounds, q, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    q = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    delta = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5

    z = find_z(target, rounds, q, delta)
    eps = eps_for_z(z, rounds, q, delta)
    print(f"Target eps={target}, R={rounds}, q={q}, delta={delta:.0e}")
    print(f"  noise_mult (z) = {z:.4f}")
    print(f"  verified eps   = {eps:.4f}")


if __name__ == "__main__":
    main()
