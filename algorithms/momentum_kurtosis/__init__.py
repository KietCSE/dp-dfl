"""Two-layer malicious defense: Momentum Cosine (L1) + Kurtosis (L2).

- L1: Karimireddy, He, Jaggi "Learning from History for Byzantine Robust
      Optimization", ICML 2021. Momentum EMA on neighbor updates, cosine
      similarity against own momentum, MAD-adaptive threshold after warmup.
      Catches label-flip (even stealthy) and other direction-based attacks.

- L2: Sample Excess Kurtosis (see docs/dpsgd-dfl-kutosis-pseudocode.md).
      Catches scale, ALIE, non-Gaussian update distributions.

Composition: AND-of-clean — accept a neighbor only if it passes both layers.
"""
