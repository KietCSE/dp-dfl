"""CFL-DP-FedAvg: Centralized Federated Averaging with user-level Central DP.

Implements McMahan et al. 2018 "Learning Differentially Private Recurrent
Language Models" (ICLR 2018). Trusted server adds Gaussian noise once
post-aggregation; clients perform plain local SGD.

Subset chosen for this project (paper-faithful):
  - FlatClip (Eq. 1 of paper) — single global L2 bound S on each Δ_k
  - Estimator f̃_f (fixed denominator) — Δ = Σ w_k·Δ_k / (qW)
  - Rényi DP accountant (Mironov 2019 SGM) — modern equivalent of Moments
    Accountant from the original paper
  - Uniform user weights w_k = 1, W = K (n_nodes)
"""
