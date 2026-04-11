# Strategic Noise Injection for Robust and Accurate Decentralized Federated Learning (DFL)

## 1. Introduction

Decentralized Federated Learning (DFL) enables multiple clients to collaboratively train a model without sharing raw data and without relying on a central server. While this architecture improves scalability and removes single points of failure, it significantly increases the attack surface.

Two primary threats are:

- **Model Poisoning Attack**: Adversaries manipulate gradients to corrupt model updates or inject backdoors.
- **Inference Attack**: Adversaries infer sensitive information from gradients or model updates.

Differential Privacy (DP) mitigates inference attacks by adding noise to gradients. However, standard DP mechanisms suffer from:

- Isotropic (structure-agnostic) noise
- Ineffectiveness against poisoning
- Accuracy degradation due to excessive noise

---

## 2. Core Idea

We reformulate noise injection as a **strategic optimization problem** under a game-theoretic framework.

### Attacker vs Defender

- Attacker perturbs gradient:
  \[ g + \delta \]
- Defender injects noise:
  \[ n \]

Final shared gradient:
\[
\hat{g} = g + \delta + n
\]

### Defender Objective (Minimax)

\[
\min*n \max*\delta ||\delta + n||
\]

Goal: Design structured noise to neutralize adversarial perturbations while preserving learning signal.

---

## 3. Proposed Method

### 3.1 Gradient Clipping

\[
g_i^{(t)} \leftarrow \frac{g_i^{(t)}}{\max(1, ||g_i^{(t)}|| / C)}
\]

Purpose:

- Bound sensitivity for DP
- Limit extreme updates

---

### 3.2 Attack Direction Estimation

\[
v_i^{(t)} = g_i^{(t)} - g_i^{(t-1)}
\]

Interpretation:

- Small \(v_i\): stable training
- Large \(v_i\): potential attack signal

---

### 3.3 Trust-aware Scoring

\[
trust_i = \cos(g_i^{(t)}, g_i^{(t-1)})
\]

Properties:

- \(\approx 1\): stable
- \(\approx 0\): drift
- \(< 0\): adversarial flip

Adaptive noise weight:
\[
\alpha_i = \alpha (1 - trust_i)
\]

---

### 3.4 Directional Noise Injection

\[
n\_{attack} = \alpha_i v_i^{(t)}
\]

Effect:

- Cancels adversarial direction
- Minimal effect on stable gradients

---

### 3.5 Orthogonal Noise

Sample Gaussian:
\[
z \sim \mathcal{N}(0, \sigma^2)
\]

Project orthogonal to gradient:
\[
n\_{orth} = z - \frac{z \cdot g}{||g||^2} g
\]

Effect:

- Preserves optimization direction
- Provides DP randomness

---

### 3.6 Spectrum-aware Noise

Spectral decomposition:
\[
g = U \Lambda V^T
\]

Noise shaping:
\[
n\_{spec} = U \cdot diag(\lambda^{-1}) \cdot r
\]

Effect:

- Less noise on important directions
- More noise on less critical ones

---

### 3.7 Total Noise

\[
n = n*{attack} + n*{orth} + n\_{spec}
\]

Final gradient:
\[
\hat{g} = g + n
\]

---

## 4. Accuracy Enhancement Mechanisms

To counteract noise-induced degradation, we integrate signal recovery and optimization stabilization.

---

### 4.1 Momentum-based Optimization

\[
m*t = \beta m*{t-1} + (1 - \beta) \hat{g}_t
\]
\[
w_{t+1} = w_t - \eta m_t
\]

Benefits:

- Reduces stochastic noise
- Stabilizes updates

---

### 4.2 Gradient Denoising (EMA)

\[
\tilde{g}_t = \gamma \tilde{g}_{t-1} + (1 - \gamma) \hat{g}\_t
\]

Benefits:

- Filters high-frequency noise
- Preserves trend

---

### 4.3 Noise Annealing

\[
\sigma_t = \sigma_0 e^{-\kappa t}
\]

Benefits:

- High privacy early
- High accuracy later

---

### 4.4 Gradient Alignment Filtering

Reject or down-weight updates if:
\[
\cos(\hat{g}_t, \tilde{g}_{t-1}) < \tau
\]

Effect:

- Removes inconsistent gradients
- Improves convergence

---

### 4.5 Trust-aware Learning Rate

\[
\eta_i = \eta \cdot trust_i
\]

Effect:

- Reliable clients dominate updates

---

### 4.6 Variance Reduction (SCAFFOLD-style)

\[
g_i' = g_i - c_i + c
\]

Effect:

- Reduces client drift
- Accelerates convergence

---

### 4.7 Two-Track Model Update

Maintain:

- Clean model: \(w^{clean}\)
- Robust model: \(w^{robust}\)

Combine:
\[
w = \lambda w^{clean} + (1 - \lambda) w^{robust}
\]

Effect:

- Balance robustness and accuracy

---

## 5. Full Algorithm

```
Initialize w0, gi(-1)=0, momentum m=0

for each round t:

  compute gradient gi(t)
  clip gradient

  estimate vi(t) = gi(t) - gi(t-1)

  trust_i = cos(gi(t), gi(t-1))
  αi = α(1 - trust_i)

  n_attack = αi * vi(t)

  z ~ N(0, σt^2)
  n_orth = z - proj_g(z)

  n_spec = spectrum_noise(gi(t))

  n = n_attack + n_orth + n_spec

  ĝ = gi(t) + n

  denoise ĝ via EMA

  update momentum m
  update model w

  store gi(t)
```

---

## 6. Key Contributions

1. **Game-theoretic noise design** for adversarial robustness
2. **Directional + orthogonal + spectral noise decomposition**
3. **Trust-aware adaptive mechanism**
4. **Integrated signal recovery pipeline** (EMA + momentum)
5. **Dynamic noise scheduling**
6. **Hybrid robustness-accuracy architecture**

---

## 7. Expected Outcomes

- Strong resistance to poisoning attacks
- Reduced inference leakage
- Improved convergence stability
- Minimal accuracy degradation compared to standard DP-FL

---

**End of Document**
