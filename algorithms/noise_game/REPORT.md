# Noise Game — Implementation Report

## Implemented: 4-Layer Architecture

### Layer 1 — DP Gaussian Noise
- `n_DP ~ N(0, sigma_DP^2 * I)` injected riêng biệt sau clipping
- sigma_DP floor = `C * sqrt(2*ln(1.25/delta)) / epsilon_max` (guarantee DP)
- File: `mechanism.py` → `compute_dp_noise()`

### Layer 2 — Adaptive Sigma Scheduling
- `sigma_DP = f(budget, attack_signal)`
- Base annealing: `sigma_0 * exp(-kappa * t)`
- Budget awareness: `max(0.1, 1 - eps_spent/eps_max)` — giảm noise khi sắp hết budget
- Threat response: `1 + attack_signal` — tăng noise khi phát hiện attack
- Internal RDP tracker: heuristic `eps += C^2 / (2*sigma_dp^2)` (alpha=2), chỉ dùng cho scheduler, không thay thế RenyiAccountant chính thức
- `sigma_strat = beta_strat * sigma_DP` — coupling strategic noise vào DP noise
- File: `mechanism.py` → `compute_sigma_dp()`

### Layer 3 — Strategic Noise (3 components)
- **Directional**: `alpha * (1-trust) * (g_curr - g_prev)` — cancel hướng attack
- **Orthogonal**: `z - (z·g/||g||^2)*g` — noise vuông góc, giữ hướng gradient
- **Spectrum-aware**: truncated SVD → inverse-weighted noise trên singular values
- **Normalization**: `hat_n = normalize(sum)` rồi scale `n_strategic = sigma_strat * hat_n`
- File: `mechanism.py` → `directional_noise()`, `orthogonal_noise()`, `spectrum_noise()`

### Layer 4 — Recovery (unchanged)
- EMA denoising: `gamma * ema_old + (1-gamma) * g_hat`
- Momentum: `beta * m_old + (1-beta) * g_hat`
- Alignment filtering: `cos(g_hat, ema_prev) < tau` → reject, dùng EMA thay thế
- Trust-aware LR: `final = m * max(trust, 0)`
- SCAFFOLD variance reduction: `g' = g - c_i + c_global`
- Two-track model: `w = lambda*clean + (1-lambda)*robust`
- File: `node.py`, `simulator.py`

### Budget Constraint
- `||n_DP||^2 + ||n_strategic||^2 <= sigma_total^2`
- Scale cả hai proportionally khi vượt
- File: `mechanism.py` → `_enforce_budget()`

### NSR Monitoring
- `NSR = ||total_noise|| / ||gradient||`
- Log warning khi NSR > `nsr_warn`
- Metrics: `avg_nsr`, `avg_sigma_dp` ghi vào round logs

---

## Decisions Made

| Decision | Chosen | Alternatives Considered | Reason |
|---|---|---|---|
| `attack_signal` | `(1 - trust)` | `\|\|v_t\|\|/\|\|g_t\|\|` (gradient change ratio), kết hợp cả hai | KISS — trust thấp = attack cao, đủ cho heuristic scheduler |
| `sigma_total` | Cố định 3.0 | Scale theo `sqrt(param_dim)`, per-dimension cap `\|\|n\|\|^2/D <= cap` | User chọn giữ cố định. Lưu ý: trong high-dim (D=80K+), budget constraint sẽ scale down noise mạnh |
| CIFAR-10 config | Update luôn | Chỉ update MNIST, CIFAR-10 dùng default | Giữ consistency giữa 2 config files |
| `trust` param trong `compute_sigma_dp` | Loại bỏ | Giữ cả `trust` + `attack_signal` | Redundant vì `attack_signal = 1 - trust`, IDE warning unused param |

---

## Options NOT in Algorithm (đã xem xét nhưng loại)

### 1. Scale sigma_total theo sqrt(D)
- Ý tưởng: `sigma_total_effective = sigma_total * sqrt(param_dim)`
- Lý do loại: User chọn giữ cố định
- Hệ quả: Với D=80K (MLP), `||n_DP|| ~ sigma*sqrt(D) ~ 63`, budget cap=3 sẽ scale noise xuống ~0.05x. DP noise bị giảm mạnh trong practice
- **Khuyến nghị**: Nếu accuracy quá cao (noise không đủ mạnh cho DP) hoặc epsilon tích lũy quá nhanh, xem xét tăng `sigma_total` hoặc chuyển sang scale theo sqrt(D)

### 2. Rich attack signal: gradient change ratio
- Ý tưởng: `attack_signal = ||g_curr - g_prev|| / ||g_curr||` (relative gradient change)
- Lý do loại: KISS, `(1-trust)` đủ dùng
- Ưu điểm bỏ lỡ: Phân biệt được gradual drift (trust giảm từ từ) vs sudden attack (gradient nhảy lớn)

### 3. Combined attack signal
- Ý tưởng: `threat = (1-trust) + ||v_t||/||g_t||`
- Lý do loại: Phức tạp hơn, chưa rõ gain
- Có thể revisit nếu adaptive scheduler không phản ứng đủ nhanh với certain attack types

### 4. Per-dimension noise cap
- Ý tưởng: `||n||^2 / D <= sigma_total^2` (cap variance per dimension)
- Lý do loại: User chọn fixed cap
- Ưu điểm: Tự động adapt theo model size, không cần tune sigma_total khi đổi model

### 5. Full RDP accounting trong mechanism
- Ý tưởng: Dùng min-over-alpha RDP conversion thay vì alpha=2 heuristic
- Lý do loại: YAGNI — internal tracker chỉ là heuristic cho scheduler, RenyiAccountant trong simulator làm accounting chính thức
- Trade-off: epsilon_spent trong mechanism có thể drift so với epsilon thực, nhưng chỉ ảnh hưởng scheduling, không ảnh hưởng privacy guarantee

---

## Config Changes

### New fields added to `NoiseGameConfig`
```python
beta_strat: float = 0.5      # sigma_strat = beta * sigma_DP
sigma_total: float = 3.0     # total noise energy cap  
nsr_warn: float = 5.0        # NSR warning threshold
```

### Removed from dp section (by user)
```yaml
noise_mult: 1.1  # removed — noise_game self-manages noise, không dùng noise_mult
```

### Files modified
- `config.py` — +3 fields
- `config/noise_game.yaml` — +3 fields, removed `noise_mult`
- `config/cifar10_noise_game.yaml` — +3 fields, removed `noise_mult`
- `algorithms/noise_game/mechanism.py` — rewritten: 4-layer pipeline
- `algorithms/noise_game/simulator.py` — Phase 2/4/5 updates
- `run.py` — pass DP params to mechanism

### Files unchanged
- `algorithms/noise_game/node.py` — all buffers already present
- `algorithms/noise_game/simple_avg_aggregator.py` — no change needed
