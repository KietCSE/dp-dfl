# Noise Game — Implementation Report

## 1. Tổng quan

Noise Game là cơ chế adaptive noise control cho DP-DFL, kết hợp:

- **Differential Privacy (DP):** bảo vệ gradient với clipping + Gaussian noise.
- **Renyi DP (RDP):** điều phối ngân sách privacy theo thời gian.
- **Strategic noise:** tăng robust trước model poisoning.
- **Recovery:** lấy lại tín hiệu sau khi thêm noise.

Toàn bộ pipeline triển khai trong:

| File | Vai trò |
|---|---|
| [`mechanism.py`](mechanism.py) | Layer 1–3: noise generation + budget cap |
| [`simulator.py`](simulator.py) | Orchestration + privacy accounting + per-round commit |
| [`node.py`](node.py) | Layer 4 buffers (EMA, momentum, control variate, two-track) |
| [`simple_avg_aggregator.py`](simple_avg_aggregator.py) | Aggregator (không sửa cho noise game) |

Pipeline tổng hợp:

```math
g^{final}_t = \mathcal{R}\!\left(g_t + n_{DP,t} + n_{strat,t}\right)
```

trong đó $g_t$ là gradient sau clipping, $n_{DP,t}$ là Gaussian DP noise, $n_{strat,t}$ là strategic noise, và $\mathcal{R}$ là recovery (EMA + momentum + cosine filter + trust scaling).

---

## 2. Kiến trúc 4 lớp

### Layer 1 — Differential Privacy

| Bước | Công thức | Vị trí |
|---|---|---|
| L2 clipping | $g_t \leftarrow g_t / \max(1, \\|g_t\\|/C)$ | [simulator.py:55–58](simulator.py#L55-L58) |
| Gaussian DP noise | $n_{DP,t} \sim \mathcal{N}(0, \sigma_{DP,t}^2 I)$ | [mechanism.py:91–94](mechanism.py#L91-L94) |
| Sigma floor | $\sigma_{DP,t} \ge \dfrac{C\sqrt{2\ln(1.25/\delta)}}{\varepsilon_{\max}}$ | [mechanism.py:45–47](mechanism.py#L45-L47) |

Floor đảm bảo guarantee $(\varepsilon_{\max}, \delta)$-DP độc lập với scheduler.

### Layer 2 — RDP-based Adaptive Control

| Bước | Công thức | Vị trí |
|---|---|---|
| RDP per-round cost | $\Delta\text{RDP}_t = \alpha \cdot C^2 / (2 \sigma_{DP,t}^2)$ | [mechanism.py:21–32](mechanism.py#L21-L32) (`commit_round_rdp`) |
| RDP→DP (Mironov 2017) | $\varepsilon(\alpha,\delta) = \text{RDP}_{\text{total}}(\alpha) + \dfrac{\ln(1/\delta)}{\alpha-1}$ | [mechanism.py:62–72](mechanism.py#L62-L72) |
| Adaptive scheduler | $\sigma_{DP,t} = \max\!\left(\sigma_0 e^{-\kappa t} \cdot \beta_{\text{budget}} \cdot (1+\text{attack}),\; \sigma_{\text{floor}}\right)$ | [mechanism.py:74–87](mechanism.py#L74-L87) |
| Strategic coupling | $\sigma_{\text{strat},t} = \beta_{\text{strat}} \cdot \sigma_{DP,t}$ | [mechanism.py:194](mechanism.py#L194) |

Trong đó:
- $\beta_{\text{budget}} = \max(0.05,\ (\varepsilon_{\max}-\varepsilon_{\text{spent}})/\varepsilon_{\max})$
- $\text{attack} = 1 - \text{trust}$, với $\text{trust} = \cos(g_t, g_{t-1})$

### Layer 3 — Strategic Noise

```math
\hat n_t = \frac{n_{\text{attack}} + n_{\text{orth}} + n_{\text{spec}}}{\\|n_{\text{attack}} + n_{\text{orth}} + n_{\text{spec}}\\|}
\qquad
n_{\text{strat},t} = \sigma_{\text{strat},t} \cdot \hat n_t
```

| Thành phần | Công thức | Vị trí | Mục đích |
|---|---|---|---|
| Directional | $n_{\text{attack}} = \alpha (1-\text{trust})(g_t - g_{t-1})$ | [mechanism.py:111–115](mechanism.py#L111-L115) | Phòng thủ theo hướng attack |
| Orthogonal | $n_{\text{orth}} = z - \dfrac{z\cdot g_t}{\\|g_t\\|^2} g_t$ | [mechanism.py:117–125](mechanism.py#L117-L125) | Tăng entropy không phá descent |
| Reshape-decomposed (a.k.a. "spectrum") | xem chi tiết §2.1 dưới | [mechanism.py:132–158](mechanism.py#L132-L158) | Cấu trúc noise theo phân rã low-rank của 2D-reshape gradient |

#### 2.1 Reshape-decomposed noise (chi tiết)

**Pipeline thực tế trong [mechanism.py:132–158](mechanism.py#L132-L158)**:

```math
\begin{aligned}
& \text{1. Pad: } \tilde g = [g_t \;\|\; \mathbf{0}_{kc - D}] \in \mathbb{R}^{kc}, \quad k=\min(K, D),\; c=\lceil D/k \rceil \\
& \text{2. Reshape: } M = \mathrm{reshape}(\tilde g, \; (k, c)) \in \mathbb{R}^{k \times c} \\
& \text{3. Truncated SVD: } M \approx U \Sigma V^\top, \quad U \in \mathbb{R}^{k \times r},\; V \in \mathbb{R}^{c \times r},\; r=\min(R, k, c) \\
& \text{4. Regularized inverse: } w = (\Sigma + \epsilon_{\text{reg}})^{-1}, \quad \epsilon_{\text{reg}} = 10^{-8} \\
& \text{5. Random projection: } \rho \sim \mathcal{N}(0, \sigma^2 I_r) \\
& \text{6. Reconstruct \& truncate: } n_{\text{spec}} = \mathrm{flatten}\big(U \cdot \mathrm{diag}(w \odot \rho) \cdot V^\top\big)[:D]
\end{aligned}
```

**Tham số code**: $K$ = `svd_reshape_k` (mặc định 64), $R$ = `svd_rank` (mặc định 16).

**Caveat — semantic của "spectrum" ở đây**:

- $U$, $\Sigma$, $V$ là singular decomposition của **2D reshape nhân tạo** của gradient vector, **KHÔNG PHẢI** spectrum của model weights theo từng layer. Reshape là `g[i·c + j] → M[i,j]` — tuần tự, KHÔNG theo layer boundaries → params từ nhiều layers (weights, biases, BN) bị trộn vào cùng row/col mà không có ý nghĩa cấu trúc.
- Tên gọi "spectrum-aware" có thể gây hiểu nhầm; honest naming là **"reshape-decomposed noise"** hoặc **"low-rank-shaped structured noise"**.
- Hyperparam $K=64$ chọn empirical; thay đổi $K$ → matrix shape khác → SVD khác → noise pattern khác hẳn (KHÔNG có theoretical justification chọn 64).

**Tính chất giữ được**:

- $\Sigma_{ii} \ge 0$ → $w_i = 1/(\Sigma_{ii} + 10^{-8}) \in (0, 10^8]$, không Inf/NaN.
- Singular value càng nhỏ ($\Sigma_{ii} \to 0$) → weight $w_i$ càng lớn → noise tập trung vào "spectral mode yếu" của reshape matrix (đúng claim của doc gốc, nhưng "mode" ở đây là mode của reshape, không phải của model).
- Output $n_{\text{spec}} \in \mathbb{R}^D$ đảm bảo cùng dim với gradient.

**Edge case**: nếu `torch.svd_lowrank` raise `RuntimeError` (matrix degenerate) → fallback `n_spec = randn_like(g) * sigma` ([mechanism.py:148–149](mechanism.py#L148-L149)).

### Layer 4 — Recovery

| Bước | Công thức | Vị trí |
|---|---|---|
| EMA smoothing | $\tilde g_t = \gamma \tilde g_{t-1} + (1-\gamma)\hat g_t$ | [node.py:71–75](node.py#L71-L75) |
| Momentum | $m_t = \beta m_{t-1} + (1-\beta)\tilde g_t$ | [node.py:63–67](node.py#L63-L67) |
| Cosine filter | reject nếu $\cos(\hat g_t, \tilde g_{t-1}) < \tau$ | [node.py:79–89](node.py#L79-L89) |
| Trust scaling | $g^{\text{final}}_t = \max(\text{trust}, 0) \cdot m_t$ | [simulator.py:99](simulator.py#L99) |
| Two-track (option) | $w = \lambda w^{\text{clean}} + (1-\lambda) w^{\text{robust}}$ | [node.py:93–105](node.py#L93-L105) |
| SCAFFOLD (option) | $g' = g - c_i + c_{\text{global}}$ | [node.py:52–59](node.py#L52-L59) |

---

## 3. Energy Budget & NSR

### Energy cap (per-dimension semantics)

```math
\\|n_{DP}\\|^2 + \\|n_{\text{strat}}\\|^2 \le \sigma_{\text{total}}^2 \cdot D
```

- **Quan trọng:** `sigma_total` = **per-dimension std cap**, không phải L2-norm tuyệt đối. Cap thực tế = $\sigma_{\text{total}}^2 \cdot D$.
- Khi vượt: cả `n_DP` và `n_strat` được rescale tỷ lệ về cap ([mechanism.py:160–176](mechanism.py#L160-L176)).
- Aligned với `sigma_0` (cũng là per-dim std).

### NSR monitoring

```math
\text{NSR} = \frac{\\|n_{DP} + n_{\text{strat}}\\|}{\\|g\\|}
```

Cảnh báo khi `NSR > nsr_warn` ([simulator.py:85–87](simulator.py#L85-L87)). Round logs lưu `avg_nsr`, `avg_n_dp_norm_postcap`, `avg_sigma_dp_precap` để theo dõi xu hướng.

---

## 4. Privacy Accounting

### Hai counter song song

Hệ dùng **2 RDP counter độc lập** với mục đích khác nhau:

| Counter | Vai trò | Cập nhật | Vị trí |
|---|---|---|---|
| **Internal** (`mechanism.rdp_spent`) | Heuristic cho adaptive scheduler quyết định `σ_DP,t` | 1 lần/round qua `commit_round_rdp(avg_σ_dp)` | [mechanism.py:21–32](mechanism.py#L21-L32), [simulator.py:115–120](simulator.py#L115-L120) |
| **Outer** (`RenyiAccountant`) | Privacy auditing chính thức (Opacus SGM tight bound) | 1 lần/round qua `accountant.step()` | [renyi_accountant.py](../../core/renyi_accountant.py), [simulator.py:152–167](simulator.py#L152-L167) |

### Outer accountant dùng POST-cap σ

```python
avg_n_dp_norm = mean([e["n_dp_norm"] for e in extra_node.values()])  # POST-cap
sigma_eff      = avg_n_dp_norm / sqrt(param_dim)
effective_mult = max(sigma_eff / C, 0.01)
accountant.step(honest_steps, q_batch, effective_mult)
```

Không dùng pre-cap `σ_DP` từ scheduler vì cap có thể rescale `n_dp` → privacy thực phụ thuộc vào noise sau cap, không phải scheduler output.

---

## 5. Bug Fixes Timeline

### Bug #2 — RDP→DP conversion (đã fix)

**Trước:** scheduler so sánh `rdp_spent` trực tiếp với `epsilon_max` (sai đơn vị).
**Sau:** dùng Mironov 2017 Theorem 8: $\varepsilon = \text{RDP}_\alpha + \ln(1/\delta)/(\alpha-1)$ trước khi compare.
**Ref:** [mechanism.py:62–72](mechanism.py#L62-L72)

### Bug #3 — Post-cap σ cho accounting (đã fix)

**Trước:** outer accountant nhận pre-cap `σ_DP` → khi cap kích hoạt, privacy bị under-report ~10000×.
**Sau:** dùng `avg(n_dp_norm)/√D` (post-cap σ_eff) làm input cho accountant.
**Ref:** [simulator.py:146–167](simulator.py#L146-L167)

### Bug #4 — Energy cap scale theo √D (đã fix)

**Trước:** `cap = sigma_total²` (L2-norm tuyệt đối) → với D lớn, cap luôn nuốt noise → `σ_eff = σ_total/√D` rất nhỏ → privacy vỡ.

**Sau:** `cap = sigma_total² · D` (per-dim semantics), đơn vị thống nhất với `sigma_0`.

**Ref:** [mechanism.py:160–176](mechanism.py#L160-L176), [run.py:107–124](../../run.py#L107-L124) (truyền `param_dim` xuống mechanism).

**Tác động:**
| Trước fix (D=80k, σ_0=1, σ_total=2) | Sau fix |
|---|---|
| σ_eff ≈ 0.006 (cap absolute) | σ_eff ≈ σ_DP (cap inactive) |
| z bị clamp 0.01 | z = σ/C (đúng) |
| ε round 1 ≈ hàng nghìn | ε round 1 ≈ 1–3 |

### Bug #5 — Per-round RDP commit (đã fix)

**Trước:** `mechanism.rdp_spent` được tăng MỖI lần `compute_total_noise` chạy → 40 lần/round (40 honest nodes) → scheduler crash budget ở round 3.

**Sau:** `compute_total_noise` không tự update `rdp_spent`. Simulator gọi `mechanism.commit_round_rdp(avg_σ_dp)` MỘT lần/round sau Phase 2 loop, đồng pace với outer accountant.

**Ref:** [mechanism.py:21–32](mechanism.py#L21-L32), [simulator.py:115–120](simulator.py#L115-L120)

**Tác động:**
| Trước fix (50 nodes, ε_max=50) | Sau fix |
|---|---|
| Round 1: ε=2.65, Round 2: ε=4.12, Round 3: ε=158 (spike → break) | Round 1–5: ε linear 2.65→4.24, run đủ 50 round |
| σ_DP crash xuống floor mid-round 3 | σ_DP smooth decay theo `e^{-κt}` |

### Bug #6 — Client sampling_rate compose với batch q (đã fix)

**Trước:** `accountant.step` chỉ nhận `q_batch = batch_size / n_samples`. Client-level Poisson sub-sampling (`config.dp.sampling_rate`) chỉ dùng để chọn active nodes, không đi vào accountant → **bỏ lỡ privacy amplification by sub-sampling**.

**Sau:** compose theo item-level DP independent Poisson layers:
```math
q_{\text{composed}} = q_{\text{client}} \cdot q_{\text{batch}}
```
truyền vào `accountant.step(steps, q_composed, ...)`.

**Ref:** [simulator.py:154–162](simulator.py#L154-L162)

**Tác động** (MNIST, sampling_rate=0.5, q_batch=0.053 → q_composed=0.0265):
| Cấu hình | q tới accountant | ε round 4 |
|---|---|---|
| Trước fix | 0.053 | ~19 (như sampling_rate=1.0) |
| Sau fix | 0.0265 | ~4–6 (~4× amplification) |

**Threat model:** giả định **item-level DP** (mỗi data record là 1 unit). Nếu cần client-level DP, q chỉ là `q_client` (không nhân q_batch) — refactor riêng, không thuộc fix này.

---

## 6. Parameter Specification & Tuning Guide

### 6.1 Bảng tham số

| Tham số | Vai trò | Tăng | Giảm | Mặc định MNIST |
|---|---|---|---|---|
| `C` (clip_bound) | Sensitivity DP | Ít bias, robust | Mất thông tin | 1.0 |
| `epsilon_max` | Privacy strength | Noise ↓, acc ↑ | Noise ↑, privacy ↑ | 50 |
| `delta` | Privacy slack | Noise ↓ chút | Noise ↑ chút | 1e-5 |
| `sigma_0` | Noise base ban đầu | Robust ↑ | Acc ↑ | 1.0 |
| `sigma_total` | Per-dim cap noise | Linh hoạt | Dễ clamp | 0.5 |
| `anneal_kappa` | Decay rate | Convergence nhanh | Noise giữ lâu | 0.01 |
| `beta_strat` | Strat/DP ratio | Robust ↑ | DP dominate | 0.3 |
| `alpha_attack` | Attack noise | Robust ↑ | Ít bảo vệ | 0.1 |
| `nsr_warn` | NSR threshold log | Ít cảnh báo | Cảnh báo nhiều | 200 |
| `ema_gamma` | EMA smoothing | Mượt, ổn định | Nhạy, dao động | 0.9 |
| `momentum_beta` | Momentum | Ổn định | Dao động | 0.9 |
| `align_tau` | Cosine filter | Reject nhiều | Nhận nhiều | 0.2 |

### 6.2 Quan hệ NSR — privacy — D

Với `D = param_dim` chiều, công thức quan trọng:

```math
\text{NSR}_{\text{design}} = \frac{\sigma_{\text{total}} \cdot \sqrt{D}}{C}
\qquad
\sigma_{\text{floor}} = \frac{C\sqrt{2\ln(1.25/\delta)}}{\varepsilon_{\max}}
```

Để `σ_floor < σ_total` (floor không nuốt scheduler):

```math
\varepsilon_{\max} \ge \frac{\sqrt D \cdot \sqrt{2\ln(1.25/\delta)}}{\text{NSR}_{\text{design}}}
```

**Bảng tra cho MNIST MLP** (D=79510, C=1, δ=1e-5):

| NSR_design target | sigma_total | epsilon_max tối thiểu |
|---|---|---|
| 100 | 0.354 | 14 |
| 141 (≈√D·z=0.5) | 0.5 | 10 |
| 200 | 0.71 | 7 |
| 282 (z=1.0) | 1.0 | 5 |

**Lưu ý quan trọng:** `noise_mult z = σ/C ≥ 0.5` để Opacus SGM bound không nổ. Suy ra `NSR_design ≥ 0.5 · √D / 1 ≈ 141` cho MNIST. **Không thể tới NSR=15 với D lớn** mà vẫn giữ privacy có ý nghĩa.

### 6.3 Sample configs

**MNIST IID (50 nodes, balanced):**
```yaml
training: { lr: 0.01, batch_size: 64 }
dp: { clip_bound: 1.0, delta: 1e-5, epsilon_max: 50 }
noise_game:
  sigma_0: 1.0
  sigma_total: 0.5      # NSR_design ≈ 141
  anneal_kappa: 0.01
  beta_strat: 0.3
  nsr_warn: 200
```
Kỳ vọng: ε cuối ≈ 30–50, acc ≥ 0.6 sau 50 round.

**CIFAR-10 (medium complexity):**
```yaml
dp: { clip_bound: 2.0, epsilon_max: 30 }
noise_game:
  sigma_0: 1.5
  sigma_total: 1.0
  anneal_kappa: 0.02
  beta_strat: 0.5
```

**DFL adversarial (attack-heavy):**
```yaml
dp: { clip_bound: 1.0, epsilon_max: 20 }
noise_game:
  sigma_0: 2.0
  sigma_total: 1.5
  beta_strat: 0.6
  alpha_attack: 0.4
```

### 6.4 Quy trình tuning

1. **Fix privacy constraint trước:** chọn `epsilon_max`, `delta`, `C` theo yêu cầu nghiệp vụ.
2. **Tính `sigma_total`** theo NSR_design mong muốn (xem bảng 6.2).
3. **Set `sigma_0 ≈ sigma_total · 2`** để scheduler có buffer decay.
4. **Tinh chỉnh `lr`** (thường 5–10× DP-SGD baseline) để gradient grow nhanh, observed NSR rớt.
5. **Tune robust** (`alpha_attack`, `beta_strat`) nếu có attack.
6. **Verify** end-to-end: ε linear theo round, acc tăng đều, NSR observed giảm dần.

---

## 7. Quyết định thiết kế chính

### 7.1 `attack_signal = 1 - trust`

Chọn cosine similarity giữa $g_t$ và $g_{t-1}$ làm proxy cho trust. Đơn giản, trực quan, không cần chỉ báo bổ sung. Khi gradient không ổn định (trust thấp), tăng noise để tăng robustness.

### 7.2 `sigma_total` per-dim, không phải L2-norm tuyệt đối

Quyết định sau Bug #4. Lý do: aligned đơn vị với `sigma_0` (cũng per-dim), tránh cap nuốt noise khi D lớn. Trade-off: phải lưu `param_dim` trong mechanism.

### 7.3 RDP heuristic (internal) tách rời privacy auditing (outer)

Internal counter chỉ phục vụ scheduler — không claim DP guarantee. RenyiAccountant (Opacus SGM) là source of truth cho privacy auditing.

### 7.4 Per-round commit cadence

Quyết định sau Bug #5. Mỗi round chỉ tăng RDP nội bộ 1 lần (dùng `avg(σ_DP)` qua các honest active nodes), match pace với outer accountant. Mặc định node-level DP, không phải group privacy.

### 7.5 Strategic noise normalize-then-scale

Combine 3 thành phần (directional + orthogonal + spectrum) → normalize unit vector → scale bởi `σ_strat`. Điều này tách "hướng noise" khỏi "magnitude noise" — magnitude bị kiểm soát bởi scheduler.

---

## 8. Ý tưởng đã cân nhắc nhưng không dùng

| Ý tưởng | Lý do bỏ |
|---|---|
| Attack signal phức tạp ($\\|g_t - g_{t-1}\\|/\\|g_t\\|$, multi-indicator) | KISS: trust đơn giản đủ tốt |
| Full RDP min-α accounting trong mechanism | Outer accountant đã làm chính xác hơn |
| Per-dimension noise cap | Không cần thiết, fixed global budget đủ giám sát |
| RDP nhân với num_honest_nodes (group privacy) | Sai paradigm — node-level DP đủ |
| True per-layer SVD cho spectrum noise (Case A) | Phá existing experiments; cần re-tune σ_total + alignment threshold; eps trajectory đổi. Hiện giữ artificial reshape (Case A') để backward-compat — đã document caveat trong §2.1. |

---

## 9. Verification

### 9.1 Unit tests

`tests/test_noise_game_postcap_accounting.py` (16 test cases):

- `TestPostCapAccounting`: cap activation, post-cap σ_eff bounds, pre/post ratio.
- `TestRdpToEpsConversion`: Mironov formula, α≤1 edge case, per-round commit cadence, zero-σ no-op.

### 9.2 End-to-end

```bash
python run.py --algorithm noise-game config/fast-experiment/mnist/noise_game.yaml
```

Healthy run signature:
- ε **linear** theo round (∆ε ≈ 0.4–0.8 per round với MNIST default).
- `avg_sigma_dp_precap` decay smooth theo $e^{-\kappa t}$.
- `avg_n_dp_norm_postcap ≈ σ_DP · √D` (cap inactive trong điều kiện thường).
- Acc tăng đều, không oscillate dưới noise dominant.
- Run hoàn thành đủ `n_rounds`, không break do budget.

---

## 10. Future work

- **Sensitivity analysis:** ablation `sigma_total`, `beta_strat`, `α_attack` trên CIFAR-10 + DFL.
- **Stability proof:** bound formal về convergence rate dưới noise game scheduler.
- **Scale `nsr_warn` theo √D tự động:** hiện đang thủ công per dataset.
- **Tighter SGM bound:** explore PRV accountant cho ε chặt hơn ở round nhỏ.
