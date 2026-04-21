# Mã giả: Adaptive Noise DP Decentralized Federated Learning
## User-Level Clipping + Adaptive Gaussian Noise + Loss-based Ratio (EMA)
## Tổng hợp: **2-Layer Malicious Detection** (Momentum Cosine + Kurtosis) → Simple Averaging trên neighbors sạch

---

## 1. Ký hiệu và Giải thích

| Ký hiệu | Ý nghĩa |
|----------|----------|
| $N$ | Tổng số nodes trong mạng |
| $\mathcal{N}_n$ | Tập hàng xóm của node $n$ |
| $d_n$ | Số hàng xóm của node $n$, $d_n = \|\mathcal{N}_n\|$ |
| $T$ | Tổng số communication rounds |
| $E$ | Số local epochs mỗi node train trong 1 round |
| $\eta$ | Learning rate |
| $B$ | Batch size |
| $C$ | **User-level clipping norm** — ngưỡng cắt TOÀN BỘ model update $\Delta w$ |
| $\sigma_{n,t}$ | **Noise std** của node $n$ tại round $t$ (ADAPTIVE) |
| $\sigma_0$ | Noise std khởi tạo |
| $\sigma_{\min}$ | Floor của noise std (bảo vệ privacy tối thiểu) |
| $\alpha$ | Bậc của Rényi Divergence, $\alpha > 1$ |
| $\delta$ | Target delta trong $(\epsilon, \delta)$-DP |
| $w_{n,t}$ | Model của node $n$ tại round $t$ |
| $\Delta w_{n,t}$ | Model update raw của node $n$ tại round $t$ |
| $\Delta \tilde{w}_{n,t}$ | Model update đã clip + noise của node $n$ |
| $D_n$ | Dữ liệu cục bộ của node $n$ (train/validation split) |
| $D$ | Số chiều của model |
| $\mathrm{Loss}_n^t$ | **Loss cục bộ** của node $n$ tại round $t$ (trên tập train hoặc validation cục bộ) |
| $\overline{\mathrm{Loss}}_n^t$ | EMA của loss tại node $n$ qua các round |
| $r_n^t$ | Adaptive ratio (loss ratio) tại node $n$ tại round $t$, $\in [0, 1]$ |
| $\gamma$ | EMA smoothing factor (khuyến nghị $\gamma = 0.9$) |
| $\beta_{\min}$ | Max decay rate mỗi round (khuyến nghị $0.95$) |
| $\varepsilon$ | Hằng số tránh chia 0 (ví dụ $10^{-8}$) |
| $q$ | **Poisson client subsampling rate** mỗi round (nếu $q<1$: amplification $q^2$) |
| $\beta_m$ | Momentum coefficient cho EMA update (khuyến nghị $0.9$) |
| $\mathbf{m}_{n,j}^t$ | Momentum buffer của neighbor $j$ tại node $n$ sau round $t$ |
| $\cos_{n,j}^t$ | Cosine similarity giữa $\mathbf{m}_{n,j}^t$ và $\mathbf{m}_{n,n}^t$ |
| $\tau_t$ | MAD-adaptive threshold cho cosine (tính mỗi round) |
| $\gamma_{\mathrm{mad}}$ | Hệ số MAD multiplier (khuyến nghị $2.0$) |
| $T_w$ | Warmup rounds trước khi kích hoạt Layer 1 (khuyến nghị $5$) |
| $K(\Delta \tilde{w}_j)$ | Sample Excess Kurtosis của update từ neighbor $j$ |
| $T_k$ | Kurtosis threshold: $T_k = c \cdot \sqrt{24/D}$ |
| $c$ | Confidence multiplier cho kurtosis (khuyến nghị $1.96$ = 95% CI) |
| $\mathcal{M}_n^t$ | Tập neighbor bị đánh dấu malicious bởi node $n$ tại round $t$ |
| $\mathcal{N}'_n$ | Tập neighbor sạch sau filter: $\mathcal{N}_n \setminus \mathcal{M}_n^t$ |

---

## 2. Công thức Chính

### 2.1. User-level Clipping + Gaussian Noise

```
Model update raw:
  Δw_{n,t} = w_{n,t}^{local} − w_{n,t−1}

Clip tại USER-LEVEL (toàn bộ update, KHÔNG per-sample):
  Δŵ_{n,t} = Δw_{n,t} · min(1, C / ‖Δw_{n,t}‖₂)
  → ‖Δŵ_{n,t}‖₂ ≤ C

Thêm Gaussian noise với std σ_{n,t} (ADAPTIVE):
  Δw̃_{n,t} = Δŵ_{n,t} + N(0, σ_{n,t}² · I_D)

Sensitivity (user-level): Δ₂ = C
  (vì thay đổi 1 user ⟹ update khác 1 vector có norm ≤ C)
```

### 2.2. Adaptive Signal từ Loss cục bộ

Mỗi node tự đánh giá mức độ hội tụ của bản thân thông qua **loss** trên
dữ liệu cục bộ (train hoặc validation). KHÔNG cần tín hiệu từ hàng xóm
→ đơn giản hơn, không phụ thuộc vào chất lượng neighbors.

```
Bước 1 — Tính Loss Ratio:
  So sánh loss hiện tại với EMA loss quá khứ để biết mô hình đang
  tiến bộ (loss giảm) hay bão hòa (loss ngang/tăng).

                    ⎛         Loss_n^t          ⎞
    r_n^t  =  min ⎜  1,  ─────────────────────  ⎟
                    ⎝     Loss̄_n^{t−1} + ε      ⎠

  • Loss giảm so với quá khứ  ⟹  r_n^t < 1 (mô hình đang học tốt)
  • Loss bão hòa hoặc tăng    ⟹  r_n^t = 1 (giữ nguyên σ)

Bước 2 — Cập nhật EMA của Loss (làm mượt đường cong):
  Loss̄_n^t  =  γ · Loss̄_n^{t−1}  +  (1 − γ) · Loss_n^t

  (khuyến nghị γ = 0.9)

  Khởi tạo: Loss̄_n^0 ← Loss_n^1 (round đầu tiên)

Bước 3 — Phân rã Nhiễu (Noise Decay):
  σ_{n,t+1}  =  max ⎛ σ_min,  σ_{n,t} · (β_min + (1 − β_min) · r_n^t) ⎞
                    ⎝                                                  ⎠

  (khuyến nghị β_min = 0.95)

  Trường hợp biên:
    r_n^t = 1 (bão hòa)       → σ_{n,t+1} = σ_{n,t}          (không giảm)
    r_n^t → 0 (giảm mạnh)     → σ_{n,t+1} = β_min · σ_{n,t}  (giảm mạnh nhất)

Trường hợp đặc biệt t = 1:
  ÉP r_n^1 = 1 (giữ nguyên σ₀), chưa đủ lịch sử loss để adapt.
```

### 2.3. Rényi DP Accounting (Subsampled Gaussian Mechanism, User-level)

```
Mỗi round, mỗi node được chọn tham gia với xác suất q (Poisson client
subsampling). Sensitivity per round = C (user-level). Mỗi node tự tính
accountant riêng.

─────────────────────────────────────────────────────────────────────
BASE: RDP của Gaussian Mechanism (không subsampling)
─────────────────────────────────────────────────────────────────────

                         α · C²
  ε_step,full(α)  =   ─────────────
                        2 · σ²

─────────────────────────────────────────────────────────────────────
SUBSAMPLED: Mironov-Talwar-Zhang 2019 approximation (q nhỏ)
─────────────────────────────────────────────────────────────────────

Với Poisson subsampling rate q:

                            q² · α · C²
  ε_step,sub(α)  =        ─────────────────       (q < 1)
                              2 · σ²

  • q = 1   (no subsampling) → ε_step = α·C²/(2σ²)
  • q < 1   → amplification q² (giảm cost ~100× nếu q=0.1)

Ghi chú:
  • Công thức q² là UPPER BOUND (approximation), chặt khi q nhỏ và
    α·q < 1. Các thư viện chặt hơn (Opacus, prv_accountant) dùng exact
    SGM-RDP via numerical log-MGF.
  • Approximation này khớp với code (per_node_rdp_accountant.py:45):
        amp = q² if q < 1.0 else 1.0

─────────────────────────────────────────────────────────────────────
ADAPTIVE σ: RDP của node n tại round t
─────────────────────────────────────────────────────────────────────

                            q² · α · C²
  ε_round,n,t(α)  =       ─────────────────
                            2 · (σ_{n,t})²

  (khi q = 1 thì q² biến mất; code tự xử lý trường hợp q ≥ 1)

─────────────────────────────────────────────────────────────────────
COMPOSITION sau T rounds (additive per-client):
─────────────────────────────────────────────────────────────────────

                       T
  ε_total,n(α)  =   Σ   ε_round,n,t(α)
                      t=1

                          T      q² · α · C²
                 =     Σ    ───────────────────
                        t=1   2 · (σ_{n,t})²

                          q² · α · C²        T          1
                 =      ──────────────  ·  Σ   ─────────────
                              2             t=1  (σ_{n,t})²
```

### 2.4. Chuyển RDP → $(\epsilon, \delta)$-DP

```
Per-client ε:
                                        ln(1/δ)
  ε_n  =  min    { ε_total,n(α) + ─────────── }
         α > 1                          α − 1

System-wide worst-case:
  ε_system  =  max   ε_n
              n ∈ {1,...,N}

Distribution statistics:
  ε_avg = (1/N) · Σ_n  ε_n
  ε_std = std({ε_n})
```

---

## 2.5. Tổng Hợp với 2-Layer Malicious Detection

Adaptive noise giúp tối ưu privacy-utility trade-off nhưng **không chống Byzantine
attacks**. Simple averaging trên toàn bộ hàng xóm dễ bị đầu độc nếu có node
malicious. Ta bổ sung **2 layer detection trực giao** trước khi aggregate:

### 2.5.1. Triết lý Hai Layer

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — Momentum Cosine (TEMPORAL/DIRECTION-based)           │
│  Paper: Karimireddy, He, Jaggi — ICML 2021                      │
│  "Learning from History for Byzantine Robust Optimization"      │
│                                                                  │
│  → Bắt: label flipping (kể cả stealthy), targeted poisoning,    │
│          backdoor, mọi tấn công giữ nguyên direction qua rounds │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 — Kurtosis (DISTRIBUTION/SHAPE-based)                  │
│  Paper: dpfl project — Sample Excess Kurtosis Detection         │
│                                                                  │
│  → Bắt: scale attack, ALIE (A Little Is Enough), gaussian       │
│          random, mọi tấn công làm méo phân phối Gaussian       │
└─────────────────────────────────────────────────────────────────┘
```

**Tại sao trực giao?**

| Attack type | Statistical shape | Direction persistence |
|-------------|-------------------|----------------------|
| Scale ×A | Non-Gaussian (méo tail) → L2 bắt | — |
| ALIE (shift trong σ-band) | Non-Gaussian signature → L2 bắt | — |
| Gaussian random | Non-Gaussian nếu norm sai → L2 bắt | — |
| Label flip (naive) | Có thể méo → L2 bắt | Direction ngược → L1 bắt |
| **Stealthy label flip** | Gaussian (giả clip+noise) → **L2 fail** | Direction ngược persistent → **L1 bắt** |

### 2.5.2. Layer 1 — Momentum Cosine Detection

**Insight**: DP noise có $\mathbb{E}[\mathrm{noise}] = 0$. EMA qua nhiều rounds
triệt tiêu nhiễu ($\mathrm{Var}[\mathrm{noise}_{\mathrm{EMA}}] \approx \sigma^2 \cdot \frac{1-\beta_m}{1+\beta_m}$).
Attack signal (label flip) **persistent cùng một hướng sai** → preserve qua EMA.

SNR boost $\sim 3\text{-}4\times$ sau 10 rounds với $\beta_m = 0.9$ → direction
sai của attacker **hiện nguyên hình** dù đã ngụy trang bằng clip+noise.

```
Momentum buffer (per neighbor j ∈ N_n ∪ {n}):
  m_{n,j}^t  =  β_m · m_{n,j}^{t−1}  +  (1 − β_m) · Δw̃_{j,t}

Cosine similarity trên momentum (direction indicator):
                    ⟨ m_{n,j}^t ,  m_{n,n}^t ⟩
  cos_{n,j}^t  =  ─────────────────────────────────
                   ‖m_{n,j}^t‖ · ‖m_{n,n}^t‖ + ε

MAD-adaptive threshold (robust với non-IID và DP noise):
  μ_cos    =  median({cos_{n,j}^t : j ∈ N_n})
  σ_cos    =  1.4826 · MAD({cos_{n,j}^t : j ∈ N_n})
  τ_t      =  μ_cos − γ_mad · σ_cos                 (γ_mad = 2.0)

Flag Layer 1:
  F1_{n,j}^t  =  ( cos_{n,j}^t  <  τ_t )

Warmup (momentum chưa hội tụ):
  IF t < T_w THEN F1_{n,j}^t ← False
```

### 2.5.3. Layer 2 — Kurtosis Detection

**Insight**: Update honest sau clip + Gaussian noise xấp xỉ Gaussian → Sample
Excess Kurtosis $\approx 0$. Attacker làm méo phân phối (scale, ALIE, random)
để lại dấu vết kurtosis lệch khỏi 0.

```
Sample Excess Kurtosis (uncentered, RMS-normalized):
                      1     D   ⎛ Δw̃_j[i]  ⎞⁴
  K(Δw̃_j)  =   ─────   Σ  ⎜ ──────────── ⎟   −  3
                      D   i=1  ⎝ RMS(Δw̃_j) ⎠

  trong đó:  RMS(Δw̃_j)  =  √( (1/D) · Σ_i Δw̃_j[i]² )

Ngưỡng (dựa trên phân phối kurtosis sample của Gaussian):
  T_k  =  c · √(24/D)     (c = 1.96 → 95% CI)

Flag Layer 2:
  F2_{n,j}^t  =  ( |K(Δw̃_{j,t})|  >  T_k )
```

### 2.5.4. Composition: AND-of-Clean

Neighbor chỉ được accept nếu **qua cả 2 layer**:

```
accept_{n,j}^t  =  ¬ F1_{n,j}^t  ∧  ¬ F2_{n,j}^t

Tập malicious của node n tại round t:
  M_n^t  =  { j ∈ N_n  :  F1_{n,j}^t ∨ F2_{n,j}^t }

Tập neighbor sạch:
  N'_n  =  N_n \ M_n^t,       d'_n = |N'_n|
```

**Tại sao AND-of-clean?** Hai layer bắt 2 họ tấn công trực giao. Attacker muốn
qua cả 2 phải **vừa Gaussian-looking vừa cùng-hướng-peers** → gần như honest
→ không còn là attack hiệu quả nữa.

### 2.5.5. Aggregation chỉ trên Neighbors Sạch

```
IF d'_n > 0 THEN:
                                1
  w_{n,t}  =  w_{n,t−1}  +  ────────── · ⎛ Δw̃_{n,t}  +  Σ  Δw̃_{j,t} ⎞
                             d'_n + 1    ⎝               j∈N'_n      ⎠
ELSE:
  w_{n,t}  =  w_{n,t−1}  +  Δw̃_{n,t}     // chỉ dùng local
END IF
```

---

## 3. Mã Giả

```
═══════════════════════════════════════════════════════════════
  THUẬT TOÁN: Adaptive DP Decentralized FedAvg
  Cơ chế: User-level Clipping + Adaptive Gaussian Noise + Rényi DP
  Adaptive Signal: Loss-based Ratio (EMA của Loss cục bộ)
═══════════════════════════════════════════════════════════════

INPUT:
  N           ← số nodes
  T           ← số communication rounds tối đa
  E           ← số local epochs
  η           ← learning rate
  B           ← batch size (cho local training, không liên quan DP)
  C           ← user-level clipping norm
  q           ← Poisson client subsampling rate (1.0 = no subsampling)
  σ₀          ← noise std khởi tạo
  σ_min       ← floor của noise std
  γ           ← EMA smoothing factor cho Loss (0.9)
  β_min       ← max decay rate mỗi round (0.95)
  ε           ← numerical constant tránh chia 0 (1e-8)
  δ           ← target delta
  α_list      ← danh sách bậc Rényi khảo sát
                 (vd: [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100])
  ε_max       ← privacy budget tối đa per-client

  // --- 2-Layer Malicious Detection ---
  β_m         ← momentum coefficient (khuyến nghị 0.9)
  γ_mad       ← MAD multiplier cho cosine threshold (khuyến nghị 2.0)
  T_w         ← warmup rounds trước khi kích hoạt L1 (khuyến nghị 5)
  c           ← confidence multiplier cho kurtosis (khuyến nghị 1.96)
  D           ← số chiều model (dùng cho T_k = c·√(24/D))

OUTPUT:
  {w_{n,T}}   ← model của mỗi node sau huấn luyện
  {ε_n}       ← privacy cost của từng node
  ε_system    ← max privacy cost toàn hệ thống


═══════════════════════════════════════════════════════════════
  KHỞI TẠO (mỗi node n thực hiện độc lập)
═══════════════════════════════════════════════════════════════

1.  Tất cả nodes khởi tạo cùng model: w_{n,0} ← w₀  (∀n)

2.  Mỗi node n khởi tạo RDP accountant:
      FOR mỗi α ∈ α_list:
        ε_rdp,n[α] ← 0

3.  Khởi tạo noise std:
      σ_{n,0} ← σ₀

4.  Khởi tạo EMA loss state:
      Loss̄_n_prev ← None       // sẽ set ở round 1

4b. Khởi tạo momentum buffers cho 2-layer detection:
      FOR mỗi j ∈ N_n ∪ {n}:
        m_{n,j} ← 0_D             // zero vector cùng shape với Δw

4c. Tính kurtosis threshold (cố định theo D):
      T_k ← c · √(24 / D)


═══════════════════════════════════════════════════════════════
  VÒNG LẶP CHÍNH — MỖI NODE n CHẠY SONG SONG
═══════════════════════════════════════════════════════════════

5.  FOR t = 1 TO T DO:

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 0: POISSON CLIENT SUBSAMPLING             │
    │                                                  │
    │  Mỗi node honest (chưa frozen) tham gia round    │
    │  với xác suất q. Attacker luôn active.          │
    │  → amplification q² vào RDP cost (BƯỚC 8).      │
    ├─────────────────────────────────────────────────┤
5a. │  active_ids ← ∅                                  │
5b. │  FOR mỗi node n ∈ {1,...,N} DO:                 │
    │    IF node n frozen (ε_n > ε_max): continue     │
    │    IF node n là attacker: active_ids ← ∪ {n}    │
    │    ELIF q ≥ 1.0 OR random() < q:                 │
    │      active_ids ← active_ids ∪ {n}               │
    │  END FOR                                         │
    │                                                  │
    │  // Các bước dưới chỉ áp dụng cho n ∈ active_ids │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 1: LOCAL TRAINING (không DP tại đây)      │
    ├─────────────────────────────────────────────────┤
6.  │  w_temp ← w_{n,t−1}                             │
    │                                                  │
7.  │  FOR epoch e = 1 TO E DO:                        │
8.  │    Shuffle D_n ngẫu nhiên                        │
    │                                                  │
9.  │    FOR mỗi mini-batch b ⊂ D_n (size B) DO:      │
10. │      g ← ∇L(w_temp; b)                          │
11. │      w_temp ← w_temp − η · g                    │
12. │    END FOR                                       │
13. │  END FOR                                         │
    │                                                  │
    │  // Model update raw (CHƯA có DP)                │
14. │  Δw_{n,t} ← w_temp − w_{n,t−1}                  │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 2: TÍNH LOSS CỤC BỘ                        │
    │  (trên tập train hoặc validation cục bộ)         │
    ├─────────────────────────────────────────────────┤
15. │  Loss_n^t ← L(w_temp; D_n)                       │
    │             // hoặc L(w_temp; D_n^val) nếu có   │
    │             // dùng chính model vừa train xong  │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 3: USER-LEVEL CLIPPING + GAUSSIAN NOISE   │
    ├─────────────────────────────────────────────────┤
    │  // Clip update xuống norm C                     │
16. │  Δŵ_{n,t} ← Δw_{n,t} · min(1, C / ‖Δw_{n,t}‖₂) │
    │                                                  │
    │  // Thêm Gaussian noise với σ_{n,t} (adaptive)   │
17. │  ζ_{n,t} ← N(0, (σ_{n,t})² · I_D)               │
18. │  Δw̃_{n,t} ← Δŵ_{n,t} + ζ_{n,t}                │
    │                                                  │
    │  // Sensitivity = C, noise std = σ_{n,t}         │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 4: GỬI VÀ NHẬN (P2P COMMUNICATION)        │
    ├─────────────────────────────────────────────────┤
19. │  GỬI Δw̃_{n,t} cho mọi m ∈ N_n                  │
20. │  NHẬN Δw̃_{m,t} từ mọi m ∈ N_n                  │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 4.5a: UPDATE MOMENTUM BUFFERS              │
    │  (bao gồm cả bản thân node n để lấy reference)  │
    ├─────────────────────────────────────────────────┤
20a.│  FOR mỗi j ∈ N_n ∪ {n} DO:                      │
20b.│    m_{n,j} ← β_m · m_{n,j} + (1 − β_m) · Δw̃_{j,t}│
    │  END FOR                                         │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 4.5b: LAYER 1 — MOMENTUM COSINE FLAGS     │
    │  (Karimireddy ICML 2021 — direction defense)    │
    ├─────────────────────────────────────────────────┤
20c.│  FOR mỗi j ∈ N_n DO:                            │
    │                                                  │
    │                       ⟨m_{n,j}, m_{n,n}⟩         │
20d.│    cos_j ← ──────────────────────────────        │
    │             ‖m_{n,j}‖ · ‖m_{n,n}‖ + ε            │
    │                                                  │
20e.│  END FOR                                         │
    │                                                  │
    │  // MAD-adaptive threshold per round             │
20f.│  μ_cos ← median({cos_j : j ∈ N_n})              │
20g.│  σ_cos ← 1.4826 · MAD({cos_j : j ∈ N_n})        │
20h.│  τ_t   ← μ_cos − γ_mad · σ_cos                  │
    │                                                  │
    │  F1 ← ∅                                          │
20i.│  FOR mỗi j ∈ N_n DO:                            │
20j.│    IF (t ≥ T_w) AND (cos_j < τ_t) THEN           │
20k.│      F1 ← F1 ∪ {j}                               │
    │    END IF                                        │
    │  END FOR                                         │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 4.5c: LAYER 2 — KURTOSIS FLAGS            │
    │  (Distribution shape defense)                   │
    ├─────────────────────────────────────────────────┤
    │  F2 ← ∅                                          │
20l.│  FOR mỗi j ∈ N_n DO:                            │
    │                                                  │
    │                 1    D  ⎛Δw̃_{j,t}[i]⎞⁴           │
20m.│    K_j ← ──── Σ ⎜──────────────⎟ − 3            │
    │                 D   i=1 ⎝ RMS(Δw̃_{j,t})⎠         │
    │                                                  │
20n.│    IF |K_j| > T_k THEN                           │
20o.│      F2 ← F2 ∪ {j}                               │
    │    END IF                                        │
20p.│  END FOR                                         │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 5: TỔNG HỢP TRÊN NEIGHBORS SẠCH           │
    │  (AND-of-clean: accept iff pass L1 AND L2)      │
    ├─────────────────────────────────────────────────┤
20q.│  M_n^t ← F1 ∪ F2               // tập malicious │
20r.│  N'_n  ← N_n \ M_n^t           // tập clean     │
20s.│  d'_n  ← |N'_n|                                  │
    │                                                  │
    │  IF d'_n > 0 THEN:                               │
21. │    w_{n,t} ← w_{n,t−1} + (1/(d'_n + 1)) ·        │
    │              (Δw̃_{n,t} + Σ_{j ∈ N'_n} Δw̃_{j,t}) │
    │  ELSE:                                           │
22. │    w_{n,t} ← w_{n,t−1} + Δw̃_{n,t}              │
    │  END IF                                          │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 6: LOSS-BASED ADAPTIVE RATIO + EMA        │
    │                                                  │
    │  QUAN TRỌNG: Dùng Loss̄_n^{t−1} (EMA CŨ) trong  │
    │  mẫu số của r_n^t → tránh self-cancellation.    │
    │  Round t=1: ÉP r_n^1 = 1 (chưa đủ lịch sử).    │
    ├─────────────────────────────────────────────────┤
    │                                                  │
    │  IF t == 1 THEN:                                 │
    │    // Round đầu: khởi tạo EMA, ÉP r = 1          │
23. │    Loss̄_n^t ← Loss_n^t                          │
24. │    r_n^t ← 1     // giữ nguyên σ₀ (no adapt)    │
    │                                                  │
    │  ELSE:                                           │
    │    // Loss ratio (so với EMA quá khứ)            │
    │                         ⎛        Loss_n^t       ⎞│
25. │    r_n^t ← min ⎜  1,  ───────────────────────  ⎟│
    │                         ⎝ Loss̄_n_prev + ε       ⎠│
    │                                                  │
    │    // Update EMA SAU KHI đã dùng Loss̄_n_prev    │
26. │    Loss̄_n^t ← γ · Loss̄_n_prev + (1 − γ) · Loss_n^t│
    │  END IF                                          │
    │                                                  │
    │  // Lưu EMA cho round sau                        │
27. │  Loss̄_n_prev ← Loss̄_n^t                         │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 7: UPDATE NOISE STD CHO ROUND SAU         │
    ├─────────────────────────────────────────────────┤
28. │  decay_factor ← β_min + (1 − β_min) · r_n^t    │
29. │  σ_{n,t+1} ← max(σ_min, σ_{n,t} · decay_factor) │
    │                                                  │
    │  // r_n^t → 1  ⟹  σ_{n,t+1} = σ_{n,t} (giữ)    │
    │  // r_n^t → 0  ⟹  σ_{n,t+1} = β_min · σ_{n,t} │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 8: RÉNYI DP ACCOUNTING (SUBSAMPLED GM)    │
    │                                                  │
    │  DÙNG σ_{n,t} CỦA ROUND HIỆN TẠI                │
    │  (σ_{n,t+1} chỉ áp dụng cho round SAU)          │
    │                                                  │
    │  Apply q² amplification nếu có subsampling.      │
    │  Cost tích lũy cho MỌI non-frozen node (bất kể  │
    │  round này có được sample hay không) — xuất phát │
    │  từ bất định của adversary về client được chọn. │
    ├─────────────────────────────────────────────────┤
    │  // Subsampling amplification factor             │
29a.│  IF q < 1.0 THEN amp ← q²  ELSE amp ← 1.0       │
    │                                                  │
30. │  FOR mỗi α ∈ α_list DO:                         │
    │                                                  │
    │                         amp · α · C²             │
31. │    ε_round ← ─────────────────────               │
    │                    2 · (σ_{n,t})²                │
    │                                                  │
32. │    ε_rdp,n[α] ← ε_rdp,n[α] + ε_round            │
    │                                                  │
33. │  END FOR                                         │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  BƯỚC 9: CHUYỂN RDP → (ε, δ)-DP & KIỂM TRA     │
    ├─────────────────────────────────────────────────┤
    │                     ┌                        ┐   │
    │                     │               ln(1/δ)  │   │
34. │  ε_n  ←  min        │ ε_rdp,n[α] + ───────── │   │
    │         α ∈ α_list  │                α − 1   │   │
    │                     └                        ┘   │
    │                                                  │
35. │  IF ε_n > ε_max THEN:                            │
36. │    STOP training cho node n                      │
37. │    Node n chỉ nhận updates, không gửi           │
38. │  END IF                                          │
    └─────────────────────────────────────────────────┘

39. END FOR    // kết thúc round t


═══════════════════════════════════════════════════════════════
  TỔNG HỢP PRIVACY TOÀN HỆ THỐNG
═══════════════════════════════════════════════════════════════

40. Thu thập {ε_n}_{n=1..N}

41. ε_system ← max_{n ∈ {1,...,N}} ε_n     // worst-case
42. ε_avg    ← (1/N) · Σ_n ε_n             // average
43. ε_std    ← std({ε_n})                  // heterogeneity

44. OUTPUT {w_{n,T}}, {ε_n}, ε_system, ε_avg, ε_std
```

---

## 4. Bảng So Sánh với Non-Adaptive và Hybrid cũ

```
┌──────────────────────┬─────────────────┬──────────────────┬──────────────────────┐
│                      │ Non-Adaptive    │ Hybrid (cũ)      │ Loss-based (mới)     │
├──────────────────────┼─────────────────┼──────────────────┼──────────────────────┤
│ Clipping level       │ User-level (C)  │ User-level (C)   │ User-level (C)       │
│ Noise std            │ σ cố định       │ σ_{n,t} adaptive │ σ_{n,t} adaptive     │
│ Sensitivity          │ C               │ C                │ C                    │
│ Signal source        │ Không có        │ Neighbor disagr. │ Local loss           │
│                      │                 │ + intrinsic norm │ (EMA)                │
│ Phụ thuộc neighbors  │ —               │ Có               │ KHÔNG                │
│ Robust Non-IID       │ —               │ Trung bình       │ Cao (tín hiệu riêng) │
│ Subsampling          │ Tuỳ chọn (q=1)  │ Tuỳ chọn         │ Poisson q (q²-amp)   │
│ Privacy per round    │ q²·α·C²/(2σ²)   │ q²·α·C²/(2σ_t²) │ q²·α·C²/(2σ_t²)      │
│ Total RDP (T rounds) │ T·q²·α·C²/(2σ²) │ Σ q²·α·C²/(2σ_t²)│ Σ q²·α·C²/(2σ_t²)    │
│ Per-client ε         │ Đồng đều        │ Heterogeneous    │ Heterogeneous        │
│ Adaptation cost      │ 0               │ 0 (post-proc.)   │ 0 (loss là local)    │
│ Byzantine robust     │ ❌              │ ❌               │ ✅ (2-layer detect)  │
│ Chống label flip     │ ❌              │ ❌               │ ✅ L1 Momentum       │
│ Chống scale/ALIE     │ ❌              │ ❌               │ ✅ L2 Kurtosis       │
└──────────────────────┴─────────────────┴──────────────────┴──────────────────────┘
```

---

## 5. Ví Dụ Tính Toán

### Thiết lập:
```
N = 20 nodes, D = 79,510
C = 2.0, σ₀ = 2.0, σ_min = 0.8
γ = 0.9, β_min = 0.95
q = 0.1              // Poisson client subsampling → amp = q² = 0.01
T = 100, δ = 10⁻⁵
```

### Kịch bản: Node n hội tụ từ round 30

```
Loss cục bộ (giả định):
  • Round 1-29: Loss giảm mạnh (2.30 → 0.80), mỗi round giảm ~5%
  • Round 30+ : Loss bão hòa quanh 0.75

Round 1 (ÉP r_n^1 = 1, chưa adapt):
  Loss̄_n^1 ← Loss_n^1 = 2.30
  σ_{n,1} = σ₀ = 2.0

Round 2-29 (loss giảm liên tục, r < 1):
  Ví dụ round 10: Loss_n^10 = 1.50, Loss̄_n^9 ≈ 1.60
    r_n^10 = min(1, 1.50 / (1.60 + ε)) ≈ 0.938
    decay_factor = 0.95 + 0.05·0.938 = 0.9969
    σ giảm rất chậm (~0.3%/round) vì loss giảm không nhiều so với EMA

  Ví dụ round 20: Loss_n^20 = 0.90, Loss̄_n^19 ≈ 1.05
    r_n^20 = min(1, 0.90 / 1.05) ≈ 0.857
    decay_factor = 0.95 + 0.05·0.857 = 0.9929

Round 30-100 (loss bão hòa, r ≈ 1):
  Loss_n^t ≈ Loss̄_n^{t−1} ≈ 0.75
  r_n^t ≈ 1.0
  σ_{n,t+1} ≈ σ_{n,t}   (không giảm thêm)

→ Noise chỉ giảm trong giai đoạn mô hình còn học được.
→ Khi bão hòa, noise giữ nguyên → privacy budget không tiêu hao quá mức.
```

### Tính RDP có subsampling (q = 0.1):

```
Round 1 (σ=2.0):
  ε_step,sub(α) = q² · α · C² / (2σ²)
                = 0.01 · α · 4 / 8
                = 0.005 · α

  So với no-subsampling: 0.5 · α → q² giảm cost 100× ở round này.

Round 20 (σ đã giảm nhẹ, giả định σ=1.7):
  ε_step,sub(α) = 0.01 · α · 4 / (2·2.89)
                ≈ 0.00692 · α

Tích lũy sau T=100 rounds (ước lượng):
  ε_total(α) ≈ Σ_{t=1..100} q² · α · C² / (2·σ_t²)
              ≈ 0.01 · α · (T · avg(1/σ_t²)) · C²/2
              ≈ 0.01 · α · 100 · 0.35 · 2         (avg 1/σ² ≈ 0.35)
              ≈ 0.7 · α

Tìm α tối ưu cho (ε, δ)-DP với δ=10⁻⁵, ln(1/δ) ≈ 11.51:
  ε(α) = 0.7·α + 11.51/(α−1)

  dε/dα = 0 → α* ≈ 1 + √(11.51/0.7) ≈ 5.05
  ε* ≈ 0.7·5.05 + 11.51/4.05 ≈ 6.4

→ Với q=1 (no subsampling) kết quả tương đương sẽ là ε ≈ 64, tức
  subsampling GIẢM ε_final ~10× so với không subsample.
```

**Insight quan trọng**:
So với hybrid cũ (dùng neighbor disagreement), Loss-based:
- **Không phụ thuộc** vào chất lượng neighbors → robust trong Non-IID.
- **Phản ánh đúng** trạng thái hội tụ của bản thân node.
- **Đơn giản hơn**: chỉ cần tính loss trên local data (việc vốn dĩ đã làm trong training).
- **Privacy-safe tuyệt đối**: loss được tính trên model cục bộ, KHÔNG dùng
  thông tin noisy từ neighbors → không rò rỉ, không tốn thêm budget.

---

## 6. Ý Nghĩa và Insight

```
Nguyên tắc hoạt động:

  1. Mỗi node n tự đánh giá mức độ hội tụ của bản thân thông qua
     Loss_n^t trên dữ liệu cục bộ (train hoặc validation).

  2. EMA Loss̄_n^t làm reference trend dài hạn, ổn định hơn
     giá trị tức thời (tránh dao động ngẫu nhiên làm hỏng ratio).

  3. Loss ratio r_n^t cho biết:
     • r_n^t < 1 (Loss hiện tại < EMA quá khứ)  → mô hình đang học
                  → GIẢM noise để tăng signal-to-noise
     • r_n^t = 1 (Loss bão hòa hoặc tăng)       → mô hình hết học
                  → GIỮ NGUYÊN noise (không hoang phí budget)

  4. Noise σ_{n,t} giảm dần khi node còn đang hội tụ
     → privacy cost tăng dần (nhưng hợp lý, có trade-off đáng giá)
     → accuracy cải thiện vì signal ít bị nuốt bởi noise

  5. Tín hiệu loss là LOCAL → post-processing của model cục bộ
     → KHÔNG tốn thêm privacy budget cho adaptation
     → KHÔNG phụ thuộc vào neighbors (robust Non-IID)
```

---

## 7. Hyperparameter Defaults

```
┌──────────────┬───────────┬────────────────────────────────────┐
│  Tên         │  Giá trị  │  Ý nghĩa                            │
├──────────────┼───────────┼────────────────────────────────────┤
│  σ₀          │  2.0      │  Initial noise std                  │
│  σ_min       │  0.8–1.5  │  Floor (tuning theo ε_max)          │
│  C           │  2.0      │  User-level clipping norm           │
│  q           │  0.1–1.0  │  Poisson client subsampling rate    │
│  γ           │  0.9      │  EMA smoothing cho Loss             │
│  β_min       │  0.95     │  Max decay/round (5% mỗi round)     │
│  ε           │  10⁻⁸     │  Numerical tránh chia 0             │
│  Loss source │  train/val│  Dữ liệu tính loss (khuyến nghị val)│
├──────────────┼───────────┼────────────────────────────────────┤
│  β_m         │  0.9      │  Momentum EMA coef (L1 cosine)      │
│  γ_mad       │  2.0      │  MAD multiplier → cosine threshold  │
│  T_w         │  5        │  Warmup rounds (momentum hội tụ)    │
│  c           │  1.96     │  Confidence multiplier cho kurtosis │
│  T_k         │  c·√(24/D)│  Kurtosis threshold (auto từ D)     │
│  Combine     │  AND-clean│  Accept iff pass L1 AND L2          │
└──────────────┴───────────┴────────────────────────────────────┘
```

---

## 8. 2-Layer Defense — Insights & Trade-offs

### 8.1. Khai thác Asymmetry: DP Noise vs Attack Signal

```
Nguyên lý:

  DP Gaussian noise:     E[noise] = 0   →  EMA triệt tiêu qua rounds
  Attack direction:      persistent     →  EMA khuếch đại signal

Hệ quả:
  Var[noise_EMA] ≈ σ²_DP · (1 − β_m) / (1 + β_m)
                 ≈ 0.053 · σ²_DP          (β_m = 0.9)

  → Signal-to-Noise Ratio tăng ~4× sau 10 rounds
  → Direction ngược của stealthy label flipper hiện rõ
     dù attacker đã giả clip + noise giống honest
```

### 8.2. Orthogonality — tại sao cần cả 2 Layer

```
Không có defense ĐƠN nào bắt hết. Two-layer vì:

  L1 (Momentum Cosine) thấy:
    ✓ Stealthy label flip (direction sai)
    ✓ Backdoor attack (direction bất nhất qua rounds)
    ✗ Scale attack (direction đúng, magnitude sai)

  L2 (Kurtosis) thấy:
    ✓ Scale × A (tail heavy)
    ✓ ALIE shift (shape méo)
    ✓ Gaussian random (norm/direction sai)
    ✗ Stealthy label flip (Gaussian shape giữ nguyên)

  AND-of-clean = union coverage:
    Attacker phải vừa Gaussian vừa cùng-hướng-peers → honest
```

### 8.3. Expected Recall (từ literature + scope dự án)

```
┌───────────────────────────┬──────────┬──────────────┬──────────────┐
│ Attack                    │ L2-only  │ L1+L2 (AND)  │ FLTrust (ref)│
├───────────────────────────┼──────────┼──────────────┼──────────────┤
│ Scale × A                 │  ~95%    │   ~95%       │   ~90%       │
│ ALIE                      │  ~90%    │   ~90%       │   ~85%       │
│ Label flip (naive)        │   ~0%    │   ~85-90%    │   ~90%       │
│ Stealthy label flip, IID  │   ~0%    │   ~80-85%    │   ~90%       │
│ Stealthy label flip,      │   ~0%    │   ~70-75%    │   ~80%       │
│   Non-IID (α=0.3)         │          │              │              │
└───────────────────────────┴──────────┴──────────────┴──────────────┘

  → L1+L2 gần bằng FLTrust nhưng KHÔNG cần trusted root data
```

### 8.4. Limitations (brutal honesty)

```
• Warmup T_w = 5 rounds: momentum chưa hội tụ → L1 vô hiệu
  Mitigation: L2 vẫn active, catch được scale/ALIE ngay từ round 1

• Non-IID nặng (α < 0.3): honest momentum direction phân tán
  Mitigation: MAD-adaptive threshold (không fixed τ)

• Adaptive time-varying attacker: flip nhãn khác nhau mỗi round
  → momentum attacker không hội tụ, khó detect
  Trade-off: nếu attacker thay đổi direction liên tục, attack
             cũng yếu (không có direction ổn định để poison)

• Majority attacker (f ≥ N/2): median threshold bị độc
  Assumption: codebase giới hạn n_attackers < N/2

• Memory: O(|N_n| · D) momentum buffers per node
  MNIST MLP (D≈79K): ~1.3 MB per node — chấp nhận được
  CIFAR CNN (D≈62K): ~40 MB per node — vẫn tractable
```

### 8.5. Privacy Budget Impact

```
KHÔNG THÊM privacy cost:
  • Momentum tính từ {Δw̃_{j,t}} đã noised → post-processing
  • Cosine, kurtosis cũng là post-processing của updates đã noise
  • Không query trực tiếp data → không tốn ε thêm

→ 2-Layer defense hoàn toàn "free" về privacy budget
  Adaptive noise (Section 2.2) và defense (Section 2.5) độc lập
```

### 8.6. Subsampling Amplification — Lưu ý về Tính Chặt

```
Công thức ε_step,sub ≈ q²·α·C²/(2σ²) là APPROXIMATION (upper bound):

  • Mironov-Talwar-Zhang 2019 chứng minh bound chính xác dùng
    log-MGF binomial — TIGHTER nhưng phức tạp hơn nhiều.
  • Code hiện tại dùng q² approximation (per_node_rdp_accountant.py).
  • Approximation CHẶT khi q·α nhỏ (q<0.2, α<20) → phù hợp setup
    q=0.1, α_max=100 CÓ THỂ lỏng ở α lớn.

Nếu cần tight bound cho paper quality:
  • Dùng Opacus `RDPAccountant` hoặc `prv_accountant`
  • Tight bound thường giảm ε_final thêm ~20-40% so với q²
  • Trade-off: code phức tạp hơn, cần numerical integration

Kết luận: q² approximation ĐỦ cho baseline experiment, cần swap sang
exact SGM-RDP nếu muốn benchmark tight với literature.
```
