# Mã giả: DP-SGD Decentralized Federated Learning
# Sử dụng Per-Sample Clipping + Per-Step Noise + Rényi DP Accounting
# Tổng hợp: Simple Averaging (không dùng Mixing Matrix)
# Quy ước: z là noise multiplier, actual noise std = z · C / B

---

## 1. Ký hiệu và Giải thích

| Ký hiệu | Ý nghĩa |
|----------|----------|
| N | Tổng số nodes trong mạng |
| Nᵢ | Tập hàng xóm của node i |
| dᵢ | Số hàng xóm của node i, dᵢ = \|Nᵢ\| |
| T | Tổng số communication rounds |
| E | Số local epochs mỗi node train trong 1 round |
| η | Learning rate |
| B | Batch size (mini-batch) |
| C | Per-sample clipping norm — ngưỡng cắt MỖI gradient sample |
| z | Noise multiplier — actual noise std = z · C / B |
| α | Bậc của Rényi Divergence, α > 1 |
| δ | Target delta trong (ε, δ)-DP cuối cùng |
| wᵢₜ | Model của node i tại round t |
| Δw̃ᵢ | Model update đã noised của node i (tích lũy qua các step) |
| Dᵢ | Dữ liệu cục bộ của node i, nᵢ = \|Dᵢ\| |
| nᵢ | Số mẫu dữ liệu của node i |
| D | Số chiều của model (tổng params) |
| gₛ | Gradient của sample s |
| g̃ₛ | Gradient đã clip của sample s |
| K(·) | Sample Excess Kurtosis |
| T_k | Ngưỡng phát hiện kurtosis: T_k = c · √(24/D) |
| c | Hệ số confidence (mặc định c = 1.96 cho 95% CI) |
| Mᵢ | Tập hàng xóm bị đánh dấu malicious bởi node i |
| q | Batch sampling rate: q = B / nᵢ |

---

## 2. Công thức DP-SGD với Rényi DP

### 2.1. Sensitivity per step (khác Output Perturbation)

```
Output Perturbation (cũ):
  Clip toàn bộ Δw → sensitivity = C
  Noise: N(0, σ²·I) với σ là input trực tiếp

DP-SGD (mới):
  Clip TỪNG sample gradient gₛ tới norm C
  Average B gradients đã clip → sensitivity = C/B
  Noise: N(0, (z·C/B)²·I) với z là noise multiplier
```

### 2.2. Tại sao sensitivity = C/B

```
Batch gồm B samples: {x₁, x₂, ..., x_B}

Gradient đã clip:    g̃ₛ = gₛ · min(1, C / ‖gₛ‖₂)
                     → ‖g̃ₛ‖₂ ≤ C  (∀s)

Average:             ḡ = (1/B) Σ_{s=1}^{B} g̃ₛ

Adjacent dataset (thay sample x_B bằng x'_B):
                     ḡ' = (1/B)(g̃₁ + ... + g̃_{B-1} + g̃'_B)

Sensitivity:
  ‖ḡ − ḡ'‖₂ = (1/B) ‖g̃_B − g̃'_B‖₂
             ≤ (1/B)(‖g̃_B‖₂ + ‖g̃'_B‖₂)
             ≤ (1/B)(C + C)
             = 2C/B

Tight bound (standard trong literature): Δ₂ = C/B
```

### 2.3. Gaussian Mechanism per step

```
Noised gradient:   ḡ_noised = ḡ + N(0, (z · C/B)² · I_D)

Actual noise std per dimension = z · C/B
Noise norm = z · C/B · √D
```

### 2.4. RDP mỗi step (với Poisson subsampling)

```
Mỗi step, batch B được lấy ngẫu nhiên từ nᵢ samples.
Sampling rate: q = B / nᵢ

Subsampled Gaussian Mechanism RDP:

                   q² · α
ε_step(α) ≈  ─────────────
                 2 · z²

Lưu ý: C và B triệt tiêu nhau:
  sensitivity² / noise² = (C/B)² / (z·C/B)² = 1/z²
→ ε_step chỉ phụ thuộc z và q, KHÔNG phụ thuộc C hay B
```

### 2.5. Composition sau 1 round (nhiều steps)

```
Mỗi round, mỗi node chạy K steps (K = ⌊nᵢ/B⌋ × E)
  (Drop incomplete batch cuối — đảm bảo noise calibration consistent)

ε_round(α) = K · q² · α / (2·z²)
```

### 2.6. Composition sau T rounds

```
Tổng steps = T · K

                    T · K · q² · α
ε_total(α)  =  ───────────────────
                      2 · z²
```

### 2.7. Chuyển RDP → (ε, δ)-DP

```
                                ln(1/δ)
ε_final = min  { ε_total(α) + ───────── }
         α > 1                   α − 1
```

---

## 2.8. Kurtosis-based Malicious Detection

_(Giữ nguyên — không thay đổi so với Output Perturbation)_

### Nguyên lý

Sau khi node trung thực clip per-sample + noise mỗi step, model update tích lũy Δw̃ᵢ bao gồm nhiều lớp noise Gaussian → vẫn xấp xỉ Gaussian → Excess Kurtosis ≈ 0.

Node malicious (Scale Attack) vẫn clip per-sample (stealthy) nhưng **bỏ noise** rồi scale lên → phân phối non-Gaussian → Kurtosis ≠ 0.

### Công thức Excess Kurtosis (Uncentered)

```
                1   D   ⎛ Δw̃[i] ⎞⁴
K(Δw̃) =  ───  Σ   ⎜ ─────── ⎟   − 3
                D  i=1  ⎝ RMS(Δw̃)⎠

              ┌─────────────────────┐
RMS(Δw̃) =    │  (1/D) · Σ Δw̃[i]²  │
              └─────────────────────┘
```

### Ngưỡng phát hiện

```
T_k = c · √(24/D)       c = 1.96 → 95% CI

Quy tắc: |K(Δw̃ⱼ)| > T_k → node j malicious → loại khỏi aggregation
```

---

## 3. Mã giả

```
══════════════════════════════════════════════════════════════
  THUẬT TOÁN: DP-SGD Decentralized FedAvg
  Cơ chế: Per-Sample Clipping + Per-Step Noise + Rényi DP
  Quy ước: z là noise multiplier, noise std = z·C/B
══════════════════════════════════════════════════════════════

INPUT:
  N           ← số nodes
  T           ← số communication rounds
  E           ← số local epochs
  η           ← learning rate
  B           ← batch size
  C           ← per-sample clipping norm
  z           ← noise multiplier (vd: 1.1)
  δ           ← target delta (vd: 10⁻⁵)
  α_list      ← danh sách bậc Rényi khảo sát
                 (vd: [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100])
  ε_max       ← privacy budget tối đa cho phép
  c           ← hệ số confidence cho kurtosis threshold
  D           ← số chiều model (tổng params)

OUTPUT:
  {wᵢ_T}     ← model của mỗi node sau huấn luyện
  ε_final     ← tổng privacy budget đã tiêu


══════════════════════════════════════════════════════════════
  KHỞI TẠO
══════════════════════════════════════════════════════════════

1.  Tất cả nodes khởi tạo cùng model: wᵢ₀ ← w₀  (∀i)

2.  Mỗi node i khởi tạo RDP accountant:
      FOR mỗi α ∈ α_list:
        ε_rdp[α] ← 0

2b. Tính ngưỡng kurtosis:
      T_k ← c · √(24/D)

2c. Tính batch sampling rate:
      q ← B / nᵢ


══════════════════════════════════════════════════════════════
  VÒNG LẶP CHÍNH — MỖI NODE i CHẠY SONG SONG
══════════════════════════════════════════════════════════════

3.  FOR t = 1 TO T DO:

      ┌────────────────────────────────────────────────┐
      │  BƯỚC 1: HUẤN LUYỆN CỤC BỘ VỚI DP-SGD         │
      │                                                 │
      │  Thay vì train rồi clip+noise output,           │
      │  CLIP + NOISE tại MỖI mini-batch step.          │
      ├────────────────────────────────────────────────┤
4.    │  wᵢ_temp ← wᵢₜ₋₁                               │
5.    │  n_steps ← 0                                    │
      │                                                 │
6.    │  FOR epoch e = 1 TO E DO:                        │
7.    │    Shuffle Dᵢ ngẫu nhiên                         │
      │                                                 │
8.    │    FOR mỗi mini-batch b = {x₁,...,x_B} ⊂ Dᵢ DO: │
      │      // Drop incomplete batch (|b| < B)         │
      │      IF |b| < B THEN SKIP                       │
      │                                                 │
      │      // ─── Per-sample gradient ───              │
9.    │      FOR s = 1 TO B DO:                          │
10.   │        gₛ ← ∇L(wᵢ_temp; xₛ)                    │
      │      END FOR                                    │
      │                                                 │
      │      // ─── Per-sample clipping ───              │
      │      // Clip TỪNG gradient tới norm C            │
11.   │      FOR s = 1 TO B DO:                          │
12.   │        g̃ₛ ← gₛ · min(1, C / ‖gₛ‖₂)            │
      │      END FOR                                    │
      │                                                 │
      │      // ─── Average clipped gradients ───        │
      │      // Sensitivity = C/B                        │
13.   │      ḡ ← (1/B) Σ_{s=1}^{B} g̃ₛ                  │
      │                                                 │
      │      // ─── Add noise ───                        │
      │      // σ_step = z · C / B                       │
14.   │      ḡ_noised ← ḡ + N(0, (z·C/B)² · I_D)       │
      │                                                 │
      │      // ─── SGD update ───                       │
15.   │      wᵢ_temp ← wᵢ_temp − η · ḡ_noised          │
16.   │      n_steps ← n_steps + 1                      │
      │                                                 │
17.   │    END FOR                                       │
18.   │  END FOR                                         │
      │                                                 │
      │  // Model update (đã noised bên trong)           │
19.   │  Δw̃ᵢ ← wᵢ_temp − wᵢₜ₋₁                        │
      └────────────────────────────────────────────────┘

      ┌────────────────────────────────────────────────┐
      │  BƯỚC 2: GỬI VÀ NHẬN                            │
      │                                                 │
      │  Gửi cùng một Δw̃ᵢ cho tất cả hàng xóm         │
      │  Δw̃ᵢ đã chứa noise (thêm tại mỗi step)        │
      ├────────────────────────────────────────────────┤
20.   │  GỬI Δw̃ᵢ cho mọi j ∈ Nᵢ                       │
21.   │  NHẬN Δw̃ⱼ từ mọi j ∈ Nᵢ                       │
      └────────────────────────────────────────────────┘

      ┌────────────────────────────────────────────────┐
      │  BƯỚC 3: KURTOSIS DETECTION                     │
      │                                                 │
      │  Giống hệt Output Perturbation —                │
      │  kiểm tra |K(Δw̃ⱼ)| > T_k                       │
      ├────────────────────────────────────────────────┤
      │                                                 │
22.   │  Mᵢ ← ∅                   // tập malicious      │
      │                                                 │
23.   │  FOR mỗi j ∈ Nᵢ DO:                             │
      │                                                 │
      │    // Tính Uncentered Excess Kurtosis            │
      │    //                                            │
      │    //         1   D   ⎛ Δw̃ⱼ[k] ⎞⁴              │
      │    // K_j = ───  Σ   ⎜ ──────── ⎟  − 3          │
      │    //         D  k=1  ⎝ RMS(Δw̃ⱼ)⎠              │
      │                                                 │
24.   │    K_j ← ExcessKurtosis(Δw̃ⱼ)                   │
      │                                                 │
25.   │    IF |K_j| > T_k THEN:                          │
26.   │      Mᵢ ← Mᵢ ∪ {j}     // đánh dấu malicious   │
      │    END IF                                       │
      │                                                 │
27.   │  END FOR                                         │
      │                                                 │
      │  // Tập hàng xóm sạch = hàng xóm \ malicious   │
28.   │  N'ᵢ ← Nᵢ \ Mᵢ                                 │
      │  d'ᵢ ← |N'ᵢ|                                    │
      └────────────────────────────────────────────────┘

      ┌────────────────────────────────────────────────┐
      │  BƯỚC 4: TỔNG HỢP (SIMPLE AVERAGING)            │
      │                                                 │
      │  Chỉ aggregate từ hàng xóm SẠCH (đã lọc)      │
      ├────────────────────────────────────────────────┤
      │                                                 │
      │  IF d'ᵢ > 0 THEN:                               │
      │            1                                    │
29.   │    wᵢₜ ← wᵢₜ₋₁ + Δw̃ᵢ + ──── · Σ_{j∈N'ᵢ} Δw̃ⱼ  │
      │                          d'ᵢ                    │
      │  ELSE:                                          │
      │    wᵢₜ ← wᵢₜ₋₁ + Δw̃ᵢ   // chỉ dùng local     │
      │  END IF                                         │
      └────────────────────────────────────────────────┘

      ┌────────────────────────────────────────────────┐
      │  BƯỚC 5: RÉNYI DP ACCOUNTING (DP-SGD)            │
      │                                                 │
      │  Mỗi step có cost riêng. Tích lũy n_steps       │
      │  steps trong round này.                          │
      │                                                 │
      │  Key: C và B triệt tiêu →                       │
      │       ε_step chỉ phụ thuộc z và q               │
      ├────────────────────────────────────────────────┤
30.   │  FOR mỗi α ∈ α_list DO:                         │
      │                                                 │
      │              n_steps · q² · α                   │
31.   │    ε_round ← ─────────────────                   │
      │                  2 · z²                         │
      │                                                 │
32.   │    ε_rdp[α] ← ε_rdp[α] + ε_round                │
      │                                                 │
33.   │  END FOR                                         │
      └────────────────────────────────────────────────┘

      ┌────────────────────────────────────────────────┐
      │  BƯỚC 6: CHUYỂN RDP → (ε, δ)-DP & KIỂM TRA     │
      ├────────────────────────────────────────────────┤
      │                                                 │
      │                    ┌                    ┐       │
      │                    │            ln(1/δ) │       │
34.   │  ε_final ← min     │ ε_rdp[α] + ─────── │       │
      │          α ∈ α_list │             α − 1  │       │
      │                    └                    ┘       │
      │                                                 │
35.   │  IF ε_final > ε_max THEN:                        │
36.   │    DỪNG — đã hết privacy budget                  │
37.   │  END IF                                          │
      └────────────────────────────────────────────────┘

38.  END FOR

39.  OUTPUT {wᵢ_T}, (ε_final, δ)
```

---

## 4. So sánh Output Perturbation vs DP-SGD

### 4.1. Bảng so sánh thuật toán

```
┌──────────────────┬─────────────────────────┬──────────────────────────────┐
│                  │  Output Perturbation     │  DP-SGD (thuật toán này)     │
├──────────────────┼─────────────────────────┼──────────────────────────────┤
│ Clip gì?         │ Toàn bộ Δw (1 lần)     │ TỪNG sample gₛ (mỗi step)   │
│ Noise ở đâu?     │ 1 lần sau khi train    │ MỖI mini-batch step          │
│ Sensitivity      │ C                       │ C/B                          │
│ Noise std        │ σ (input trực tiếp)    │ z·C/B (z = noise multiplier) │
│ RDP per round    │ q²·α·C²/(2σ²)         │ n_steps·q²_batch·α/(2z²)    │
│ Subsampling q    │ q_round (data fraction) │ q_batch = B/nᵢ (tự nhiên)   │
│ Bước train       │ Train → clip → noise   │ (clip+noise) per step → ΣΔw │
└──────────────────┴─────────────────────────┴──────────────────────────────┘
```

### 4.2. Tại sao DP-SGD tốt hơn

```
1. Sensitivity giảm B lần:
   Output Perturbation:  Δ₂ = C
   DP-SGD:               Δ₂ = C/B = C/32

2. Subsampling rate nhỏ hơn nhiều:
   Output Perturbation:  q = 0.5 (50% data/round)  → q² = 0.25
   DP-SGD:               q = B/nᵢ = 32/3000 = 0.01 → q² = 0.0001

3. C/B triệt tiêu trong RDP:
   (C/B)² / (z·C/B)² = 1/z²
   → RDP cost per step CHỈ phụ thuộc z, KHÔNG phụ thuộc C hay B

4. Kết hợp: nhiều steps nhưng q² cực nhỏ →
   tổng RDP cost THẤP HƠN nhiều so với Output Perturbation
```

---

## 5. Ví dụ Tính Toán

### Thiết lập:
```
N = 20 nodes, mỗi node ~3,000 mẫu
B = 32, C = 2.0, z = 1.1, T = 100, E = 1, δ = 10⁻⁵
→ ln(1/δ) ≈ 11.51

Model: MLP 784→100→10 → D = 79,510
Kurtosis threshold: T_k = 1.96 × √(24/79510) ≈ 0.0340
```

### Tính per-round:
```
q = B/nᵢ = 32/3000 = 0.01067
n_steps = ⌊3000/32⌋ = 93 steps/round  (drop incomplete batch of 24)

            n_steps · q² · α     93 × 0.000114 × α
ε_round = ─────────────────── = ──────────────────── = 0.00438 · α
                2 · z²                2 × 1.21
```

### Sau 100 rounds:
```
ε_total(α) = 100 × 0.00438 · α = 0.438 · α
```

### Chuyển sang (ε, δ)-DP:
```
ε_DP(α) = 0.438·α + 11.51/(α − 1)
```

### Bảng sweep α:
```
┌────────┬──────────┬──────────────┬──────────┐
│   α    │ 0.438·α  │ 11.51/(α−1)  │  ε_DP    │
├────────┼──────────┼──────────────┼──────────┤
│  1.5   │   0.66   │    23.02     │  23.68   │
│  2     │   0.88   │    11.51     │  12.38   │
│  3     │   1.31   │     5.76     │   7.07   │
│  5     │   2.19   │     2.88     │   5.07   │ ← tối ưu
│ 10     │   4.38   │     1.28     │   5.66   │
│ 20     │   8.76   │     0.61     │   9.37   │
│ 50     │  21.90   │     0.23     │  22.13   │
│ 100    │  43.80   │     0.12     │  43.92   │
└────────┴──────────┴──────────────┴──────────┘

→ α tối ưu ≈ 5, ε_final ≈ 5.07
→ Kết quả: (5.07, 10⁻⁵)-DP
```

### So sánh với Output Perturbation (cùng setup):
```
┌──────────────────────┬──────────────────┬──────────────┐
│                      │ Output Perturb.  │ DP-SGD       │
├──────────────────────┼──────────────────┼──────────────┤
│ Sensitivity          │ C = 2.0          │ C/B = 0.0625 │
│ Noise std per dim    │ σ = 2.0          │ z·C/B = 0.069│
│ Noise norm           │ 564              │ 19.4         │
│ ε_final (100 rounds) │ 36.51            │ 5.07         │
│ Privacy              │ Yếu (ε >> 10)   │ Meaningful   │
└──────────────────────┴──────────────────┴──────────────┘
```

### Kurtosis Detection:
```
Ngưỡng: T_k = 0.0340

Node trung thực (DP-SGD, noise tích lũy qua 93 steps/round):
  Δw̃ᵢ = Σ(noised_grad_steps) → tổng nhiều Gaussian → vẫn Gaussian
  K ≈ 0.001  →  |K| < T_k  →  ✓ Accepted

Node malicious (Scale Attack × 3, clip per-sample nhưng KHÔNG noise):
  Attacker vẫn clip (stealthy) nhưng bỏ noise → scale lên
  Δw̃_malicious giữ non-Gaussian → K >> T_k  →  ✗ Rejected
```

---

## 6. Tại Sao DP-SGD + Kurtosis Vẫn Hoạt Động

### Honest node (DP-SGD):
1. Mỗi step: clip per-sample → average → noise N(0, (z·C/B)²·I)
2. Sau K=93 steps: Δw̃ᵢ = Σ (noised gradients) = signal + Σ noise
3. Tổng noise norm = √K × z·C/B × √D = √93 × 0.069 × 282 ≈ 188
4. Δw̃ᵢ vẫn chứa nhiều Gaussian noise → **K ≈ 0**

### Malicious node (Scale Attack, stealthy):
1. Train với per-sample clipping (để trông bình thường) nhưng **KHÔNG noise** (z=0)
2. Scale × A — gradient đã clip nhưng non-Gaussian vì không có noise
3. Kurtosis scale-invariant: K(A·x) = K(x)
4. Δw̃_malicious vẫn non-Gaussian → **K >> 0**

### Key insight:
- DP-SGD noise **tích lũy qua nhiều steps** → honest update vẫn Gaussian
- Attacker clip per-sample (stealthy) nhưng bỏ noise → non-Gaussian signature giữ nguyên
- Detection hoạt động **không phụ thuộc** cơ chế DP cụ thể
- Chỉ cần noise dominant trong honest update → K ≈ 0
