# Experiment Plan — DPFL Rank-A Paper

## 1. Overview

Tài liệu mô tả kế hoạch thực nghiệm cho paper Rank-A về **Differential Privacy trong Decentralized Federated Learning**. Layout: **Option B (Focused)** — 2 Tables + 6 Figures = 8 items, phù hợp paper 10 trang top conference.

### Scope

| Thành phần | Hiện tại | Mục tiêu |
|---|---|---|
| Datasets | MNIST (MLP) | + CIFAR-10 (CNN) |
| Attacks | Scale | + SignFlip, ALIE, GaussRandom, LabelFlip |
| Algorithms | 5 (FedAvg, Krum, Kurtosis, Trust-Aware, Noise Game) | + DP-FedAvg, Trimmed Mean, FLTrust, FLAME = **9 total** |
| Metrics | accuracy, ε, P/R/F1 | + convergence speed, Pareto curves |

### Estimated Effort

| Hạng mục | Runs | Thời gian ước tính |
|---|---|---|
| Implementation components mới | — | ~2.5 tuần |
| Chạy toàn bộ experiments | ~1,700 | ~85 GPU-hours |
| Sinh figures + format tables | — | ~1 tuần |

---

## 2. Standardized Parameters

Tham số cố định cho mọi experiment (trừ khi experiment đó vary tham số cụ thể).

```yaml
topology:
  n_nodes: 20
  n_attackers: 4          # 20% Byzantine
  n_neighbors: 10
  seed: 42

training:
  n_rounds: 100           # MNIST; 200 cho CIFAR-10
  local_epochs: 1
  batch_size: 64
  lr: 0.01

dp:
  clip_bound: 2.0
  noise_mult: 1.1
  delta: 1.0e-5
  epsilon_max: 10.0

seeds: [42, 123, 456]     # 3 runs, report mean ± std
```

### Dataset–Model Pairings

| Dataset | Model | Input | Params | Normalize |
|---|---|---|---|---|
| MNIST | MLP (784→100→10) | 1×28×28 | ~79.5K | (0.1307, 0.3081) |
| CIFAR-10 | CNN (2conv+2fc) | 3×32×32 | ~120K | (0.4914,0.4822,0.4465)/(0.2470,0.2435,0.2616) |

### 9 Algorithms (3 Trục Baseline + 2 Novel)

Papers top-venue yêu cầu so sánh trên 3 trục: **(A) Robust only**, **(B) DP only**, **(C) DP+Robust combined**.

| # | Algorithm | Trục | Color | Line | Defense | DP | Paper gốc |
|---|---|---|---|---|---|---|---|
| 1 | FedAvg | No defense | gray | dashed | None | None | McMahan 2017 |
| 2 | DP-FedAvg | **(B) DP only** | lightblue | dashed | None (simple avg) | Gaussian DP-SGD | McMahan 2018 |
| 3 | Krum | **(A) Robust only** | orange | solid | Distance-based | None | Blanchard, NeurIPS 2017 |
| 4 | Trimmed Mean | **(A) Robust only** | brown | solid | Statistical trim | None | Yin, ICML 2018 |
| 5 | FLTrust | **(A) Robust only** | green | solid | Trust via root dataset | None | Cao, NDSS 2021 |
| 6 | FLAME | **(C) DP + Robust** | cyan | solid | HDBSCAN clustering | Gaussian post-cluster | Nguyen, USENIX Sec 2022 |
| 7 | DP-SGD + Kurtosis | **(C) DP + Robust** | blue | solid | Excess kurtosis | Gaussian DP-SGD | (ours, baseline) |
| 8 | **Trust-Aware D2B-DP** | **(C) DP + Robust** | **red** | **bold** | Multi-signal trust | Per-edge bounded | **(novel ★)** |
| 9 | **Noise Game** | **(C) DP + Robust** | **purple** | **bold** | Game-theoretic noise | Annealed RDP | **(novel ★)** |

**Lý do chọn baselines:**
- **FedAvg** → universal no-defense baseline (mọi paper đều dùng)
- **DP-FedAvg** → chứng minh "DP alone ≠ robust" (accuracy vẫn drop dưới attack)
- **Krum** → most-cited Byzantine defense (2000+ citations)
- **Trimmed Mean** → 2nd most-cited robust aggregation (Yin, ICML 2018)
- **FLTrust** → trust-based defense → **directly comparable** với Trust-Aware D2B-DP (NDSS 2021, 800+ citations)
- **FLAME** → **chỉ baseline uy tín duy nhất kết hợp DP+Robust** (USENIX Security 2022, 400+ citations)
- **DP-SGD+Kurtosis** → DP + statistical detection (stepping stone)
- **Trust-Aware D2B-DP ★** → novel multi-signal trust + per-edge DP
- **Noise Game ★** → novel game-theoretic minimax noise

**Paper narrative flow:**
1. FedAvg → "No defense fails catastrophically"
2. DP-FedAvg → "DP alone doesn't provide robustness"
3. Krum, TrimMean → "Classic robust defenses work but lack privacy guarantee"
4. FLTrust → "Trust-based defense is strong but no formal DP"
5. FLAME → "Existing DP+Robust has limitations (clustering-dependent, sensitive to non-IID)"
6. Trust-Aware → "Our multi-signal trust + per-edge DP outperforms all"
7. Noise Game → "Our game-theoretic approach achieves best robustness-privacy trade-off"

### 6 Attack Types

| Attack | Type | Mechanism | Config |
|---|---|---|---|
| No Attack | — | Baseline (không tấn công) | — |
| Scale | Model poisoning | g × scale_factor | `scale_factor: 3.0` |
| Sign-Flip | Model poisoning | g → −g | — |
| ALIE | Model poisoning (adaptive) | g = μ − z_max × σ (subtle shift) | `alie_z_max: 1.0` |
| Gaussian Random | Model poisoning | g → N(0, σ²) matching ‖g‖ | — |
| Label Flip | Data poisoning | Rotate labels: label → (label+1) % K | `flip_mode: rotate` |

---

## 3. Experiments Chi Tiết

### EXP-1: Table 1 — So Sánh Đa Thuật Toán (Main Result)

**Câu hỏi**: Thuật toán nào đạt accuracy tốt nhất dưới mỗi loại tấn công?

| Mục | Chi tiết |
|---|---|
| **Setup** | 9 algo × 6 attacks × 2 datasets, IID, 3 seeds |
| **Runs** | 9 × 6 × 2 × 3 = **324** |
| **Metric** | Final test accuracy (mean ± std) |
| **Format** | Table grouped by defense category, bold = best, underline = 2nd best |

**Template Table 1:**

```
Test Accuracy (%) under Various Attacks — IID Setting

                          MNIST (MLP)                                        CIFAR-10 (CNN)
Algorithm        NoAtk  Scale  SignFlip  ALIE  GaussRnd  LblFlip | NoAtk  Scale  SignFlip  ALIE  GaussRnd  LblFlip
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
(No defense)
  FedAvg         xx.x   xx.x   xx.x     xx.x  xx.x      xx.x   | xx.x   xx.x   xx.x     xx.x  xx.x      xx.x
(DP only — no robust defense)
  DP-FedAvg      xx.x   xx.x   xx.x     xx.x  xx.x      xx.x   | ...
(Robust only — no DP)
  Krum           xx.x   xx.x   xx.x     xx.x  xx.x      xx.x   | ...
  Trimmed Mean   xx.x   xx.x   xx.x     xx.x  xx.x      xx.x   | ...
  FLTrust        xx.x   xx.x   xx.x     xx.x  xx.x      xx.x   | ...
(DP + Robust combined)
  FLAME          xx.x   xx.x   xx.x     xx.x  xx.x      xx.x   | ...
  Kurtosis       xx.x   xx.x   xx.x     xx.x  xx.x      xx.x   | ...
  Trust-Aware ★  xx.x   xx.x   xx.x     xx.x  xx.x      xx.x   | ...
  Noise Game ★   xx.x   xx.x   xx.x     xx.x  xx.x      xx.x   | ...

★ = novel (ours). Mỗi cell = mean ± std (3 seeds). Bold = best. Underline = 2nd.
Rows grouped: No defense → DP only → Robust only → DP+Robust.
```

**Vai trò**: Central result table — reviewer đọc đầu tiên. Grouping theo category chứng minh:
1. No defense → collapse
2. DP alone → vẫn vulnerable
3. Robust alone → works nhưng no privacy
4. DP+Robust → FLAME có hạn chế, Trust-Aware/Noise Game vượt trội

---

### EXP-2: Table 2 — Ablation Study

**Câu hỏi**: Mỗi component đóng góp bao nhiêu vào performance?

| Mục | Chi tiết |
|---|---|
| **Setup** | MNIST + CIFAR-10, Scale attack, IID, 3 seeds |
| **Runs** | (6 + 7) variants × 2 datasets × 3 seeds = **78** |
| **Metric** | Final test accuracy (mean ± std) |
| **Format** | Table, full system bolded |

**Trust-Aware D2B-DP Ablation Variants:**

| Variant | Mô tả |
|---|---|
| Full | Tất cả components |
| w/o Adaptive Clipping | Fixed clip_bound = 2.0, không dùng rolling history |
| w/o Per-edge Noise | Uniform noise thay vì trust-weighted per-neighbor |
| w/o Cosine Similarity | Chỉ dùng Z-Score, bỏ cosine check |
| w/o MAD Threshold | Fixed threshold thay vì MAD-based dynamic |
| w/o Trust Penalty | Không giảm trust khi node bị filter |

**Noise Game Ablation Variants:**

| Variant | Mô tả |
|---|---|
| Full | Tất cả components |
| w/o Directional Noise | Bỏ n_attack (không cancel attack direction) |
| w/o Orthogonal Noise | Bỏ n_orth (không có DP noise perpendicular) |
| w/o Spectrum Noise | Bỏ n_spec (không có SVD-based noise) |
| w/o SCAFFOLD | Bỏ variance reduction via control variates |
| w/o Momentum | Bỏ momentum acceleration |
| w/o Two-track Model | Bỏ dual clean/robust parameter tracks |

**Vai trò**: Bắt buộc cho novelty claim — chứng minh mỗi component đều cần thiết.

---

### EXP-3: Fig 1 — Accuracy vs Attacker Fraction

**Câu hỏi**: Thuật toán nào robust khi tỷ lệ attacker tăng?

| Mục | Chi tiết |
|---|---|
| **Setup** | Vary n_attackers ∈ {0, 2, 4, 6, 8, 10} (0%–50%) |
| **Datasets** | MNIST + CIFAR-10 |
| **Attacks** | Scale + ALIE (2 attacks đại diện) |
| **Runs** | 9 × 6 × 2 × 2 × 3 = **648** |
| **Format** | **2×2 grid** line chart |

```
┌───────────────────────────────┬───────────────────────────────┐
│  (a) MNIST + Scale Attack     │  (b) MNIST + ALIE Attack      │
│                               │                               │
│  y: Test Accuracy (%)         │  y: Test Accuracy (%)         │
│  x: Attacker Fraction (%)    │  x: Attacker Fraction (%)    │
│  9 lines + shaded error bands│  9 lines + shaded error bands│
│                               │                               │
├───────────────────────────────┼───────────────────────────────┤
│  (c) CIFAR-10 + Scale         │  (d) CIFAR-10 + ALIE          │
│                               │                               │
│  y: Test Accuracy (%)         │  y: Test Accuracy (%)         │
│  x: Attacker Fraction (%)    │  x: Attacker Fraction (%)    │
│  9 lines + shaded error bands│  9 lines + shaded error bands│
└───────────────────────────────┴───────────────────────────────┘
```

**Kỳ vọng**: FedAvg/DP-FedAvg collapse ≥10%. Krum/TrimMean degrade ≥30%. FLTrust stable đến 30%. FLAME moderate. Trust-Aware/Noise Game duy trì >80% tại 40%.

---

### EXP-4: Fig 2 — Privacy-Utility Pareto Frontier

**Câu hỏi**: Privacy budget ảnh hưởng thế nào đến model quality?

| Mục | Chi tiết |
|---|---|
| **Setup** | 4 DP-enabled algorithms: DP-FedAvg, FLAME, Kurtosis, Trust-Aware, Noise Game |
| **Vary** | noise_mult ∈ {0.5, 0.8, 1.1, 1.5, 2.0, 3.0} → different ε |
| **Attack** | Scale |
| **Runs** | 5 × 6 × 2 × 3 = **180** |
| **Format** | **2-panel** line chart |

```
┌──────────────────────────────┬──────────────────────────────┐
│  (a) MNIST                   │  (b) CIFAR-10                │
│                              │                              │
│  x: Final ε (log scale)     │  x: Final ε (log scale)      │
│  y: Test Accuracy (%)        │  y: Test Accuracy (%)         │
│                              │                              │
│  ── DP-FedAvg (lightblue)    │  ↖ Upper-left = best         │
│  ── FLAME (cyan)             │    (high accuracy, low ε)    │
│  ── Kurtosis (blue)          │                              │
│  ── TrustAware (red)         │                              │
│  ── NoiseGame (purple)       │                              │
└──────────────────────────────┴──────────────────────────────┘
```

**Kỳ vọng**: DP-FedAvg worst (no defense). FLAME moderate. Trust-Aware/Noise Game Pareto-dominate tất cả — accuracy cao hơn ở cùng ε.

---

### EXP-5: Fig 3 — Non-IID Impact

**Câu hỏi**: Data heterogeneity ảnh hưởng thế nào đến defense?

| Mục | Chi tiết |
|---|---|
| **Setup** | All 9 algo |
| **Vary** | Dirichlet α ∈ {0.1, 0.3, 0.5, 1.0, 5.0, ∞ (IID)} |
| **Attack** | Scale |
| **Runs** | 9 × 6 × 2 × 3 = **324** |
| **Format** | **2-panel** line chart |

```
┌──────────────────────────────┬──────────────────────────────┐
│  (a) MNIST                   │  (b) CIFAR-10                │
│                              │                              │
│  x: Dirichlet α (log scale) │  x: Dirichlet α (log scale)  │
│  y: Test Accuracy (%)        │  y: Test Accuracy (%)         │
│                              │                              │
│  ← extreme non-IID    IID → │  9 lines                      │
│  α=0.1              α=∞     │                              │
└──────────────────────────────┴──────────────────────────────┘
```

**Kỳ vọng**: Distance-based (Krum, TrimMean) suy giảm mạnh non-IID. FLTrust degrade vì root dataset cũng biased. FLAME clustering bất ổn. Trust-Aware dùng temporal EMA → ổn định. Noise Game dùng directional/momentum → ổn định.

---

### EXP-6: Fig 4 — Detection F1 Across Attacks

**Câu hỏi**: Mỗi defense phát hiện attacker chính xác đến đâu?

| Mục | Chi tiết |
|---|---|
| **Setup** | 6 detection-capable algos (Krum, TrimMean, FLTrust, FLAME, Kurtosis, TrustAware) |
| **Attacks** | All 6 (incl. NoAttack) |
| **Dataset** | MNIST, IID |
| **Runs** | 6 × 6 × 3 = **108** |
| **Format** | **Grouped bar chart** |

```
┌───────────────────────────────────────────────────────────────┐
│  y: F1 Score (0–1)                                            │
│                                                               │
│  ██ Krum ██ TrimMean ██ FLTrust ██ FLAME ██ Kurtosis ██ Trust │
│                                                               │
│  ┃████┃  ┃████┃  ┃████┃  ┃████┃  ┃████┃  ┃████┃             │
│  ┃████┃  ┃████┃  ┃████┃  ┃████┃  ┃████┃  ┃████┃             │
│  ┃████┃  ┃████┃  ┃████┃  ┃████┃  ┃████┃  ┃████┃             │
│  ┃████┃  ┃████┃  ┃████┃  ┃████┃  ┃████┃  ┃████┃             │
│  ───────────────────────────────────────────────              │
│  NoAtk   Scale   SignFlip   ALIE   GaussRnd   LblFlip         │
│                                                               │
│  x: Attack Type                                               │
└───────────────────────────────────────────────────────────────┘
```

**Kỳ vọng**: Kurtosis tốt ở Scale nhưng kém ở ALIE. Krum tốt ở GaussRandom nhưng kém ALIE. FLTrust consistent nhờ cosine với root. FLAME tốt ở Scale nhưng clustering kém ở ALIE. TrustAware consistent cao nhất across all attacks nhờ multi-signal (Z-Score + Cosine + MAD).

**Lưu ý**: Noise Game không explicitly flag attackers → không tham gia detection comparison. Nếu muốn, define trust threshold (trust < 0.3 → flagged).

---

### EXP-7: Fig 5 — Convergence Curves

**Câu hỏi**: Thuật toán nào converge nhanh nhất dưới tấn công?

| Mục | Chi tiết |
|---|---|
| **Setup** | All 9 algo, Scale attack, IID |
| **Runs** | 9 × 2 × 3 = **54** |
| **Format** | **2-panel** line chart |

```
┌──────────────────────────────┬──────────────────────────────┐
│  (a) MNIST (100 rounds)     │  (b) CIFAR-10 (200 rounds)   │
│                              │                              │
│  x: Round                    │  x: Round                     │
│  y: Test Accuracy (%)        │  y: Test Accuracy (%)         │
│                              │                              │
│  FedAvg → diverges           │  9 lines + shaded bands      │
│  Krum → slow convergence     │  Novel → fast convergence    │
│  Novel → fast + high plateau │                              │
└──────────────────────────────┴──────────────────────────────┘
```

**Kỳ vọng**: Bổ sung cho Table 1 bằng dynamic view. FedAvg diverge sớm, Krum converge chậm, Trust-Aware/Noise Game converge nhanh và đạt plateau cao.

---

### EXP-8: Fig 6 — Epsilon Accumulation Over Rounds

**Câu hỏi**: Privacy budget tiêu hao nhanh thế nào?

| Mục | Chi tiết |
|---|---|
| **Setup** | 5 DP algo (DP-FedAvg, FLAME, Kurtosis, TrustAware, NoiseGame), MNIST, Scale attack |
| **Runs** | 5 × 3 = **15** |
| **Format** | **Single** line chart |

```
┌─────────────────────────────────────────────────┐
│  x: Round (0–100)                               │
│  y: Cumulative ε                                │
│                                                 │
│  ── DP-FedAvg (lightblue)                       │
│  ── FLAME (cyan)                                │
│  ── Kurtosis (blue)                             │
│  ── TrustAware (red)                            │
│  ── NoiseGame (purple)                          │
│                                                 │
│  --- Horizontal dashed line at ε_max = 10       │
│                                                 │
│  Annotate round khi mỗi algo chạm budget limit │
└─────────────────────────────────────────────────┘
```

**Kỳ vọng**: Trust-Aware per-edge budget + Noise Game annealing → tiêu hao ε chậm hơn standard DP-SGD composition.

---

## 4. Tổng Hợp Runs

| Experiment | Type | Runs | Est. Time |
|---|---|---|---|
| EXP-1: Main Table | Table 1 | 324 | ~16h |
| EXP-2: Ablation | Table 2 | 78 | ~4h |
| EXP-3: Attacker Fraction | Fig 1 (2×2 grid) | 648 | ~32h |
| EXP-4: Privacy-Utility | Fig 2 (2-panel) | 180 | ~9h |
| EXP-5: Non-IID | Fig 3 (2-panel) | 324 | ~16h |
| EXP-6: Detection F1 | Fig 4 (grouped bar) | 108 | ~5h |
| EXP-7: Convergence | Fig 5 (2-panel) | 54 | ~3h |
| EXP-8: Epsilon | Fig 6 (line) | 15 | ~1h |
| **TOTAL** | **2 Tables + 6 Figures** | **~1,731** | **~86h** |

---

## 5. Components Cần Implement

### 5.1 Dataset Mới: CIFAR-10

| Mục | Chi tiết |
|---|---|
| File | `data/cifar10-dataset.py` |
| Base class | `BaseDataset` |
| Registry key | `"cifar10"` |
| Input shape | (3, 32, 32) |
| Classes | 10 |
| Normalize | mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616) |
| Split | Reuse `_split_iid()` / `_split_dirichlet()` từ `mnist_dataset.py` |

### 5.2 Model Mới: Simple CNN

| Mục | Chi tiết |
|---|---|
| File | `models/cnn-model.py` |
| Base class | `BaseModel` |
| Registry key | `"cnn"` |
| Architecture | Conv(3→32, 3×3, pad=1) → ReLU → Conv(32→64, 3×3, pad=1) → ReLU → MaxPool(2) → Flatten → FC(64×16×16→128) → ReLU → FC(128→10) |
| Params | ~120K |
| Note | Compatible với tất cả algorithms kể cả Noise Game SVD |

### 5.3 Attacks Mới (4 files)

| Attack | File | Registry Key | Complexity |
|---|---|---|---|
| Sign-Flip | `core/sign-flip-attack.py` | `"sign_flip"` | Thấp: `return -honest_update` |
| Gaussian Random | `core/gaussian-random-attack.py` | `"gaussian_random"` | Thấp: `return randn_like(g) * g.norm()` |
| ALIE | `core/alie-attack.py` | `"alie"` | Trung bình: cần neighbor gradient stats |
| Label Flip | `core/label-flip-attack.py` | `"label_flip"` | Trung bình: cần pre-training hook |

### 5.4 Baselines Mới (3 algorithms)

#### DP-FedAvg (Trivial — chỉ cần config)

**Paper**: McMahan et al., "Learning Differentially Private Recurrent Language Models" (ICLR 2018, 3000+ citations)

Không cần code mới! Đã có trong system. Chỉ cần config:
```yaml
dp:
  noise_mode: per_step    # enable DP-SGD
aggregation:
  type: simple_avg        # no defense
```
**Vai trò**: Chứng minh "DP alone ≠ robust".

#### FLTrust (Medium effort)

| Mục | Chi tiết |
|---|---|
| **Paper** | Cao et al., "FLTrust: Byzantine-robust FL via Trust Bootstrapping" (NDSS 2021, 800+ citations) |
| **File** | `algorithms/fltrust/fltrust-aggregator.py` |
| **Registry key** | `"fltrust"` |

**Mechanism:**
1. Mỗi node giữ small root dataset (5-10% local data as validation)
2. Train trên root → root gradient
3. Trust Score: `TS_i = max(0, cos(g_root, g_i))` (ReLU cosine)
4. Normalize: `TS_i = TS_i / sum(TS_j)`
5. Aggregate: `g = sum(TS_i * (||g_root|| / ||g_i||) * g_i)`

**Vai trò**: Directly comparable với Trust-Aware D2B-DP (cùng trust-based). Khác biệt: FLTrust dùng static root reference, Trust-Aware dùng temporal multi-signal.

#### FLAME (Medium-high effort)

| Mục | Chi tiết |
|---|---|
| **Paper** | Nguyen et al., "FLAME: Taming Backdoors in FL" (USENIX Security 2022, 400+ citations) |
| **File** | `algorithms/flame/flame-aggregator.py` |
| **Registry key** | `"flame"` |
| **Dependency** | `sklearn.cluster.HDBSCAN` (sklearn >= 1.3) |

**Mechanism (3 bước):**
1. **Clustering**: HDBSCAN trên cosine distances → loại outlier clusters
2. **Norm Clipping**: Clip updates theo median norm trong largest cluster
3. **DP Noise**: Gaussian noise calibrated theo clipping bound → (ε,δ)-DP

**Vai trò**: Chỉ baseline uy tín duy nhất kết hợp DP+Robust. Không có FLAME → reviewer hỏi: "How does your method compare with existing DP+Robust?"

#### Trimmed Mean

| Mục | Chi tiết |
|---|---|
| **Paper** | Yin et al., "Byzantine-Robust Distributed Learning" (ICML 2018, 1500+ citations) |
| **File** | `algorithms/trimmed-mean/trimmed-mean-aggregator.py` |
| **Registry key** | `"trimmed_mean"` |
| **Logic** | Per-dimension: sort, trim top/bottom k%, average remaining. `trim_ratio=0.2` |

### 5.5 Batch Experiment Runner

| Mục | Chi tiết |
|---|---|
| File | `batch-runner.py` |
| Features | Parameter grid iteration, multi-seed, `--experiment EXP1`, auto-aggregate results |
| Output | Aggregated CSV (mean ± std across seeds), ready for plotting |

### 5.6 Files Cần Sửa

| File | Thay đổi |
|---|---|
| `run.py` | Import new components + ALGORITHMS entries cho FLTrust, FLAME, TrimMean, DP-FedAvg |
| `config.py` | Thêm `alie_z_max`, `flip_mode` vào `AttackConfig`; thêm `root_data_ratio` cho FLTrust |
| `core/base_node.py` | Thêm pre-training hook cho LabelFlip; thêm root_data split cho FLTrust |
| `experiment_runner.py` | Support seed list parameter |

---

## 6. Implementation Priority

```
[WEEK 1] ────────────────────────────────────────────
  P0: CIFAR-10 dataset + CNN model
  P0: Sign-Flip + Gaussian Random attacks (trivial)
  P0: Trimmed Mean aggregator
  P0: DP-FedAvg config (no code, chỉ YAML)

[WEEK 2] ────────────────────────────────────────────
  P0: FLTrust aggregator (trust via root dataset)
  P0: FLAME aggregator (HDBSCAN + clip + DP noise)
  P1: ALIE attack (cần neighbor stats context)
  P1: Label Flip attack (cần Node pre-training hook)

[WEEK 3] ────────────────────────────────────────────
  P1: Batch runner cho experiment sweeps
  P1: Dry-run all 9 algorithms (1 seed, 5 rounds)
  P2: Run EXP-1 (main table) + EXP-7 (convergence)

[WEEK 4] ────────────────────────────────────────────
  P2: Run EXP-3 (attacker fraction sweep)
  P2: Run EXP-4 (privacy-utility) + EXP-5 (non-IID)

[WEEK 5] ────────────────────────────────────────────
  P3: Run EXP-2 (ablation) + EXP-6 (detection) + EXP-8 (epsilon)
  P3: Generate all matplotlib figures + format LaTeX tables
  P3: Final review + sanity checks
```

---

## 7. Presentation Guidelines

### Statistical Reporting
- **3 seeds minimum** (42, 123, 456), report mean ± std
- **Shaded bands** (không phải error bars) trên line charts — dễ đọc hơn
- **Statistical significance**: Wilcoxon signed-rank test giữa best novel vs best baseline

### Table Formatting
- **Bold** = best per column
- **Underline** = second-best
- LaTeX: dùng `booktabs` (không có vertical lines)
- Mỗi cell: `xx.x ± y.y`

### Figure Formatting
- **Consistent color scheme** across ALL figures (xem Section 2)
- **Multi-panel** figures (2×2 grid) = 1 figure nhưng 4× information
- **Log-scale** cho trục ε và trục Dirichlet α
- **Font size** match paper body (~9pt)
- **Legend** đặt ngoài plot hoặc trong khoảng trống
- matplotlib style: `seaborn-v0_8-whitegrid` hoặc tương đương

### LaTeX Figure/Table Placement
```latex
% Khuyến nghị placement
\begin{table}[t]    % Tables ở top of page
\begin{figure}[t]   % Figures ở top of page
\begin{figure*}[t]  % Full-width figures cho 2×2 grid
```

---

## 8. Verification Checklist

- [ ] Mỗi new component: unit test chạy pass
- [ ] Single run mỗi algorithm với CIFAR-10+CNN: không crash
- [ ] Batch runner dry-run (1 seed, 5 rounds): output CSV đúng format
- [ ] FedAvg NoAttack sanity: MNIST ≈ 95%, CIFAR-10 ≈ 70%
- [ ] Tất cả attacks hoạt động: accuracy drop so với NoAttack
- [ ] Detection metrics có giá trị hợp lý (F1 > 0 khi có attack)
- [ ] Figures render đúng labels, legends, colors
- [ ] Tables match CSV data (auto-generated)
- [ ] 3 seeds → std < 5% cho hầu hết configs

---

## 9. Design Decisions (Resolved Questions)

Các vấn đề thiết kế đã được giải quyết trong implementation plan (`plans/260412-0928-experiment-components-implementation/`).

### Q1: LabelFlip interface — ✅ Resolved (Phase 2 + Phase 6)

**Vấn đề**: `BaseAttack.perturb(gradient)` hoạt động post-training. LabelFlip cần modify dataset trước training.

**Giải pháp**: Tạo `LabelFlipDataset` wrapper flip labels on-the-fly + phân nhánh logic trong `Node.compute_update()`.
- `LabelFlipAttack.wrap_dataset(dataset)` → trả về wrapped dataset với labels đã flip
- `Node.apply_data_attack(attack)` gọi trong `BaseSimulator.setup()` cho attacker nodes
- `perturb()` là no-op — poisoning xảy ra qua training data, không qua gradient
- **Logic phân nhánh trong `Node.compute_update()`**:
  - Nếu `hasattr(attack, 'wrap_dataset')` → data poisoning path: train WITH DP noise (data đã poisoned, gradient "honest" từ góc nhìn model)
  - Ngược lại → model poisoning path: train WITHOUT noise, rồi `attack.perturb(update, context)`
- Attacker dùng LabelFlip train như honest node → DP noise vẫn áp dụng → DP có thể giảm bớt hiệu quả attack (feature, không phải bug)

### Q2: ALIE trong decentralized — ✅ Resolved (Phase 2 + Phase 6)

**Vấn đề**: ALIE cần mean/std honest gradients. Attacker chỉ thấy neighbors trong DFL.

**Giải pháp**: Thay đổi signature `BaseAttack.perturb()` toàn hệ thống + 2-pass training.
- **Signature change (Phase 2)**: `BaseAttack.perturb(honest_update, context=None)` — tất cả attacks (Scale, SignFlip, GaussRandom, ALIE, LabelFlip) đều dùng signature mới. Attacks cũ ignore `context`.
- **Cập nhật đồng bộ**: `ScaleAttack.perturb()` + `base_node.py` call site cũng update khi Phase 2 chạy
- `context["neighbor_updates"]` chứa Dict neighbor gradients (chỉ ALIE dùng)
- ALIE compute: `g_attack = mean(neighbors) - z_max * std(neighbors)`
- **2-pass trong `_train_all_nodes()`**: (1) train tất cả nodes → collect updates, (2) ALIE attackers re-perturb dùng neighbor context
- Threat model: neighbor-visible only (realistic cho DFL)

### Q3: Trimmed Mean trim_ratio — ⚠️ Partially Resolved (Phase 3)

**Vấn đề**: Degree=10, max 4 attackers/neighborhood → `trim_ratio` bao nhiêu?

**Giải pháp**: Default `trim_ratio=0.2`.
- N=11 (own + 10 neighbors) → k = floor(0.2 × 11) = 2 → trim 2 top + 2 bottom mỗi dimension
- **Detection heuristic** (approximate, không từ paper gốc): node bị trim >50% dimensions → flagged. Heuristic này cần cho EXP-6 (Detection F1) vì paper gốc Yin 2018 không define explicit detection.
- Giá trị 0.2 là standard trong paper gốc (Yin, ICML 2018)

**Còn mở**: Cần validate empirically sau implement. Nếu 4 attackers trong neighborhood size 10, k=2 chỉ trim 2 → có thể không đủ. Thử `trim_ratio=0.3` (k=3) nếu detection kém.

### Q4: CIFAR-10 convergence rounds — ✅ Resolved (Phase 7)

**Vấn đề**: CNN/CIFAR-10 cần bao nhiêu rounds?

**Giải pháp**: Tất cả CIFAR-10 configs set `n_rounds: 200`.
- MNIST: 100 rounds (MLP converge nhanh)
- CIFAR-10: 200 rounds (CNN cần nhiều rounds hơn)
- Batch runner hardcode: `DATASETS["cifar10"]["rounds"] = 200`

### Q5: FLTrust root dataset trong DFL — ✅ Resolved (Phase 4)

**Vấn đề**: FLTrust cần clean root dataset. Trong DFL không có central server.

**Giải pháp**: Mỗi node split 10% local data làm root set. FLTrust dùng **custom simulator** (không reuse DFLSimulator).
- `Node.split_root_data(ratio=0.1)` → tách root data từ training data (thao tác trên `Subset.indices` — validated accessible)
- `FLTrustSimulator` extend `DFLSimulator`, override `setup()` + `run()`
- Root gradient computed mỗi round: save params → train on root_data → get update → restore params
- **FLTrust aggregate() thêm `root_gradient` parameter** — vi phạm `BaseAggregator` ABC signature nhưng pattern đã tồn tại trong codebase (TrustAwareD2BAggregator cũng thêm extra params). Custom simulator handle call.
- **Non-IID degradation expected**: root dataset biased theo local distribution → FLTrust accuracy drop khi non-IID → đây là điểm yếu cần highlight trong paper (Trust-Aware dùng temporal multi-signal nên không bị ảnh hưởng)

### Q6: FLAME HDBSCAN trên neighbor set nhỏ — ✅ Resolved (Phase 5)

**Vấn đề**: Neighbor set size = 10. HDBSCAN clustering có thể unstable.

**Giải pháp**: HDBSCAN `min_cluster_size=2`, `allow_single_cluster=True` + fallback. FLAME dùng standard 3-param `aggregate()` signature → reuse DFLSimulator (không cần custom simulator).
- Primary: `sklearn.cluster.HDBSCAN` trên cosine distance matrix (precomputed)
- Fallback `_simple_outlier_detection()`: MAD-based average cosine distance thresholding khi HDBSCAN unavailable hoặc N ≤ 3
- Largest cluster = honest, outlier clusters + noise points (-1) = flagged

### Q7: FLAME double noise — ✅ Resolved (Phase 5 + Phase 7)

**Vấn đề**: FLAME thêm DP noise sau clustering. Nếu system cũng có DP-SGD noise → double noise.

**Giải pháp**: FLAME config set `dp.noise_mode: "none"` — FLAME tự quản lý DP.
- Nodes train WITHOUT DP-SGD noise (plain SGD, giống Krum/TrimMean)
- FLAME aggregator tự handle DP pipeline: clustering → adaptive norm clipping (median norm) → `_add_dp_noise(sigma = noise_mult × clip_bound / n_clean)`
- Tránh double noise hoàn toàn
- **Lưu ý cho EXP-4 (Privacy-Utility Pareto)**: Khi sweep `noise_mult`, FLAME adjust qua `aggregation.params.noise_mult` (trong aggregator), KHÔNG qua `dp.noise_mult` (trong DPConfig). Batch runner cần handle mapping này.

### Q8: Noise Game trong EXP-6 — ✅ Resolved (EXP-6 design)

**Vấn đề**: Noise Game không explicitly flag attackers → không compute detection F1.

**Giải pháp**: Loại Noise Game khỏi EXP-6 (Detection F1 comparison).
- EXP-6 dùng 6 detection-capable algos: Krum, TrimMean, FLTrust, FLAME, Kurtosis, TrustAware
- Noise Game defense qua game-theoretic noise injection, không qua explicit detection → không phù hợp cho detection F1 metric
- Noise Game vẫn tham gia đầy đủ trong EXP-1 (main table), EXP-3-5, EXP-7-8

---

## 10. References (Baselines)

| Baseline | Paper | Venue | Year | Citations |
|---|---|---|---|---|
| FedAvg | McMahan et al. | AISTATS | 2017 | 5000+ |
| DP-FedAvg | McMahan et al. | ICLR | 2018 | 3000+ |
| Krum | Blanchard et al. | NeurIPS | 2017 | 2000+ |
| Trimmed Mean | Yin et al. | ICML | 2018 | 1500+ |
| FLTrust | Cao et al. | NDSS | 2021 | 800+ |
| FLAME | Nguyen et al. | USENIX Security | 2022 | 400+ |
