# Focused Experiment Plan — EXP1 + EXP6 + EXP7 + EXP8

## Context

Kế hoạch thực nghiệm rút gọn để chọn thuật toán tốt nhất cho paper.
Chạy theo từng file config riêng lẻ via `run.py -a <algo> <config>`.
Config mới tạo trong `config/fast-experiment/` — **không sửa config gốc**.
`start_round` đã được implement sẵn trong `base_simulator.py:117`.

---

## Standardized Parameters

### MNIST (non-IID — harder)

```yaml
dataset:
  name: mnist
  split:
    mode: dirichlet
    alpha: 0.3           # non-IID → accuracy ceiling ~80-85%
model:
  name: mlp
  hidden_size: 100
topology:
  n_nodes: 20
  n_attackers: 8        # 40% Byzantine
  n_neighbors: 10
training:
  n_rounds: 150
  local_epochs: 1
  batch_size: 64
  lr: 0.01
attack:
  scale_factor: 3.0
  start_round: 20
```

### CIFAR-10 (IID — baseline clean)

```yaml
dataset:
  name: cifar10
  split:
    mode: iid            # IID — CIFAR-10 đã khó hơn MNIST tự nhiên
model:
  name: cnn
  hidden_size: 128
topology:
  n_nodes: 20
  n_attackers: 8
  n_neighbors: 10
training:
  n_rounds: 100          # CNN cần nhiều rounds hơn
  local_epochs: 1
  batch_size: 64
  lr: 0.01
attack:
  scale_factor: 3.0
  start_round: 20
```

**Lý do:**
- MNIST dùng non-IID (alpha=0.3) để accuracy tự nhiên thấp hơn → gap giữa các algo rõ hơn
- CIFAR-10 dùng IID vì CNN + CIFAR đã đủ thách thức, non-IID có thể làm kết quả quá noisy
- Cả 2: 40% attackers + scale attack bắt đầu round 20

---

## Step 1 — Tạo Config Files

**Cấu trúc thư mục:**

```
config/fast-experiment/
├── mnist/           ← 9 files, non-IID dirichlet alpha=0.3, 100 rounds
│   ├── fedavg.yaml
│   ├── dp_fedavg.yaml
│   ├── krum.yaml
│   ├── trimmed_mean.yaml
│   ├── fltrust.yaml
│   ├── flame.yaml
│   ├── dpsgd_kurtosis.yaml
│   ├── trust_aware.yaml
│   └── noise_game.yaml
└── cifar10/         ← 9 files, IID, 150 rounds, model=cnn
    ├── fedavg.yaml
    ├── dp_fedavg.yaml
    ├── krum.yaml
    ├── trimmed_mean.yaml
    ├── fltrust.yaml
    ├── flame.yaml
    ├── dpsgd_kurtosis.yaml
    ├── trust_aware.yaml
    └── noise_game.yaml
```

### MNIST configs — base và thay đổi

| File | Base | Thay đổi so với base |
|------|------|----------------------|
| `mnist/fedavg.yaml` | `config/fedavg.yaml` | n_attackers=8, dirichlet/0.3, start_round=20, n_rounds=100 |
| `mnist/dp_fedavg.yaml` | `config/dp_fedavg.yaml` | như trên |
| `mnist/krum.yaml` | `config/krum.yaml` | như trên |
| `mnist/trimmed_mean.yaml` | `config/trimmed_mean.yaml` | như trên |
| `mnist/fltrust.yaml` | `config/fltrust.yaml` | như trên |
| `mnist/flame.yaml` | `config/flame.yaml` | như trên |
| `mnist/dpsgd_kurtosis.yaml` | `config/dpsgd_kurtosis.yaml` | như trên + `scale_factor: 3.0` (từ 5.0) |
| `mnist/trust_aware.yaml` | `config/trust_aware.yaml` | như trên |
| `mnist/noise_game.yaml` | `config/noise_game.yaml` | như trên + `scale_factor: 3.0` (từ -5.0) |

### CIFAR-10 configs — base và thay đổi

Dùng `cifar10_*.yaml` nếu có, còn không thì dùng mnist base + đổi dataset/model.

| File | Base | Thay đổi so với base |
|------|------|----------------------|
| `cifar10/fedavg.yaml` | `config/cifar10_fedavg.yaml` | n_attackers=8, iid, start_round=20, n_rounds=200 |
| `cifar10/dp_fedavg.yaml` | `config/dp_fedavg.yaml` | như trên + dataset=cifar10, model=cnn/128 |
| `cifar10/krum.yaml` | `config/cifar10_krum.yaml` | n_attackers=8, iid, start_round=20, n_rounds=200 |
| `cifar10/trimmed_mean.yaml` | `config/trimmed_mean.yaml` | như trên + dataset=cifar10, model=cnn/128 |
| `cifar10/fltrust.yaml` | `config/fltrust.yaml` | như trên + dataset=cifar10, model=cnn/128 |
| `cifar10/flame.yaml` | `config/flame.yaml` | như trên + dataset=cifar10, model=cnn/128 |
| `cifar10/dpsgd_kurtosis.yaml` | `config/cifar10_dpsgd_kurtosis.yaml` | n_attackers=8, iid, start_round=20, n_rounds=200, scale_factor=3.0 |
| `cifar10/trust_aware.yaml` | `config/cifar10_trust_aware.yaml` | n_attackers=8, iid, start_round=20, n_rounds=200 |
| `cifar10/noise_game.yaml` | `config/cifar10_noise_game.yaml` | n_attackers=8, iid, start_round=20, n_rounds=200, scale_factor=3.0 |

---

## Step 2 — Run Commands

```bash
cd "/Users/lap15791/Documents/Differential privacy/robust & privacy/dpfl"

# ── MNIST ─────────────────────────────────────────────────────────
python run.py -a fedavg         config/fast-experiment/mnist/fedavg.yaml
python run.py -a dp-fedavg      config/fast-experiment/mnist/dp_fedavg.yaml
python run.py -a krum           config/fast-experiment/mnist/krum.yaml
python run.py -a trimmed-mean   config/fast-experiment/mnist/trimmed_mean.yaml
python run.py -a fltrust        config/fast-experiment/mnist/fltrust.yaml
python run.py -a flame          config/fast-experiment/mnist/flame.yaml
python run.py -a dpsgd-kurtosis config/fast-experiment/mnist/dpsgd_kurtosis.yaml
python run.py -a trust-aware    config/fast-experiment/mnist/trust_aware.yaml
python run.py -a noise-game     config/fast-experiment/mnist/noise_game.yaml

# ── CIFAR-10 ──────────────────────────────────────────────────────
python run.py -a fedavg         config/fast-experiment/cifar10/fedavg.yaml
python run.py -a dp-fedavg      config/fast-experiment/cifar10/dp_fedavg.yaml
python run.py -a krum           config/fast-experiment/cifar10/krum.yaml
python run.py -a trimmed-mean   config/fast-experiment/cifar10/trimmed_mean.yaml
python run.py -a fltrust        config/fast-experiment/cifar10/fltrust.yaml
python run.py -a flame          config/fast-experiment/cifar10/flame.yaml
python run.py -a dpsgd-kurtosis config/fast-experiment/cifar10/dpsgd_kurtosis.yaml
python run.py -a trust-aware    config/fast-experiment/cifar10/trust_aware.yaml
python run.py -a noise-game     config/fast-experiment/cifar10/noise_game.yaml
```

**Output mỗi run** (auto-generated tại `results/<algo>_<timestamp>/`):

```
metrics.csv          ← per-round: round, accuracy, f1_score, precision, recall, epsilon, ...
accuracy.png         ← single-run accuracy curve (auto)
detection.png        ← single-run precision/recall (auto)
report.txt           ← human-readable summary
config.yaml          ← bản copy config đã dùng
```

**Schema của metrics.csv** (23 cột):

```
algorithm, attack_type, dataset, seed, n_nodes, n_attackers,
noise_mult, clip_bound, split_mode, dirichlet_alpha,
round, epsilon, accuracy, test_loss, f1_score,
precision, recall, best_alpha,
mean_update_norm_honest, mean_update_norm_attacker,
kurtosis_honest, kurtosis_attacker
```

---

## Step 3 — Visualize Results

Sau khi chạy xong 18 runs (9 MNIST + 9 CIFAR-10), load CSV vào công cụ tùy chọn và vẽ 4 figures.
Mỗi figure có 2 panel (MNIST + CIFAR-10) nếu cần so sánh cả hai dataset.

---

### Figure 1 — EXP-1: Table 1 (Accuracy Comparison)

**Data cần lấy:**
- MNIST: cột `accuracy` tại `round = 99` (last round)
- CIFAR-10: cột `accuracy` tại `round = 199` (last round)

**Loại biểu đồ:** Grouped bar chart (hoặc Table dạng grid cho paper)

- Trục x: 9 algorithms (nhóm theo category)
- Trục y: Final Test Accuracy (%)
- Mỗi nhóm có 2 bars — MNIST và CIFAR-10
- Hoặc vẽ 2 chart riêng (MNIST / CIFAR-10), mỗi chart x = 9 algos
- Color scheme:
  - gray = FedAvg | lightblue = DP-FedAvg
  - orange = Krum | brown = TrimMean | green = FLTrust
  - cyan = FLAME | blue = Kurtosis
  - red = TrustAware ★ | purple = NoiseGame ★

> Cho paper: Table dạng grid dễ đọc hơn — rows = 9 algos, cols = 2 datasets, cell = accuracy

---

### Figure 2 — EXP-6: Fig 4 (Detection F1 Across Attacks)

**Data cần lấy:** trung bình cột `f1_score` của **10 rounds cuối** từ MNIST metrics.csv của **6 algos có detection**: Krum, TrimMean, FLTrust, FLAME, Kurtosis, TrustAware

> EXP-6 chỉ dùng MNIST (IID theo experiment plan gốc). CIFAR-10 không cần.
> Dùng mean 10 rounds cuối (MNIST: `round >= 90`) để tránh nhiễu.
> FedAvg, DP-FedAvg, NoiseGame không có explicit detection → không tham gia.

**Loại biểu đồ:** Grouped bar chart

- Trục x: 6 attack types (NoAttack, Scale, SignFlip, ALIE, GaussRandom, LabelFlip)
- Trục y: F1 Score (0–1)
- Mỗi nhóm x có 6 bars — 1 bar mỗi algo có detection
- Cùng color scheme

---

### Figure 3 — EXP-8: Fig 6 (Epsilon Accumulation Over Rounds)

**Data cần lấy:** cột `round` và `epsilon` của **5 DP-enabled algos**: DP-FedAvg, FLAME, Kurtosis, TrustAware, NoiseGame
- MNIST: 100 rounds | CIFAR-10: 200 rounds

> FedAvg, Krum, TrimMean, FLTrust không có DP → không tham gia.
> Cột `epsilon` là cumulative, đã có sẵn trong metrics.csv.

**Loại biểu đồ:** Multi-line chart (2 panel: MNIST + CIFAR-10)

- Trục x: Round
- Trục y: Cumulative ε
- 5 lines mỗi panel, color: lightblue=DP-FedAvg, cyan=FLAME, blue=Kurtosis, red=TrustAware, purple=NoiseGame
- **Đường ngang dashed tại ε = 10.0** với label `"ε_max"` → algo chạm sớm hơn = tiêu thụ budget nhanh hơn
- Kỳ vọng: TrustAware + NoiseGame tiêu hao chậm hơn DP-FedAvg standard

---

### Figure 4 — EXP-7: Fig 5 (Convergence Curves)

**Data cần lấy:** cột `round` và `accuracy` toàn bộ rounds, attack = scale, từ cả MNIST và CIFAR-10

**Loại biểu đồ:** Multi-line chart (2 panel: MNIST + CIFAR-10)

- Trục x: Round (0–100 MNIST | 0–200 CIFAR-10)
- Trục y: Test Accuracy (%)
- 9 lines mỗi panel, cùng color scheme
- **Đường dọc dashed tại round = 20** với label `"Attack starts"`
- Phase 1 (0–19): lines gần nhau — đang converge bình thường
- Phase 2 (20+): lines phân tán — FedAvg/DP-FedAvg drop mạnh, TrustAware/NoiseGame giữ cao

---

## Expected Behavior

### MNIST (non-IID, alpha=0.3)

| Algorithm | Clean (Rd 0–19) | After attack (Rd 20+) |
|-----------|----------------|----------------------|
| FedAvg | ~82% | drops to ~28% |
| DP-FedAvg | ~79% | ~30% |
| Krum | ~80% | ~50% |
| TrimMean | ~81% | ~55% |
| FLTrust | ~80% | ~62% |
| FLAME | ~79% | ~58% |
| Kurtosis | ~80% | ~65% |
| **Trust-Aware ★** | ~81% | **~75%** |
| **Noise Game ★** | ~80% | **~72%** |

### CIFAR-10 (IID)

| Algorithm | Clean (Rd 0–19) | After attack (Rd 20+) |
|-----------|----------------|----------------------|
| FedAvg | ~55% | drops to ~20% |
| DP-FedAvg | ~52% | ~22% |
| Krum | ~54% | ~38% |
| TrimMean | ~55% | ~42% |
| FLTrust | ~54% | ~48% |
| FLAME | ~52% | ~44% |
| Kurtosis | ~53% | ~52% |
| **Trust-Aware ★** | ~55% | **~62%** |
| **Noise Game ★** | ~54% | **~58%** |

---

## Sanity Checks

- [ ] 18 files trong `config/fast-experiment/mnist/` và `config/fast-experiment/cifar10/`
- [ ] Tất cả 18 file có `attack.start_round: 20` và `topology.n_attackers: 8`
- [ ] MNIST configs: `dataset.name: mnist`, `model.name: mlp`, `n_rounds: 100`, `split.mode: dirichlet`
- [ ] CIFAR-10 configs: `dataset.name: cifar10`, `model.name: cnn`, `n_rounds: 200`, `split.mode: iid`
- [ ] `scale_factor: 3.0` trong tất cả 18 (đặc biệt dpsgd_kurtosis và noise_game)
- [ ] FedAvg MNIST NoAttack ≈ 82% (thấp hơn IID do alpha=0.3)
- [ ] FedAvg CIFAR-10 NoAttack ≈ 55% (CNN + IID baseline)
- [ ] Cả 2 dataset: FedAvg drop rõ sau round 20 trong Figure 4
