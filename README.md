# DPFL — Decentralized Federated Learning Framework

Framework mô phỏng hệ thống **Decentralized Federated Learning** với nhiều thuật toán defense/privacy, dùng để **so sánh** hiệu quả giữa các approach.

## Thuật Toán Hỗ Trợ

| Algorithm | Defense | DP | noise_mode | Chạy |
|---|---|---|---|---|
| **FedAvg** | Không | Không | `none` | `python run.py -a fedavg` |
| **DP-FedAvg** | Không | Gaussian DP-SGD | `per_step` | `python run.py -a dp-fedavg` |
| **Krum** | Distance-based Byzantine filtering | Không | `none` | `python run.py -a krum` |
| **Trimmed Mean** | Coordinate-wise trim top/bottom | Không | `none` | `python run.py -a trimmed-mean` |
| **FLTrust** | Cosine trust scoring via root data | Không | `none` | `python run.py -a fltrust` |
| **FLAME** | Clustering + adaptive clipping | Gaussian (self-managed) | `none` | `python run.py -a flame` |
| **DP-SGD + Kurtosis** | Kurtosis excess filtering | Gaussian DP-SGD | `per_step` | `python run.py -a dpsgd-kurtosis` |
| **Trust-Aware D2B-DP** | Z-Score + Cosine + MAD + Trust | Per-edge bounded Gaussian (self-managed) | `none` | `python run.py -a trust-aware` |
| **Noise Game** | Strategic directional + orthogonal + spectral noise | Annealed RDP (self-managed) | `none` | `python run.py -a noise-game` |

> **Ghi chú DP:** `per_step` = DP-SGD inject noise mỗi training step qua `DPSGDTrainer`. `none` + self-managed = thuật toán tự quản lý noise riêng (không qua `dp:` config section).

## Tấn Công Hỗ Trợ

| Attack | Loại | Mô tả | `attack.type` | Params |
|---|---|---|---|---|
| **Scale** | Model poisoning | Nhân gradient × hệ số | `scale` | `scale_factor` (vd: `3.0`) |
| **Sign Flip** | Model poisoning | Đảo dấu gradient: g → -g | `sign_flip` | — |
| **Gaussian Random** | Model poisoning | Thay gradient bằng noise cùng norm | `gaussian_random` | — |
| **ALIE** | Model poisoning | mean - z×std (subtle, bypass defense) | `alie` | `z_max` (vd: `1.0`) |
| **Label Flip** | Data poisoning | Xoay label: y → (y+1) % 10 | `label_flip` | `flip_mode`: `rotate` \| `random` \| `negate` |

> **Chung cho mọi attack:** `start_round` (vd: `0`) — round bắt đầu tấn công. `0` = tấn công từ đầu. Đặt `n_attackers: 0` để tắt hoàn toàn.

## Dataset & Model

| Dataset | Model | Input | Params | Config |
|---|---|---|---|---|
| MNIST | MLP | 784 → 100 → 10 | ~80K | `dataset.name: mnist`, `model.name: mlp` |
| CIFAR-10 | CNN (2-conv + 2-fc) | 3×32×32 → 10 | ~2.1M | `dataset.name: cifar10`, `model.name: cnn` |

## Cài Đặt

```bash
pip install torch torchvision pyyaml matplotlib numpy
# Optional (cho FLAME clustering):
pip install scikit-learn
```

## Chạy

```bash
# Chạy thuật toán (default: dpsgd-kurtosis)
python run.py -a dpsgd-kurtosis
python run.py -a trust-aware
python run.py -a noise-game
python run.py -a fedavg
python run.py -a krum
python run.py -a dp-fedavg
python run.py -a trimmed-mean
python run.py -a fltrust
python run.py -a flame

# Custom config
python run.py -a krum config/krum.yaml
python run.py -a flame config/flame.yaml

# Batch experiments
python batch_runner.py --list                    # Liệt kê experiments
python batch_runner.py -e EXP1 --dry-run         # Preview (không chạy)
python batch_runner.py -e EXP8                    # Chạy EXP-8
python batch_runner.py -e ALL                     # Chạy tất cả
python batch_runner.py -e EXP1 --aggregate        # Tổng hợp kết quả
```

## Config Files

### MNIST (default)

| Config | Algorithm | DP | Aggregation |
|---|---|---|---|
| `config/dpsgd_kurtosis.yaml` | DP-SGD + Kurtosis | per_step | kurtosis_avg |
| `config/fedavg.yaml` | FedAvg | none | simple_avg |
| `config/dp_fedavg.yaml` | DP-FedAvg | per_step | simple_avg |
| `config/krum.yaml` | Krum | none | krum |
| `config/trimmed_mean.yaml` | Trimmed Mean | none | trimmed_mean |
| `config/fltrust.yaml` | FLTrust | none | fltrust |
| `config/flame.yaml` | FLAME | none (self-managed) | flame |
| `config/trust_aware.yaml` | Trust-Aware D2B-DP | none (self-managed) | trust_aware_d2b |
| `config/noise_game.yaml` | Noise Game | none (self-managed) | simple_avg |

### CIFAR-10

| Config | Algorithm |
|---|---|
| `config/cifar10_dpsgd_kurtosis.yaml` | DP-SGD + Kurtosis (CNN, 200 rounds) |
| `config/cifar10_fedavg.yaml` | FedAvg (CNN, 200 rounds) |
| `config/cifar10_krum.yaml` | Krum (CNN, 200 rounds) |
| `config/cifar10_trust_aware.yaml` | Trust-Aware D2B-DP (CNN, 200 rounds) |
| `config/cifar10_noise_game.yaml` | Noise Game (CNN, 200 rounds) |

### Output

Mỗi run tạo `results/<prefix>_<timestamp>/`:

```
results/trust_d2b_20260411_151119/
├── metrics.csv              ← Bảng metrics mỗi round (chính, dùng cho paper)
├── metrics.json             ← Cùng data, JSON format (dùng cho scripting)
├── config.yaml              ← Config đã dùng cho run này
├── report.txt               ← Tóm tắt kết quả cuối
├── experiment.log           ← Log chi tiết quá trình chạy
├── accuracy.png             ← Biểu đồ accuracy theo round
├── accuracy_spread.png      ← Accuracy spread của honest nodes
├── epsilon.png              ← Privacy budget (ε) theo round
├── detection.png            ← Precision / Recall theo round
└── node_data/               ← Metrics từng node từng round
    ├── round_000.json
    ├── round_001.json
    └── ...
```

#### metrics.csv — File chính để collect data cho paper

Mỗi row = 1 round. Gồm **metadata columns** (cố định mọi row, dùng để identify run) + **metric columns** (thay đổi mỗi round).

**Metadata columns** (đầu mỗi row):

| Column | Mô tả | Ví dụ |
|---|---|---|
| `algorithm` | Tên thuật toán (flag `-a`) | `trust-aware` |
| `attack_type` | Loại tấn công | `scale`, `sign_flip`, `alie` |
| `dataset` | Dataset | `mnist`, `cifar10` |
| `seed` | Random seed | `42` |
| `n_nodes` | Tổng nodes | `20` |
| `n_attackers` | Số attacker | `4` |
| `noise_mult` | DP noise multiplier | `1.1` |
| `clip_bound` | L2-norm clip bound | `2.0` |
| `sampling_rate` | Poisson client subsampling rate (1.0 = no subsampling) | `0.3` |
| `split_mode` | Cách chia data | `iid`, `dirichlet` |
| `dirichlet_alpha` | Dirichlet α (non-IID) | `0.5` |

**Metric columns** (mỗi round):

| Column | Mô tả | Dùng cho |
|---|---|---|
| `round` | Round number (0-indexed) | Tất cả |
| `accuracy` | Test accuracy trung bình honest nodes | EXP-1,3,5,7 |
| `test_loss` | Test loss trung bình honest nodes | General |
| `epsilon` | Cumulative privacy budget ε | EXP-4,8 |
| `precision` | Detection precision (TP/(TP+FP)) — xem giải thích bên dưới | EXP-6 |
| `recall` | Detection recall (TP/(TP+FN)) — xem giải thích bên dưới | EXP-6 |
| `f1_score` | Detection F1 score — xem giải thích bên dưới | EXP-6 |
| `mean_update_norm_honest` | Norm trung bình gradient honest | Debug |
| `mean_update_norm_attacker` | Norm trung bình gradient attacker | Debug |
| `best_alpha` | Rényi alpha tối ưu | Debug |

#### Đọc hiểu Detection Metrics (P / R / F1)

Ba chỉ số `precision`, `recall`, `f1_score` đo **khả năng phát hiện attacker** của aggregator mỗi round — **không phải** accuracy của model hay chất lượng labeling.

**Cách tính:** Mỗi round, aggregator trả về danh sách `flagged_ids` (node bị nghi là attacker) và `clean_ids` (node được coi là honest). So sánh với `attacker_ids` thật:

| Ký hiệu | Nghĩa |
|---|---|
| **TP** (True Positive) | Attacker thật, bị flag đúng |
| **FP** (False Positive) | Honest node, bị flag nhầm |
| **FN** (False Negative) | Attacker thật, không bị flag (lọt lưới) |
| **TN** (True Negative) | Honest node, không bị flag (đúng) |

**Công thức:**

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| **Precision** | TP / (TP + FP) | Trong số node bị flag, bao nhiêu % là attacker thật? Cao = ít flag nhầm honest |
| **Recall** | TP / (TP + FN) | Trong số attacker thật, bao nhiêu % bị phát hiện? Cao = ít lọt attacker |
| **F1** | 2 × P × R / (P + R) | Cân bằng giữa Precision và Recall |

**Ví dụ đọc log:**

| Log | Đánh giá |
|---|---|
| `P: 1.00 R: 1.00 F1: 1.00` | Hoàn hảo — phát hiện hết attacker, không flag nhầm honest |
| `P: 0.80 R: 1.00 F1: 0.89` | Bắt hết attacker, nhưng 20% node bị flag là honest (false positive) |
| `P: 1.00 R: 0.50 F1: 0.67` | Không flag nhầm, nhưng chỉ phát hiện 50% attacker (lọt lưới nhiều) |
| `P: 1.00 R: 0.00 F1: 0.00` | Không phát hiện được attacker nào (defense thất bại) |
| `P: 0.50 R: 0.50 F1: 0.50` | Vừa flag nhầm vừa bỏ sót — defense kém |

> **Lưu ý**: Khi `n_attackers: 0` (không có attacker), Precision mặc định = 1.0 và Recall = 0.0 (không có attacker để phát hiện).

> **Lưu ý**: Một số algorithm ghi thêm defense-specific columns (vd: `trust_toward_honest`, `kurtosis_honest`, `tau_drop_attacker`...) tự động append vào cuối.

**Cách đọc nhanh với pandas:**

```python
import pandas as pd

# Load 1 run
df = pd.read_csv("results/trust_d2b_20260411_151119/metrics.csv")

# Final accuracy
final_acc = df.iloc[-1]["accuracy"]

# Convergence curve
df.plot(x="round", y="accuracy")

# Metadata
print(df.iloc[0][["algorithm", "attack_type", "dataset", "seed"]])
```

**Gộp nhiều runs cho paper (vd: mean ± std across seeds):**

```python
import glob

# Load tất cả CSV vào 1 DataFrame
dfs = [pd.read_csv(f) for f in glob.glob("results/*/metrics.csv")]
all_data = pd.concat(dfs, ignore_index=True)

# EXP-1: Final accuracy per algorithm per attack
final = all_data.groupby(["algorithm", "attack_type", "dataset"]).apply(
    lambda g: g[g["round"] == g["round"].max()])
table1 = final.groupby(["algorithm", "attack_type", "dataset"])["accuracy"].agg(["mean", "std"])
```

#### node_data/round_XXX.json — Metrics từng node

Mỗi file = 1 round, chứa dict `{node_id: {...}}`:

```json
{
  "round": 0,
  "nodes": {
    "0": {
      "accuracy": 0.62,
      "test_loss": 1.86,
      "update_norm": 0.32,
      "is_attacker": true,
      "mean_trust": 0.5,
      "tau_drop": 0.498,
      "n_rejected": 0
    }
  }
}
```

| Field | Mô tả |
|---|---|
| `accuracy` | Test accuracy riêng node đó |
| `test_loss` | Test loss riêng node |
| `update_norm` | L2-norm của gradient update |
| `is_attacker` | `true` nếu node là attacker |

> Fields khác (`mean_trust`, `tau_drop`, `kurtosis`...) phụ thuộc vào algorithm.

#### report.txt — Tóm tắt nhanh

```
REPORT: DP-SGD Decentralized Federated Learning
Rounds completed:           50
Final accuracy:             0.8869
Final epsilon:              8750747.97
Avg precision:              1.0000
Avg recall:                 0.0000
Avg F1 score:               0.0000
```

#### config.yaml — Tái tạo run

Bản copy chính xác config YAML đã dùng. Dùng để reproduce:

```bash
python run.py -a trust-aware results/trust_d2b_20260411_151119/config.yaml
```

## Cấu Trúc Project

```
dpfl/
├── run.py                      ← unified entry point (9 thuật toán)
├── batch_runner.py             ← batch experiment runner (8 experiments)
├── experiment_runner.py        ← shared run template
├── config.py                   ← tất cả config dataclasses
├── registry.py                 ← registry pattern cho components
│
├── config/                     ← YAML configs (14 files)
│   ├── dpsgd_kurtosis.yaml
│   ├── fedavg.yaml
│   ├── dp_fedavg.yaml
│   ├── krum.yaml
│   ├── trimmed_mean.yaml
│   ├── fltrust.yaml
│   ├── flame.yaml
│   ├── trust_aware.yaml
│   ├── noise_game.yaml
│   ├── cifar10_dpsgd_kurtosis.yaml
│   ├── cifar10_fedavg.yaml
│   ├── cifar10_krum.yaml
│   ├── cifar10_trust_aware.yaml
│   └── cifar10_noise_game.yaml
│
├── core/                       ← shared base classes + components
│   ├── base_simulator.py       ← BaseSimulator ABC
│   ├── base_node.py            ← Node (train, compute_update)
│   ├── base_aggregator.py      ← BaseAggregator ABC
│   ├── base_noise_mechanism.py ← BaseNoiseMechanism ABC
│   ├── base_accountant.py      ← BaseAccountant ABC
│   ├── base_attack.py          ← BaseAttack ABC
│   ├── gaussian_mechanism.py   ← L2 clip + Gaussian noise
│   ├── renyi_accountant.py     ← Rényi DP accounting
│   ├── dpsgd_trainer.py        ← Local DP-SGD trainer
│   ├── scale_attack.py         ← Scale attack
│   ├── sign_flip_attack.py     ← Sign-flip attack
│   ├── gaussian_random_attack.py ← Gaussian random attack
│   ├── alie_attack.py          ← ALIE attack (A Little Is Enough)
│   └── label_flip_attack.py    ← Label-flip data poisoning
│
├── algorithms/                 ← 1 folder = 1 thuật toán
│   ├── dpsgd_kurtosis/
│   │   ├── simulator.py
│   │   └── kurtosis_aggregator.py
│   ├── trust_aware/
│   │   ├── simulator.py
│   │   ├── node.py
│   │   ├── aggregator.py
│   │   ├── adaptive_clipper.py
│   │   ├── bounded_gaussian.py
│   │   └── per_neighbor_accountant.py
│   ├── noise_game/
│   │   ├── simulator.py
│   │   ├── node.py
│   │   ├── mechanism.py
│   │   └── simple_avg_aggregator.py
│   ├── krum/
│   │   └── krum_aggregator.py
│   ├── trimmed_mean/
│   │   └── trimmed_mean_aggregator.py
│   ├── fltrust/
│   │   ├── fltrust_aggregator.py
│   │   └── simulator.py
│   └── flame/
│       └── flame_aggregator.py
│
├── data/                       ← datasets
│   ├── base_dataset.py
│   ├── mnist_dataset.py
│   └── cifar10_dataset.py
├── models/                     ← neural networks
│   ├── base_model.py
│   ├── mlp_model.py
│   └── cnn_model.py
├── topology/                   ← graph topology
│   └── random_graph.py
└── tracking/                   ← metrics + plots
    └── metrics_tracker.py
```

## Config Reference

Tự tạo file `.yaml` bằng cách chọn giá trị cho từng section:

### `dataset` — Chọn dataset

| Key | Giá trị | Mô tả |
|---|---|---|
| `name` | `mnist` \| `cifar10` | Dataset |
| `split.mode` | `iid` \| `dirichlet` | Cách chia data cho nodes |
| `split.alpha` | `float` (vd: `0.5`) | Dirichlet alpha (nhỏ = non-IID mạnh, lớn ≈ IID). Chỉ dùng khi `mode: dirichlet` |

### `model` — Chọn model

| Key | Giá trị | Mô tả |
|---|---|---|
| `name` | `mlp` \| `cnn` | `mlp` cho MNIST, `cnn` cho CIFAR-10 |
| `hidden_size` | `int` (vd: `100`, `128`) | Kích thước hidden layer |

### `topology` — Cấu hình mạng

| Key | Giá trị | Mô tả |
|---|---|---|
| `n_nodes` | `int` (vd: `20`) | Tổng số nodes |
| `n_attackers` | `int` (vd: `4`) | Số attacker nodes (0 = không tấn công) |
| `n_neighbors` | `int` (vd: `10`) | Số neighbors mỗi node |
| `seed` | `int` | Random seed cho topology |

### `training` — Huấn luyện

| Key | Giá trị | Mô tả |
|---|---|---|
| `n_rounds` | `int` (vd: `100`) | Số vòng FL |
| `local_epochs` | `int` (vd: `1`) | Epochs local training mỗi round |
| `batch_size` | `int` (vd: `64`) | Batch size |
| `lr` | `float` (vd: `0.01`) | Learning rate |
| `n_workers` | `int` (vd: `5`) | Số threads song song khi train nodes |

### `dp` — Differential Privacy

| Key | Giá trị | Mô tả | Dùng bởi |
|---|---|---|---|
| `noise_mode` | `per_step` \| `post_training` \| `none` | Khi nào inject noise. `none` = tắt DP hoặc self-managed | Tất cả |
| `clip_bound` | `float` (vd: `2.0`) | L2-norm clipping bound | DP-SGD (`per_step`/`post_training`), Noise Game |
| `noise_mult` | `float` (vd: `1.1`) | Noise multiplier (σ = noise_mult × clip_bound) | DP-SGD (`per_step`/`post_training`) only |
| `delta` | `float` (vd: `1e-5`) | DP delta parameter | DP-SGD, Trust-Aware, Noise Game |
| `epsilon_max` | `float` (vd: `10.0`) | Budget tối đa, dừng khi vượt | DP-SGD, Trust-Aware, Noise Game |
| `accountant` | `renyi_dpsgd` | Privacy accountant | DP-SGD, Noise Game |
| `accountant_params.alpha_list` | `list[float]` | Rényi alpha values | DP-SGD, Trust-Aware, Noise Game |
| `sampling_rate` | `float ∈ (0, 1]` (default `1.0`) | Poisson client subsampling per round. `1.0` = mọi node active (backward-compat). `q < 1.0` = mỗi honest node active w.p. `q` (coin flip). Seeded deterministic → cùng `seed` cho ra cùng active schedule ở mọi thuật toán (fair compare). Attackers luôn active. | Tất cả |

> **Self-managed DP:** FLAME dùng `aggregation.params` cho noise config. Trust-Aware dùng `trust:` section + `dp.delta/epsilon_max/accountant_params`. Noise Game dùng `noise_game:` section + `dp.clip_bound/delta/epsilon_max/accountant`.

### `attack` — Chọn tấn công

| Key | Giá trị | Mô tả |
|---|---|---|
| `type` | `scale` \| `sign_flip` \| `gaussian_random` \| `alie` \| `label_flip` | Loại tấn công |
| `start_round` | `int` (vd: `0`) | Round bắt đầu tấn công (0 = luôn tấn công). Chung cho mọi attack |

**Params riêng theo `type`:**

| `type` | Key | Giá trị | Mô tả |
|---|---|---|---|
| `scale` | `scale_factor` | `float` (vd: `3.0`) | Hệ số nhân gradient |
| `alie` | `z_max` | `float` (vd: `1.0`) | Cường độ ALIE: mean - z_max × std |
| `label_flip` | `flip_mode` | `rotate` \| `random` \| `negate` | Cách flip label |
| `sign_flip` | — | — | Không có param riêng |
| `gaussian_random` | — | — | Không có param riêng |

### `aggregation` — Chọn aggregator

| `type` | Params | Thuật toán |
|---|---|---|
| `simple_avg` | `{}` | FedAvg, DP-FedAvg |
| `kurtosis_avg` | `centered: bool`, `confidence: float` | DP-SGD + Kurtosis |
| `krum` | `{}` | Multi-Krum |
| `trimmed_mean` | `trim_ratio: float` (vd: `0.2`) | Trimmed Mean |
| `fltrust` | `trust_threshold: float` (vd: `0.1`) | FLTrust |
| `flame` | `noise_mult: float`, `delta: float`, `min_cluster_size: int` | FLAME |
| `trust_aware_d2b` | `{}` | Trust-Aware (cần thêm section `trust:`) |

### Sections riêng cho thuật toán đặc biệt

**`fltrust:`** (chỉ khi dùng `-a fltrust`)

| Key | Giá trị | Mô tả |
|---|---|---|
| `root_data_ratio` | `float` (vd: `0.1`) | % data dùng làm root set |

**`trust:`** (chỉ khi dùng `-a trust-aware`)

| Key | Giá trị | Mô tả |
|---|---|---|
| `trust_init` | `0.5` | Trust score ban đầu |
| `ema_lambda` | `0.8` | EMA decay cho trust |
| `gamma_z` | `3.0` | Z-score threshold |
| `gamma_penalty` | `0.5` | Penalty khi flagged |
| `clip_window` | `3` | Window cho adaptive clipping |
| `eta` | `0.1` | Bounded Gaussian eta |

**`noise_game:`** (chỉ khi dùng `-a noise-game`)

| Key | Giá trị | Mô tả |
|---|---|---|
| `alpha_attack` | `0.5` | Directional noise weight |
| `sigma_0` | `1.0` | Initial noise scale |
| `svd_rank` | `16` | Truncated SVD rank |
| `anneal_kappa` | `0.02` | Noise annealing decay |
| `scaffold` | `true` \| `false` | SCAFFOLD variance reduction |
| `two_track` | `true` \| `false` | Two-track model |

### Các key chung

| Key | Giá trị | Mô tả |
|---|---|---|
| `output_dir` | `string` (vd: `./results`) | Thư mục lưu kết quả |
| `seed` | `int` (vd: `42`) | Global random seed |
| `device` | `auto` \| `cpu` \| `cuda` | Device (optional, default: `auto`) |

### Ví dụ: Tự viết config

```yaml
# Chạy FLAME trên CIFAR-10 với ALIE attack, non-IID, 50 rounds
dataset:
  name: cifar10
  split:
    mode: dirichlet
    alpha: 0.3

model:
  name: cnn
  hidden_size: 128

topology:
  n_nodes: 20
  n_attackers: 6
  n_neighbors: 10
  seed: 42

training:
  n_rounds: 50
  local_epochs: 1
  batch_size: 64
  lr: 0.01
  n_workers: 5

dp:
  noise_mode: none
  clip_bound: 2.0
  noise_mult: 1.1
  delta: 1.0e-5
  epsilon_max: 10.0
  accountant: renyi_dpsgd

attack:
  type: alie
  z_max: 1.5
  start_round: 5

aggregation:
  type: flame
  params:
    noise_mult: 0.01
    delta: 1.0e-5
    min_cluster_size: 2

output_dir: ./results
seed: 42
```

```bash
python run.py -a flame my_config.yaml
```

## Batch Experiments

8 experiments được định nghĩa trong `batch_runner.py`:

| ID | Tên | Runs |
|---|---|---|
| EXP1 | Main Table — Cross-Algorithm Comparison | 324 |
| EXP2 | Ablation Study | 12 |
| EXP3 | Accuracy vs Attacker Fraction | 648 |
| EXP4 | Privacy-Utility Pareto | 180 |
| EXP5 | Non-IID Impact | 324 |
| EXP6 | Detection F1 Across Attacks | 108 |
| EXP7 | Convergence Curves | 54 |
| EXP8 | Epsilon Accumulation | 15 |

## Mở Rộng

### Thêm Thuật Toán Mới

**Nếu chỉ cần aggregator khác** (same training loop):

1. Tạo `algorithms/new_algo/aggregator.py` — extend `BaseAggregator`, dùng `@register`
2. Tạo `config/new_algo.yaml` — đặt `aggregation.type: new_key`
3. Thêm vào `run.py`: registry import + ALGORITHMS entry (reuse `build_dpsgd_kurtosis`)

**Nếu cần custom training loop:**

1. Tạo folder `algorithms/new_algo/`
2. `simulator.py` — extend `BaseSimulator`, override `_create_node()` + `run()`
3. `node.py` (optional) — extend `Node` nếu cần extra state
4. `config/new_algo.yaml`
5. Thêm build function + ALGORITHMS entry vào `run.py`

### Thêm Attack Mới

```python
# core/new_attack.py
from dpfl.core.base_attack import BaseAttack
from dpfl.registry import register, ATTACKS

@register(ATTACKS, "new_attack")
class NewAttack(BaseAttack):
    def perturb(self, honest_update, context=None):
        return -honest_update
```

Config: `attack.type: new_attack`. Thêm import vào `run.py`.

### Thêm Dataset / Model

Tương tự: extend `BaseDataset` hoặc `BaseModel`, `@register`, import trong `run.py`.

## So Sánh Thuật Toán

Tất cả algorithms dùng chung `BaseSimulator.setup()` → cùng seed = cùng:
- Data split (mỗi node cùng data)
- Model init (cùng weights ban đầu)
- Topology (cùng graph)
- Evaluation (cùng test set, cùng metrics)

Khác biệt chỉ ở `run()` — noise strategy + defense mechanism.
