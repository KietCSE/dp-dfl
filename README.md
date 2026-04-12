# DPFL — Decentralized Federated Learning Framework

Framework mô phỏng hệ thống **Decentralized Federated Learning** với nhiều thuật toán defense/privacy, dùng để **so sánh** hiệu quả giữa các approach.

## Thuật Toán Hỗ Trợ

| Algorithm | Defense | DP | Chạy |
|---|---|---|---|
| **FedAvg** | Không | Không | `python run.py -a fedavg` |
| **DP-FedAvg** | Không | Gaussian DP-SGD | `python run.py -a dp-fedavg` |
| **Krum** | Distance-based Byzantine filtering | Không | `python run.py -a krum` |
| **Trimmed Mean** | Coordinate-wise trim top/bottom | Không | `python run.py -a trimmed-mean` |
| **FLTrust** | Cosine trust scoring via root data | Không | `python run.py -a fltrust` |
| **FLAME** | Clustering + adaptive clipping | DP noise (self-managed) | `python run.py -a flame` |
| **DP-SGD + Kurtosis** | Kurtosis excess filtering | Gaussian DP-SGD | `python run.py -a dpsgd-kurtosis` |
| **Trust-Aware D2B-DP** | Z-Score + Cosine + MAD + Trust | Per-edge bounded Gaussian | `python run.py -a trust-aware` |
| **Noise Game** | Strategic directional + orthogonal + spectral noise | Annealed RDP | `python run.py -a noise-game` |

## Tấn Công Hỗ Trợ

| Attack | Loại | Mô tả | Config key |
|---|---|---|---|
| **Scale** | Model poisoning | Nhân gradient × hệ số | `scale` |
| **Sign Flip** | Model poisoning | Đảo dấu gradient: g → -g | `sign_flip` |
| **Gaussian Random** | Model poisoning | Thay gradient bằng noise cùng norm | `gaussian_random` |
| **ALIE** | Model poisoning | mean - z×std (subtle, bypass defense) | `alie` |
| **Label Flip** | Data poisoning | Xoay label: y → (y+1) % 10 | `label_flip` |

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
| `config/trust_aware.yaml` | Trust-Aware D2B-DP | per_step | trust_aware_d2b |
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

| File | Mô tả |
|---|---|
| `metrics.csv` | Metrics mỗi round |
| `metrics.json` | JSON format |
| `accuracy.png` | Accuracy chart |
| `epsilon.png` | Privacy budget |
| `detection.png` | Precision / Recall |
| `node_data/` | Per-node metrics |
| `config.yaml` | Config đã dùng |
| `report.txt` | Summary |

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

| Key | Giá trị | Mô tả |
|---|---|---|
| `noise_mode` | `per_step` \| `post_training` \| `none` | Khi nào inject noise. `none` = tắt DP |
| `clip_bound` | `float` (vd: `2.0`) | L2-norm clipping bound |
| `noise_mult` | `float` (vd: `1.1`) | Noise multiplier (σ = noise_mult × clip_bound) |
| `delta` | `float` (vd: `1e-5`) | DP delta parameter |
| `epsilon_max` | `float` (vd: `10.0`) | Budget tối đa, dừng khi vượt |
| `accountant` | `renyi_dpsgd` | Privacy accountant |
| `accountant_params.alpha_list` | `list[float]` | Rényi alpha values |

### `attack` — Chọn tấn công

| Key | Giá trị | Mô tả |
|---|---|---|
| `type` | `scale` \| `sign_flip` \| `gaussian_random` \| `alie` \| `label_flip` | Loại tấn công |
| `scale_factor` | `float` (vd: `3.0`) | Hệ số nhân (chỉ cho `scale`) |
| `z_max` | `float` (vd: `1.0`) | Cường độ ALIE (chỉ cho `alie`) |
| `flip_mode` | `rotate` \| `random` \| `negate` | Cách flip label (chỉ cho `label_flip`) |
| `start_round` | `int` (vd: `0`) | Round bắt đầu tấn công (0 = luôn tấn công) |

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
