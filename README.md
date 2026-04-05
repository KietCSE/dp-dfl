# DPFL — DP-SGD Decentralized Federated Learning

Modular OOP package cho **Differential Privacy SGD** trong **Decentralized Federated Learning** với **Kurtosis-based Attack Detection**.

## Tổng Quan

Package mô phỏng hệ thống DFL gồm N nodes train cùng 1 model:
- Mỗi node train local bằng **DP-SGD** (per-sample clipping + per-step Gaussian noise)
- Nodes trao đổi model updates qua **random graph topology**
- **Kurtosis-based defense**: lọc attacker bằng excess kurtosis trước khi aggregate
- **Rényi DP accounting**: theo dõi privacy budget (ε, δ)
- Hỗ trợ **Scale Attack** (attacker nhân update × hệ số)

## Cài Đặt

```bash
pip install torch torchvision pyyaml matplotlib numpy
```

Hoặc:
```bash
pip install -r dpfl/requirements.txt
```

## Chạy Nhanh

```bash
# Chạy với config mặc định
python -m dpfl dpfl/config.yaml

# Hoặc custom config
python -m dpfl path/to/my_config.yaml
```

### Output Console

Mỗi 10 round in progress:
```
Round  10/100 | Acc: 0.6523 | eps: 0.29 | P: 0.92 R: 1.00
Round  20/100 | Acc: 0.7845 | eps: 0.58 | P: 0.89 R: 1.00
```

Cuối cùng in summary:
```
SUMMARY: DP-SGD DFL + Kurtosis Detection
Rounds completed: 100
Final accuracy:   0.8912
Final epsilon:    8.73
```

### Output Files (thư mục `results/`)

| File | Mô tả |
|------|--------|
| `metrics.csv` | Metrics mỗi round (accuracy, epsilon, precision, recall, kurtosis) |
| `metrics.json` | Cùng data dạng JSON |
| `accuracy.png` | Biểu đồ accuracy qua các round |
| `epsilon.png` | Biểu đồ privacy budget |
| `kurtosis.png` | Kurtosis honest vs attacker |
| `detection.png` | Precision / Recall detection |

Xem kết quả trên server:
```bash
column -t -s',' results/metrics.csv   # bảng CSV
cat results/metrics.json              # JSON
scp server:path/results/*.png .       # download plot về local
```

## Cấu Hình (YAML)

```yaml
dataset:
  name: mnist          # dataset registry key
  split:
    mode: iid          # "iid" hoặc "dirichlet"
    alpha: 0.5         # Dirichlet concentration (nhỏ = non-IID mạnh)

model:
  name: mlp            # model registry key
  hidden_size: 100

topology:
  n_nodes: 20          # tổng số nodes
  n_attackers: 4       # nodes 0..3 là attacker
  n_neighbors: 10      # mỗi node có ~10 hàng xóm

training:
  n_rounds: 100        # communication rounds
  local_epochs: 1
  batch_size: 32
  lr: 0.1

dp:
  clip_bound: 2.0      # per-sample L2 clip bound C
  noise_mult: 1.1      # noise multiplier z (noise_std = z*C/B)
  delta: 1.0e-5
  alpha_list: [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100]  # Rényi orders
  epsilon_max: 10.0    # dừng nếu vượt budget

attack:
  type: scale           # attack registry key
  scale_factor: 3.0

aggregation:
  type: kurtosis_avg    # aggregator registry key
  centered: false       # uncentered kurtosis (dùng RMS)
  confidence: 1.96      # 95% CI cho threshold

output_dir: results
seed: 42
```

## Kiến Trúc Module

```
dpfl/
├── config.py                # Nested dataclass config + YAML loader
├── registry.py              # Registry decorator cho tất cả components
├── __main__.py              # Entry point
│
├── data/
│   ├── base_dataset.py      # ABC: load(), split(), input_shape, num_classes
│   └── mnist_dataset.py     # MNIST + IID/Dirichlet split
│
├── models/
│   ├── base_model.py        # ABC: forward(), get/set_flat_params()
│   └── mlp_model.py         # MLP 784→H→10
│
├── privacy/
│   ├── base_noise_mechanism.py  # ABC: clip(), add_noise(), clip_and_noise()
│   ├── gaussian_mechanism.py    # L2-norm clip + Gaussian noise
│   └── renyi_accountant.py      # RDP tracking
│
├── training/
│   ├── dpsgd_trainer.py     # Per-sample grad (vmap) + DP-SGD step
│   ├── node.py              # FL node (honest/attacker logic)
│   └── dfl_simulator.py     # Orchestrator: setup → run → export
│
├── attacks/
│   ├── base_attack.py       # ABC: perturb()
│   └── scale_attack.py      # update × scale_factor
│
├── aggregation/
│   ├── base_aggregator.py   # ABC: aggregate() → AggregationResult
│   └── kurtosis_avg_aggregator.py  # Kurtosis filter + simple avg
│
├── topology/
│   └── random_graph.py      # Random undirected graph
│
└── tracking/
    └── metrics_tracker.py   # CSV/JSON export + matplotlib plots
```

## Mở Rộng Hệ Thống

Mọi component đều dùng **Registry pattern** — chỉ cần:
1. Kế thừa ABC tương ứng
2. Dùng decorator `@register(REGISTRY, "tên")`
3. Import file trong `__main__.py`
4. Dùng tên trong config YAML

### Thêm Dataset Mới

```python
# dpfl/data/cifar10_dataset.py
from dpfl.data.base_dataset import BaseDataset
from dpfl.registry import register, DATASETS

@register(DATASETS, "cifar10")
class CIFAR10Dataset(BaseDataset):

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def num_classes(self):
        return 10

    def load(self):
        # Return (train_dataset, test_dataset)
        ...

    def split(self, dataset, n_nodes, mode="iid", alpha=0.5):
        # Return {node_id: Subset}
        ...
```

Config: `dataset.name: cifar10`

### Thêm Model Mới

```python
# dpfl/models/cnn_model.py
from dpfl.models.base_model import BaseModel
from dpfl.registry import register, MODELS

@register(MODELS, "cnn")
class CNN(BaseModel):
    def __init__(self, input_dim, hidden_size, num_classes):
        super().__init__()
        # Define layers...

    def forward(self, x):
        # Forward pass...
```

Config: `model.name: cnn`

> **Lưu ý:** `BaseModel` cung cấp sẵn `get_flat_params()`, `set_flat_params()`, `count_params()` — không cần override.

### Thêm Attack Mới

```python
# dpfl/attacks/noise_attack.py
from dpfl.attacks.base_attack import BaseAttack
from dpfl.registry import register, ATTACKS

@register(ATTACKS, "noise")
class NoiseAttack(BaseAttack):
    def __init__(self, noise_scale=1.0):
        self.noise_scale = noise_scale

    def perturb(self, honest_update):
        return honest_update + torch.randn_like(honest_update) * self.noise_scale
```

Config: `attack.type: noise`

### Thêm Aggregator Mới

```python
# dpfl/aggregation/trimmed_mean_aggregator.py
from dpfl.aggregation.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS

@register(AGGREGATORS, "trimmed_mean")
class TrimmedMeanAggregator(BaseAggregator):
    def __init__(self, param_dim, centered=False, confidence=1.96):
        self.trim_ratio = 0.1  # trim 10% mỗi đầu

    def aggregate(self, own_update, own_params, neighbor_updates):
        # Implement trimmed mean logic...
        return AggregationResult(
            new_params=...,
            clean_ids=[...],
            flagged_ids=[...],
            metrics={...},
        )
```

Config: `aggregation.type: trimmed_mean`

### Thêm Noise Mechanism Mới

```python
# dpfl/privacy/laplace_mechanism.py
from dpfl.privacy.base_noise_mechanism import BaseNoiseMechanism
from dpfl.registry import register, NOISE_MECHANISMS

@register(NOISE_MECHANISMS, "laplace")
class LaplaceMechanism(BaseNoiseMechanism):
    def clip(self, per_sample_grads, clip_bound):
        # L2-norm clip hoặc L1-norm clip...

    def add_noise(self, avg_grad, clip_bound, noise_mult, batch_size):
        # Laplace noise thay vì Gaussian...
```

> **Lưu ý:** `clip_and_noise()` (template method) đã được implement sẵn trong base class — chỉ cần override `clip()` và `add_noise()`.

### Checklist Mở Rộng

1. [ ] Tạo file mới trong thư mục tương ứng
2. [ ] Kế thừa ABC và implement tất cả abstract methods
3. [ ] Dùng `@register(REGISTRY, "tên")` decorator
4. [ ] Thêm import vào `dpfl/__main__.py` (dòng `import dpfl.xxx.yyy  # noqa: F401`)
5. [ ] Cập nhật config YAML với tên mới
6. [ ] Chạy test: `python -m dpfl config.yaml`

## Thuật Toán DP-SGD (Tóm Tắt)

Mỗi round, mỗi node thực hiện:

```
1. Local training (E epochs):
   Mỗi mini-batch B samples:
     a. Tính per-sample gradient: (B, D)
     b. L2-clip từng sample: ||g̃ₛ||₂ ≤ C
     c. Average: ḡ = (1/B) Σ g̃ₛ       → sensitivity = C/B
     d. Noise:  ḡ + N(0, (z·C/B)²·I)
     e. SGD update

2. Exchange updates với hàng xóm

3. Kurtosis detection:
   K(Δw̃ⱼ) = (1/D) Σ (Δw̃ⱼ[i]/RMS)⁴ - 3
   |K| > T_k = 1.96·√(24/D)  →  flagged

4. Aggregate từ clean neighbors only

5. RDP accounting:
   ε_step(α) = q²·α / (2z²),  q = B/n_local
```

Xem chi tiết: `docs/dpsgd-dfl-pseudocode.md`
