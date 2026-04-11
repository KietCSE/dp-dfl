# DPFL — Decentralized Federated Learning Framework

Framework mô phỏng hệ thống **Decentralized Federated Learning** với nhiều thuật toán defense/privacy, dùng để **so sánh** hiệu quả giữa các approach.

## Thuật Toán Hỗ Trợ

| Algorithm | Defense | DP | Chạy |
|---|---|---|---|
| **FedAvg** | Không | Không | `python run.py -a fedavg` |
| **Krum** | Distance-based Byzantine filtering | Không | `python run.py -a krum` |
| **DP-SGD + Kurtosis** | Kurtosis excess filtering | Gaussian DP-SGD | `python run.py -a dpsgd-kurtosis` |
| **Trust-Aware D2B-DP** | Z-Score + Cosine + MAD + Trust | Per-edge bounded Gaussian | `python run.py -a trust-aware` |
| **Noise Game** | Strategic directional + orthogonal + spectral noise | Annealed RDP | `python run.py -a noise-game` |

## Cài Đặt

```bash
pip install torch torchvision pyyaml matplotlib numpy
```

## Chạy

```bash
# Chạy thuật toán (default: dpsgd-kurtosis)
python run.py -a dpsgd-kurtosis
python run.py -a trust-aware
python run.py -a noise-game
python run.py -a fedavg
python run.py -a krum

# Custom config
python run.py -a krum config/krum.yaml
```

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
├── run.py                      ← unified entry point (tất cả thuật toán)
├── experiment_runner.py        ← shared run template
├── config.py                   ← tất cả config dataclasses
├── registry.py                 ← registry pattern cho components
│
├── config/                     ← YAML configs
│   ├── dpsgd_kurtosis.yaml
│   ├── trust_aware.yaml
│   ├── noise_game.yaml
│   ├── fedavg.yaml
│   └── krum.yaml
│
├── core/                       ← shared base classes + components
│   ├── base_simulator.py       ← BaseSimulator ABC (setup, eval, detect, log)
│   ├── base_node.py            ← Node (train, compute_update)
│   ├── base_aggregator.py      ← BaseAggregator ABC
│   ├── base_noise_mechanism.py ← BaseNoiseMechanism ABC
│   ├── base_accountant.py      ← BaseAccountant ABC
│   ├── base_attack.py          ← BaseAttack ABC
│   ├── gaussian_mechanism.py   ← L2 clip + Gaussian noise
│   ├── renyi_accountant.py     ← Rényi DP accounting
│   ├── dpsgd_trainer.py        ← Local DP-SGD trainer
│   └── scale_attack.py         ← Scale attack
│
├── algorithms/                 ← 1 folder = 1 thuật toán
│   ├── dpsgd_kurtosis/
│   │   ├── simulator.py        ← DFLSimulator(BaseSimulator)
│   │   └── kurtosis_aggregator.py
│   ├── trust_aware/
│   │   ├── simulator.py        ← TrustAwareDFLSimulator(BaseSimulator)
│   │   ├── node.py             ← TrustAwareNode(Node)
│   │   ├── aggregator.py       ← Z-Score + Cosine + MAD + Trust
│   │   ├── adaptive_clipper.py
│   │   ├── bounded_gaussian.py
│   │   └── per_neighbor_accountant.py
│   ├── noise_game/
│   │   ├── simulator.py        ← NoiseGameDFLSimulator(BaseSimulator)
│   │   ├── node.py             ← NoiseGameNode(Node)
│   │   ├── mechanism.py        ← Directional + Orthogonal + Spectral noise
│   │   └── simple_avg_aggregator.py
│   └── krum/
│       └── krum_aggregator.py  ← Multi-Krum Byzantine defense
│
├── data/                       ← datasets
│   ├── base_dataset.py
│   └── mnist_dataset.py
├── models/                     ← neural networks
│   ├── base_model.py
│   └── mlp_model.py
├── topology/                   ← graph topology
│   └── random_graph.py
└── tracking/                   ← metrics + plots
    └── metrics_tracker.py
```

## Cấu Hình YAML

```yaml
dataset:
  name: mnist               # registry key
  split:
    mode: iid               # "iid" | "dirichlet"
    alpha: 0.5

model:
  name: mlp
  hidden_size: 100

topology:
  n_nodes: 20
  n_attackers: 4
  n_neighbors: 10
  seed: 42

training:
  n_rounds: 30
  local_epochs: 1
  batch_size: 64
  lr: 0.01
  n_workers: 5

dp:
  noise_mode: per_step      # "per_step" | "post_training" | "none"
  clip_bound: 2.0
  noise_mult: 1.1
  delta: 1.0e-5
  epsilon_max: 10.0
  accountant: renyi_dpsgd
  accountant_params:
    alpha_list: [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100]

attack:
  type: scale
  scale_factor: 3.0
  start_round: 0            # round bắt đầu tấn công (0 = luôn tấn công)

aggregation:
  type: kurtosis_avg         # registry key
  params:                    # truyền vào constructor
    centered: false
    confidence: 1.96

output_dir: ./results
seed: 42
```

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

**Kế thừa từ `BaseSimulator` (free):**
- `setup()` — data, topology, model init (đảm bảo so sánh công bằng)
- `_train_all_nodes()` — parallel training
- `_evaluate_nodes()` — accuracy + test loss
- `_compute_detection()` — TP/FP/FN/TN
- `_log_round()` — metrics + tracker + console output

**Chỉ cần viết `run()`** — algorithm loop riêng.

```python
# algorithms/new_algo/simulator.py
from dpfl.core.base_simulator import BaseSimulator
from dpfl.core.base_node import Node

class NewAlgoSimulator(BaseSimulator):
    def _create_node(self, node_id, model, data, is_attacker):
        return Node(node_id, model, data, is_attacker)

    def run(self):
        for t in range(self.config.training.n_rounds):
            updates, steps = self._train_all_nodes(round_t=t)
            # ... your algorithm logic ...
            self._log_round(t, epsilon, updates, ...)
```

### Thêm Attack Mới

```python
# core/new_attack.py
from dpfl.core.base_attack import BaseAttack
from dpfl.registry import register, ATTACKS

@register(ATTACKS, "label_flip")
class LabelFlipAttack(BaseAttack):
    def perturb(self, honest_update):
        return -honest_update
```

Config: `attack.type: label_flip`. Thêm import vào `run.py`.

### Thêm Dataset / Model

Tương tự: extend `BaseDataset` hoặc `BaseModel`, `@register`, import trong `run.py`.

## So Sánh Thuật Toán

Tất cả algorithms dùng chung `BaseSimulator.setup()` → cùng seed = cùng:
- Data split (mỗi node cùng data)
- Model init (cùng weights ban đầu)
- Topology (cùng graph)
- Evaluation (cùng test set, cùng metrics)

Khác biệt chỉ ở `run()` — noise strategy + defense mechanism.
