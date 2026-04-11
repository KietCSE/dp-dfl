# DPFL — DP-SGD Decentralized Federated Learning

Modular OOP package cho **Differential Privacy SGD** trong **Decentralized Federated Learning** với **Kurtosis-based Attack Detection**.

## Tổng Quan

Package mô phỏng hệ thống DFL gồm N nodes train cùng 1 model:
- Mỗi node train local bằng **DP-SGD** (per-sample clipping + per-step Gaussian noise)
- Nodes trao đổi model updates qua **random graph topology**
- **Kurtosis-based defense**: lọc attacker bằng excess kurtosis trước khi aggregate
- **Rényi DP accounting**: theo dõi privacy budget (ε, δ)
- Hỗ trợ **Scale Attack** (attacker nhân update × hệ số)
- **Configurable noise timing**: per-step DP-SGD, post-training DP, hoặc plain SGD

## Cài Đặt

```bash
pip install torch torchvision pyyaml matplotlib numpy
```

Hoặc:
```bash
pip install -r requirements.txt
```

## Chạy Nhanh

```bash
# Chạy với config mặc định
python run.py

# Chạy với config tùy chỉnh
python run.py path/to/my_config.yaml
```

### Output Console

Mỗi 10 round in progress:
```
Round  10/100 | Acc: 0.6523 | eps: 0.29 | P: 0.92 R: 1.00
Round  20/100 | Acc: 0.7845 | eps: 0.58 | P: 0.89 R: 1.00
```

Cuối cùng in summary:
```
SUMMARY: DP-SGD Decentralized Federated Learning
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

Xem kết quả:
```bash
column -t -s',' results/<timestamp>/metrics.csv   # bảng CSV
cat results/<timestamp>/metrics.json               # JSON
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
  noise_mode: per_step # "per_step" | "post_training" | "none"
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
  params:               # constructor kwargs (khác nhau tùy aggregator)
    centered: false     # uncentered kurtosis (dùng RMS)
    confidence: 1.96    # 95% CI cho threshold

output_dir: results
seed: 42
```

## Noise Mode — Chế Độ Tiêm Nhiễu

Trainer hỗ trợ 3 chế độ tiêm nhiễu qua `dp.noise_mode`:

| Mode | Training Loop | DP Application | Use Case |
|------|--------------|----------------|----------|
| `per_step` | vmap per-sample grads → clip → noise mỗi batch | Mỗi batch | DP-SGD chuẩn (default) |
| `post_training` | Standard SGD (nhanh, không vmap) | Clip + noise trên final update sau khi train xong | Output perturbation |
| `none` | Standard SGD (nhanh, không vmap) | Không có DP | Baseline comparison |

**Hành vi với attacker** (`apply_noise=False`):

| Mode | Attacker behavior |
|------|-------------------|
| `per_step` | Clip per batch, không noise |
| `post_training` | Standard SGD → clip final update, không noise |
| `none` | Plain SGD (không clip, không noise) |

**Privacy accounting**:
- `per_step`: Accountant tích lũy RDP cost mỗi step
- `post_training`: Accountant tích lũy per round (dev tự viết accountant phù hợp)
- `none`: Không có accountant (`accountant=None`), epsilon = 0

## Trust-Aware D2B-DP (Thuật Toán Bảo Mật Chủ Động)

Mở rộng hệ thống DFL với **Trust-Aware D2B-DP**: per-edge personalized noise dựa trên trust scores, adaptive clipping, Z-Score + cosine anomaly detection, MAD-based dynamic filtering, và penalty mechanism.

Tham khảo chi tiết thuật toán: `docs/Trust-Aware-D2B-DP.md`

### Chạy Trust-Aware D2B-DP

```bash
# Cách 1: Từ thư mục hiện tại (dpfl/)
python run-trust-aware.py
python run-trust-aware.py path/to/my-trust-config.yaml

# Cách 2: Từ thư mục cha (chứa dpfl/)
cd ..
python -m dpfl.trust-aware-main
python -m dpfl.trust-aware-main path/to/my-trust-config.yaml
```

### Output Console

Mỗi round in progress:
```
Round   1/50 | Acc: 0.3993 | Trust H->H: 0.533 H->A: 0.534 | P: 1.00 R: 0.00 F1: 0.00
Round  10/50 | Acc: 0.7168 | Trust H->H: 0.340 H->A: 0.188 | P: 0.40 R: 0.62 F1: 0.49
Round  30/50 | Acc: 0.8200 | Trust H->H: 0.650 H->A: 0.050 | P: 0.85 R: 0.90 F1: 0.87
```

- **Trust H->H**: Trust trung bình của honest nodes → honest neighbors (mong muốn ↑)
- **Trust H->A**: Trust trung bình của honest nodes → attacker neighbors (mong muốn ↓)

### Output Files (thư mục `results/trust_d2b_<timestamp>/`)

| File | Mô tả |
|------|--------|
| `metrics.csv` | Metrics mỗi round (accuracy, precision, recall, f1, trust, tau_drop) |
| `metrics.json` | Cùng data dạng JSON |
| `accuracy.png` | Biểu đồ accuracy qua các round |
| `accuracy_spread.png` | Phân bố accuracy các honest nodes |
| `detection.png` | Precision / Recall detection |
| `node_data/round_NNN.json` | Per-node metrics chi tiết mỗi round (trust, z-score, behavior) |
| `config.yaml` | Bản sao config đã dùng |
| `report.txt` | Summary report |

Xem kết quả:
```bash
column -t -s',' results/trust_d2b_*/metrics.csv   # bảng CSV
cat results/trust_d2b_*/report.txt                 # summary
cat results/trust_d2b_*/node_data/round_049.json   # per-node round cuối
```

### Cấu Hình Trust-Aware (`trust-aware-config.yaml`)

Ngoài các section chung (`dataset`, `model`, `topology`, `training`, `dp`, `attack`), config thêm section `trust`:

```yaml
trust:
  trust_init: 0.5        # T[i,j] khởi tạo (cold-start trung lập)
  ema_lambda: 0.8        # EMA memory factor (0.8 = 80% lịch sử + 20% hiện tại)
  rho_min: 0.1           # Ngân sách privacy tối thiểu
  rho_max: 5.0           # Ngân sách privacy tối đa
  beta: 0.01             # Hệ số tăng ngân sách theo thời gian
  gamma_penalty: 0.5     # Hệ số phạt trust (nhân 0.5 khi bị reject)
  gamma_z: 3.0           # Hệ số nhân Z-Score threshold
  sigma_floor_z: 0.0001  # Sàn độ nhạy Z-Score (tránh threshold = 0)
  alpha_drop: 2.0        # Hệ số MAD cho ngưỡng loại bỏ tau_drop
  sigma_floor_drop: 0.001 # Sàn độ nhạy trust filtering
  clip_window: 3         # FIFO queue size cho adaptive clip (k)
  temporal_window: 5     # Buffer size cho temporal windowing (W)
  eta: 0.1               # Bounded noise eta parameter
```

### Pipeline 4 Giai Đoạn (Mỗi Round)

```
Phase 1: Train → Adaptive Clip (FIFO history → dynamic threshold → L2 clip)
Phase 2: Trust → Budget → σ² → Bounded Noise → Gửi per-edge
Phase 3: Nhận → Z-Score → Cosine → Harmonic Mean → EMA Trust Update
Phase 4: MAD τ_drop → Reject + Penalty → Mean Aggregate từ clean set
```

### Files Mới (Trust-Aware D2B-DP)

```
dpfl/
├── trust_aware_config.py                      # TrustConfig + TrustAwareExperimentConfig
├── trust-aware-main.py                        # Entry point
├── trust-aware-config.yaml                    # Default config
├── training/
│   ├── trust-aware-node.py                    # TrustAwareNode(Node) + trust state
│   └── trust-aware-dfl-simulator.py           # 4-phase simulator
├── privacy/
│   ├── adaptive_clipper.py                    # FIFO queue + dynamic clip
│   ├── bounded_gaussian_mechanism.py          # Per-neighbor bounded noise
│   └── per_neighbor_rdp_accountant.py         # Per-edge RDP tracking
└── aggregation/
    └── trust-aware-d2b-aggregator.py          # Z-Score+Cosine+MAD+Trust+Penalty
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
│   ├── base_accountant.py       # ABC: step(), get_epsilon()
│   └── renyi_dpsgd.py      # RDP tracking cho per-step DP-SGD
│
├── training/
│   ├── dpsgd_trainer.py     # Local training: per_step / post_training / none
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

### Cơ Chế Hoạt Động

Hệ thống dùng **Registry pattern** — mỗi component (dataset, model, attack, aggregator, noise mechanism, accountant) được đăng ký bằng decorator `@register` với một **key name**. Khi chạy, `__main__.py` đọc config YAML → tra key trong registry → khởi tạo class tương ứng.

**Luồng hoạt động:**

```
config.yaml          config.py              registry.py           __main__.py
─────────────       ────────────           ─────────────         ─────────────
dataset:            DatasetConfig          DATASETS = {}         dataset_cls = DATASETS["cifar10"]
  name: "cifar10"     name: str            ┌─"mnist"→MNISTDataset
                                           └─"cifar10"→CIFAR10Dataset  ← bạn thêm

model:              ModelConfig            MODELS = {}           model_cls = MODELS["cnn"]
  name: "cnn"         name: str            ┌─"mlp"→MLP
                                           └─"cnn"→CNN           ← bạn thêm
```

**Quy trình thêm component mới (4 bước):**

1. Tạo file `.py` mới, kế thừa base class + dùng `@register(REGISTRY, "key")`
2. Thêm `import` vào `dpfl/__main__.py` để kích hoạt decorator
3. (Nếu cần param mới) Thêm field vào dataclass config trong `dpfl/config.py` + cập nhật logic khởi tạo trong `__main__.py`
4. Cập nhật `config.yaml` với key + params mới

---

### Thêm Dataset Mới

**Base class:** `dpfl/data/base_dataset.py` — cần implement 4 abstract methods.

**Bước 1 — Tạo file:**
```python
# dpfl/data/cifar10_dataset.py
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Dict, Tuple

from dpfl.data.base_dataset import BaseDataset
from dpfl.registry import register, DATASETS


@register(DATASETS, "cifar10")          # ← key dùng trong config.yaml
class CIFAR10Dataset(BaseDataset):

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (3, 32, 32)              # C, H, W

    @property
    def num_classes(self) -> int:
        return 10

    def load(self) -> Tuple[Dataset, Dataset]:
        tf = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
        test = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
        return train, test

    def split(self, dataset: Dataset, n_nodes: int,
              mode: str = "iid", alpha: float = 0.5) -> Dict[int, Subset]:
        # Chia dataset cho n_nodes, trả về {node_id: Subset}
        # Tham khảo dpfl/data/mnist_dataset.py cho IID / Dirichlet split
        ...
```

**Bước 2 — Đăng ký import** trong `dpfl/__main__.py`:
```python
import dpfl.data.cifar10_dataset  # noqa: F401
```

**Bước 3 — Config:** Không cần sửa `config.py` (dùng chung `DatasetConfig.name`).

**Bước 4 — YAML:**
```yaml
dataset:
  name: cifar10       # ← key đã register
  split:
    mode: iid
```

> `__main__.py` sẽ gọi: `DATASETS["cifar10"]()` → lấy `input_shape`, `num_classes` → truyền vào model.

---

### Thêm Model Mới

**Base class:** `dpfl/models/base_model.py` — cần implement `forward()`. Base class cung cấp sẵn `get_flat_params()`, `set_flat_params()`, `count_params()`.

**Constructor phải nhận 3 tham số** (do `__main__.py` truyền vào):
- `input_dim`: tích các chiều của `dataset.input_shape` (ví dụ: 784 cho MNIST, 3072 cho CIFAR10)
- `hidden_size`: từ `config.model.hidden_size`
- `num_classes`: từ `dataset.num_classes`

**Bước 1 — Tạo file:**
```python
# dpfl/models/cnn_model.py
import torch
import torch.nn as nn

from dpfl.models.base_model import BaseModel
from dpfl.registry import register, MODELS


@register(MODELS, "cnn")
class CNN(BaseModel):
    def __init__(self, input_dim: int = 784, hidden_size: int = 100,
                 num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
```

**Bước 2 — Đăng ký import** trong `dpfl/__main__.py`:
```python
import dpfl.models.cnn_model  # noqa: F401
```

**Bước 3 — Config:** Không cần sửa `config.py` (dùng chung `ModelConfig`).

**Bước 4 — YAML:**
```yaml
model:
  name: cnn           # ← key đã register
  hidden_size: 128
```

> `__main__.py` gọi: `MODELS["cnn"](input_dim=784, hidden_size=128, num_classes=10)`

---

### Thêm Attack Mới

**Base class:** `dpfl/attacks/base_attack.py` — cần implement `perturb(honest_update) → Tensor`.

**Nếu attack cần tham số mới** (ngoài `scale_factor`), phải thêm field vào `AttackConfig` trong `config.py` và cập nhật logic khởi tạo trong `__main__.py`.

**Bước 1 — Tạo file:**
```python
# dpfl/attacks/noise_attack.py
import torch

from dpfl.attacks.base_attack import BaseAttack
from dpfl.registry import register, ATTACKS


@register(ATTACKS, "noise")
class NoiseAttack(BaseAttack):
    def __init__(self, noise_scale: float = 1.0):
        self.noise_scale = noise_scale

    def perturb(self, honest_update: torch.Tensor) -> torch.Tensor:
        return honest_update + torch.randn_like(honest_update) * self.noise_scale
```

**Bước 2 — Đăng ký import** trong `dpfl/__main__.py`:
```python
import dpfl.attacks.noise_attack  # noqa: F401
```

**Bước 3 — Thêm config field** trong `dpfl/config.py`:
```python
@dataclass
class AttackConfig:
    type: str = "scale"
    scale_factor: float = 3.0
    noise_scale: float = 1.0       # ← thêm field mới
```

Cập nhật khởi tạo trong `dpfl/__main__.py`:
```python
# Trước (chỉ hỗ trợ scale):
attack = ATTACKS[config.attack.type](scale_factor=config.attack.scale_factor)

# Sau (truyền toàn bộ params, mỗi class tự nhận kwargs cần thiết):
attack_kwargs = {"scale_factor": config.attack.scale_factor,
                 "noise_scale": config.attack.noise_scale}
attack = ATTACKS[config.attack.type](**attack_kwargs)
```

> Hoặc giữ đơn giản: mỗi attack class nhận `**kwargs` và lấy param mình cần.

**Bước 4 — YAML:**
```yaml
attack:
  type: noise
  noise_scale: 2.0
```

---

### Thêm Noise Mechanism Mới

**Base class:** `dpfl/privacy/base_noise_mechanism.py`

```python
class BaseNoiseMechanism(ABC):
    @abstractmethod
    def clip(self, per_sample_grads: torch.Tensor, clip_bound: float) -> torch.Tensor:
        """Clip gradients. Input shape:
        - per_step mode: (B, D) → (B, D)
        - post_training mode: (1, D) → (1, D)  (trainer unsqueeze trước khi gọi)
        """

    @abstractmethod
    def add_noise(self, avg_grad: torch.Tensor, clip_bound: float,
                  noise_mult: float, batch_size: int) -> torch.Tensor:
        """Add noise to gradient. Input shape: (D,) → (D,).
        - per_step mode: batch_size = actual batch size
        - post_training mode: batch_size = 1 (sensitivity = C cho 1 update)
        """

    def clip_and_noise(self, per_sample_grads, clip_bound, noise_mult, batch_size):
        """Template method: clip → average → noise. Chỉ dùng cho per_step mode."""
```

**Tham khảo:** `dpfl/privacy/gaussian_mechanism.py`

**Trainer gọi noise mechanism ở đâu:**
- `per_step` → `_train_dpsgd_per_step()`: `clip_and_noise(grads, C, z, B)` hoặc `clip(grads, C)`
- `post_training` → `_apply_post_training_dp()`: `clip(update.unsqueeze(0), C).squeeze(0)` rồi `add_noise(clipped, C, z, batch_size=1)`

**Ví dụ — Laplace mechanism:**

```python
# dpfl/privacy/laplace_mechanism.py
import torch

from dpfl.privacy.base_noise_mechanism import BaseNoiseMechanism
from dpfl.registry import register, NOISE_MECHANISMS


@register(NOISE_MECHANISMS, "laplace")
class LaplaceMechanism(BaseNoiseMechanism):

    def clip(self, per_sample_grads: torch.Tensor, clip_bound: float) -> torch.Tensor:
        norms = per_sample_grads.norm(2, dim=1, keepdim=True)
        clip_factors = torch.clamp(clip_bound / (norms + 1e-12), max=1.0)
        return per_sample_grads * clip_factors

    def add_noise(self, avg_grad: torch.Tensor, clip_bound: float,
                  noise_mult: float, batch_size: int) -> torch.Tensor:
        noise_scale = noise_mult * clip_bound / batch_size
        noise = torch.distributions.Laplace(0, noise_scale).sample(avg_grad.shape)
        return avg_grad + noise.to(avg_grad.device)
```

**Đăng ký import** trong `dpfl/__main__.py`:
```python
import dpfl.privacy.laplace_mechanism  # noqa: F401
```

**Lưu ý:** Hiện tại noise mechanism key đang **hardcode** `"gaussian"` trong `__main__.py` (line 47). Để dùng key khác từ config, cần sửa:

1. Thêm field `mechanism: str = "gaussian"` vào `DPConfig` trong `config.py`
2. Đổi `__main__.py`:
```python
# Trước:
noise_mechanism = NOISE_MECHANISMS["gaussian"]()
# Sau:
noise_mechanism = NOISE_MECHANISMS[config.dp.mechanism]()
```

**YAML:**
```yaml
dp:
  mechanism: laplace   # ← key đã register
```

---

### Thêm Accountant Mới

**Base class:** `dpfl/privacy/base_accountant.py`

```python
class BaseAccountant(ABC):
    @abstractmethod
    def step(self, n_steps: int, sampling_rate: float, noise_mult: float):
        """Accumulate privacy cost cho 1 round."""

    @abstractmethod
    def get_epsilon(self) -> float:
        """Return current cumulative epsilon."""

    def get_best_alpha(self) -> Optional[float]:
        """Return best Renyi order. None cho non-RDP accountants."""
        return None
```

**Tham khảo:** `dpfl/privacy/renyi_dpsgd.py`

**Simulator gọi:** `accountant.step(n_steps, q_batch, noise_mult)` mỗi round. Trong đó:
- `n_steps` = số batch iterations từ trainer
- `q_batch` = batch_size / n_samples
- `noise_mult` = từ config

**Nguyên tắc thiết kế:** Simulator **không biết** noise_mode. Nó luôn truyền cùng 3 params. **Accountant tự quyết định** cách dùng params dựa trên phương pháp tiêm nhiễu.

Ví dụ: accountant cho `post_training` sẽ bỏ qua `n_steps` và `sampling_rate` vì chỉ có 1 lần tiêm nhiễu mỗi round, không có subsampling.

**Ví dụ — Accountant cho post-training (output perturbation):**

```python
# dpfl/privacy/renyi_post_training_accountant.py
import math
from typing import List

from dpfl.privacy.base_accountant import BaseAccountant
from dpfl.registry import register, ACCOUNTANTS


@register(ACCOUNTANTS, "renyi_post_training")
class RenyiPostTrainingAccountant(BaseAccountant):
    """
    Post-training: 1 Gaussian noise event per round, no subsampling.
    eps_round(alpha) = alpha / (2 * z^2)
    """

    def __init__(self, alpha_list: List[float], delta: float):
        self.alpha_list = alpha_list
        self.delta = delta
        self.eps_rdp = {a: 0.0 for a in alpha_list}
        self.total_rounds = 0

    def step(self, n_steps: int, sampling_rate: float, noise_mult: float):
        """1 noise event per round. Ignore n_steps and sampling_rate."""
        for a in self.alpha_list:
            cost = a / (2.0 * noise_mult ** 2)
            self.eps_rdp[a] += cost
        self.total_rounds += 1

    def get_epsilon(self) -> float:
        return min(
            self.eps_rdp[a] + math.log(1.0 / self.delta) / (a - 1.0)
            for a in self.alpha_list
        )

    def get_best_alpha(self) -> float:
        return min(
            self.alpha_list,
            key=lambda a: self.eps_rdp[a] + math.log(1.0 / self.delta) / (a - 1.0)
        )
```

**Đăng ký import** trong `dpfl/__main__.py`:
```python
import dpfl.privacy.renyi_post_training_accountant  # noqa: F401
```

**Không cần sửa `config.py`** — field `accountant: str` đã tồn tại, chỉ đổi value trong YAML:
```yaml
dp:
  noise_mode: post_training
  accountant: renyi_post_training
  accountant_params:
    alpha_list: [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100]
```

**Khi `noise_mode: none`:** `__main__.py` tự skip tạo accountant, simulator nhận `accountant=None` và bỏ qua privacy accounting.

---

### Thêm Aggregator Mới

**Base class:** `dpfl/aggregation/base_aggregator.py`
**Cần implement:** `aggregate()` trả về `AggregationResult`

#### Interface chi tiết

```python
class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        own_update: torch.Tensor,       # (D,) — update vector của node hiện tại
        own_params: torch.Tensor,        # (D,) — model params hiện tại của node
        neighbor_updates: Dict[int, torch.Tensor],  # {neighbor_id: (D,)} — updates từ hàng xóm
    ) -> AggregationResult:
        ...
```

**Input giải thích:**

| Parameter | Shape | Ý nghĩa |
|-----------|-------|---------|
| `own_update` | `(D,)` | Gradient update đã qua DP (hoặc plain SGD tùy noise_mode) của node gọi aggregate. D = tổng số params của model |
| `own_params` | `(D,)` | Flat vector params hiện tại của model node đó (trước khi aggregate) |
| `neighbor_updates` | `{int: (D,)}` | Dict mapping neighbor_id → update vector. Có thể chứa attacker updates |

**Output — `AggregationResult`:**

```python
@dataclass
class AggregationResult:
    new_params: torch.Tensor              # (D,) — params mới sau aggregation
    clean_ids: List[int]                  # IDs hàng xóm KHÔNG bị flag
    flagged_ids: List[int]                # IDs hàng xóm bị flag là attacker
    metrics: Dict[str, Any]               # Internal data cho aggregator (optional)
    node_metrics: Dict[str, float]        # Scalar metrics cho logging — simulator auto-discover
```

| Field | Bắt buộc | Ý nghĩa |
|-------|----------|---------|
| `new_params` | **Có** | Params mới = kết quả aggregation. Simulator gọi `node.model.set_flat_params(result.new_params)` |
| `clean_ids` | Có | Danh sách neighbor IDs được coi là sạch. Dùng tính detection metrics (TP/FP/FN/TN) |
| `flagged_ids` | Có | Danh sách neighbor IDs bị flag. `clean_ids + flagged_ids` = tất cả neighbors |
| `metrics` | Optional | Data nội bộ aggregator, dict/list lồng nhau OK |
| `node_metrics` | Optional | **Dict scalar float** — simulator tự discover và aggregate mean honest/attacker. Ví dụ: `{"kurtosis": -0.02}` → tracker nhận `kurtosis_honest`, `kurtosis_attacker` tự động |

**Constructor:** Luôn nhận `param_dim: int` (số chiều model, do `__main__.py` truyền). Các params khác từ `config.aggregation.params` dict.

#### Ví dụ: Trimmed Mean Aggregator

**Bước 1 — Tạo file:**
```python
# dpfl/aggregation/trimmed_mean_aggregator.py
import torch
from typing import Dict

from dpfl.aggregation.base_aggregator import BaseAggregator, AggregationResult
from dpfl.registry import register, AGGREGATORS


@register(AGGREGATORS, "trimmed_mean")
class TrimmedMeanAggregator(BaseAggregator):
    def __init__(self, param_dim: int, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio

    def aggregate(self, own_update: torch.Tensor, own_params: torch.Tensor,
                  neighbor_updates: Dict[int, torch.Tensor]) -> AggregationResult:
        all_updates = torch.stack(list(neighbor_updates.values()))  # (N, D)
        k = int(len(neighbor_updates) * self.trim_ratio)
        sorted_updates, _ = all_updates.sort(dim=0)
        trimmed = sorted_updates[k:len(neighbor_updates) - k].mean(dim=0)

        new_params = own_params + own_update + trimmed
        return AggregationResult(
            new_params=new_params,
            clean_ids=list(neighbor_updates.keys()),
            flagged_ids=[],
            metrics={"trim_ratio": self.trim_ratio},
            node_metrics={},
        )
```

**Bước 2 — Đăng ký import** trong `dpfl/__main__.py`:
```python
import dpfl.aggregation.trimmed_mean_aggregator  # noqa: F401
```

**Bước 3 — Config:** Không cần sửa `config.py`. Chỉ đổi YAML:
```yaml
aggregation:
  type: trimmed_mean
  params:
    trim_ratio: 0.2
```

> `__main__.py` gọi: `AGGREGATORS["trimmed_mean"](param_dim=..., trim_ratio=0.2)`
> `dfl_simulator.py` **KHÔNG cần sửa** — hoàn toàn aggregator-agnostic.

---

### Tổng Hợp: Đăng Ký Config Cho Component Mới

| Component | Registry | Config key | Cần sửa `config.py`? | Cần sửa `__main__.py`? |
|-----------|----------|------------|----------------------|------------------------|
| Dataset | `DATASETS` | `dataset.name` | Không | Không (chỉ thêm import) |
| Model | `MODELS` | `model.name` | Không | Không (chỉ thêm import) |
| Attack | `ATTACKS` | `attack.type` | Có, nếu cần param mới | Có, nếu cần param mới |
| Noise Mechanism | `NOISE_MECHANISMS` | Hardcode `"gaussian"` | Có (`dp.mechanism`) | Có (đọc từ config) |
| Accountant | `ACCOUNTANTS` | `dp.accountant` | Không | Không (chỉ thêm import) |
| Aggregator | `AGGREGATORS` | `aggregation.type` | Không (dùng `params` dict) | Không (chỉ thêm import) |

### Checklist Mở Rộng

1. [ ] Tạo file `.py` trong thư mục tương ứng (`data/`, `models/`, `attacks/`, `aggregation/`, `privacy/`)
2. [ ] Kế thừa base class + implement tất cả abstract methods
3. [ ] Dùng `@register(REGISTRY, "key")` decorator
4. [ ] Thêm import vào `dpfl/__main__.py`: `import dpfl.xxx.yyy  # noqa: F401`
5. [ ] (Nếu cần param mới) Thêm field vào dataclass trong `dpfl/config.py` + cập nhật logic khởi tạo trong `__main__.py`
6. [ ] Cập nhật `config.yaml` với key + params
7. [ ] Chạy test: `python run.py config.yaml`

---

### Ví Dụ Hoàn Chỉnh: Setup post_training Experiment

Khi muốn chạy thí nghiệm `post_training` với Laplace noise + custom accountant + norm-based defense:

**1. Tạo 3 files mới:**
- `dpfl/privacy/laplace_mechanism.py` — Laplace noise (kế thừa `BaseNoiseMechanism`)
- `dpfl/privacy/renyi_post_training_accountant.py` — Accountant cho output perturbation (kế thừa `BaseAccountant`)
- `dpfl/aggregation/norm_filter_avg_aggregator.py` — Norm-based defense (kế thừa `BaseAggregator`)

**2. Register imports** trong `dpfl/__main__.py`:
```python
import dpfl.privacy.laplace_mechanism                    # noqa: F401
import dpfl.privacy.renyi_post_training_accountant       # noqa: F401
import dpfl.aggregation.norm_filter_avg_aggregator       # noqa: F401
```

**3. Config YAML:**
```yaml
dp:
  noise_mode: post_training
  mechanism: laplace                    # noise mechanism mới
  clip_bound: 2.0
  noise_mult: 1.1
  accountant: renyi_post_training       # accountant mới
  accountant_params:
    alpha_list: [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100]

aggregation:
  type: norm_filter_avg                 # aggregator mới
  params:
    norm_threshold: 10.0
```

**4. Chạy:** `python run.py config.yaml`

**Lưu ý:**
- Noise mechanism hiện hardcode `"gaussian"` → cần sửa `__main__.py` + thêm `mechanism` field vào `DPConfig` (1 lần duy nhất)
- Simulator và trainer **không cần sửa** — chúng agnostic với component cụ thể
- Accountant cho noise_mode mới **tự quyết** cách xử lý params từ simulator

## Thuật Toán DP-SGD (Tóm Tắt)

Mỗi round, mỗi node thực hiện:

```
1. Local training (E epochs):

   noise_mode = "per_step" (DP-SGD chuẩn):
     Mỗi mini-batch B samples:
       a. Tính per-sample gradient (vmap): (B, D)
       b. L2-clip từng sample: ||g̃ₛ||₂ ≤ C
       c. Average: ḡ = (1/B) Σ g̃ₛ       → sensitivity = C/B
       d. Noise:  ḡ + N(0, (z·C/B)²·I)
       e. SGD update

   noise_mode = "post_training":
     Mỗi mini-batch B samples:
       a. Standard forward + backward
       b. SGD update (no clip, no noise)
     Sau tất cả epochs:
       c. L2-clip final update: ||Δw||₂ ≤ C
       d. Noise: Δw + N(0, (z·C)²·I)

   noise_mode = "none":
     Mỗi mini-batch B samples:
       a. Standard forward + backward
       b. SGD update (no clip, no noise)

2. Exchange updates với hàng xóm

3. Defense detection:
   Ví dụ kurtosis: K(Δw̃ⱼ) = (1/D) Σ (Δw̃ⱼ[i]/RMS)⁴ - 3
   |K| > T_k = 1.96·√(24/D)  →  flagged

4. Aggregate từ clean neighbors only

5. Privacy accounting (tùy accountant):
   per_step RDP:  ε_step(α) = q²·α / (2z²),  q = B/n_local
   post_training: ε_round(α) = α / (2z²)
   none:          skip (ε = 0)
```

Xem chi tiết: `docs/dpsgd-dfl-pseudocode.md`
