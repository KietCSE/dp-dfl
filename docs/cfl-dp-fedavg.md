# CDP-FedAvg: Central Differential Privacy Federated Averaging

> **Mã giả chi tiết và phân tích thuật toán**
> Baseline cho nghiên cứu DP trong Decentralized Federated Learning (DFL)
> Dựa trên McMahan et al. (2017) "Learning Differentially Private Recurrent Language Models" với accountant Rényi DP hiện đại

---

## 1. Tổng quan

**CDP-FedAvg** (Central Differential Privacy FedAvg) là một thuật toán Federated Learning với đảm bảo **user-level $(\epsilon, \delta)$-DP**, trong đó:

- **"Central"** nghĩa là nhiễu được thêm **tại server** sau khi tổng hợp các update từ client
- Server được giả định là **trusted curator** — khác với UDP (Local DP) nơi nhiễu được thêm tại client
- Sensitivity được tính ở **mức user** thông qua việc clip user update $\Delta_k$
- Privacy accounting sử dụng **Rényi Differential Privacy (RDP)** để có bound chặt hơn Strong Composition

### So sánh với UDP

| Khía cạnh | **UDP (Local DP)** | **CDP-FedAvg (Central DP)** |
|-----------|-------------------|----------------------------|
| Vị trí thêm nhiễu | Tại client (trước upload) | Tại server (sau aggregation) |
| Trust assumption | Server không đáng tin (honest-but-curious) | Server đáng tin |
| Privacy budget | Mỗi client $(\epsilon_i, \delta_i)$ riêng | Toàn hệ thống $(\epsilon, \delta)$ chung |
| Đối tượng nhiễu | Tham số $w_i^{t+1}$ | Aggregate update $\Delta^{t+1}$ |
| Mức nhiễu cần thiết | Lớn hơn (mỗi client) | Nhỏ hơn (chỉ một lần ở server) |
| Utility | Thấp hơn | Cao hơn |

---

## 2. Ký hiệu (Notation)

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $K$ | Tổng số user (client) trong hệ thống |
| $T$ | Số communication rounds |
| $q \in (0, 1]$ | User sampling probability — xác suất mỗi user được chọn mỗi vòng |
| $\mathcal{C}^t$ | Tập user được chọn ở vòng $t$ (active clients) |
| $n_k$ | Số sample trong dataset cục bộ của user $k$ |
| $\hat{w}$ | Per-user example cap — ngưỡng giới hạn ảnh hưởng của user |
| $w_k = \min(n_k/\hat{w}, 1)$ | Trọng số của user $k$ (cap tại 1) |
| $W = \sum_{k} w_k$ | Tổng trọng số của tất cả user |
| $\theta^t$ | Tham số mô hình toàn cục ở vòng $t$ |
| $\Delta_k^{t+1}$ | User update của client $k$ (đã được clip) |
| $S$ | Clipping threshold (bound trên $\|\Delta_k\|$) |
| $z$ | Noise scale (noise multiplier) |
| $\sigma$ | Standard deviation của Gaussian noise |
| $\mathcal{M}$ | Rényi DP accountant |
| $(\epsilon, \delta)$ | Privacy budget toàn cục |

---

## 3. Mã giả thuật toán CDP-FedAvg

### Algorithm: CDP-FedAvg với Rényi DP Accountant

```
═══════════════════════════════════════════════════════════════════════
ALGORITHM: CDP-FedAvg (Central DP FedAvg with Rényi Accountant)
═══════════════════════════════════════════════════════════════════════

INPUT:
  • K users với datasets {D_1, D_2, ..., D_K}, |D_k| = n_k
  • Số communication rounds T
  • User sampling probability q ∈ (0, 1]
  • Per-user example cap ŵ ∈ ℝ⁺
  • Noise scale z ∈ ℝ⁺
  • Clipping threshold S (sensitivity bound)
  • Local learning rate η_l, server learning rate η_s
  • Local epochs E, local batch size B
  • Target privacy budget (ε_target, δ_target)

OUTPUT:
  • Mô hình toàn cục θ^T
  • Tổng privacy spent (ε_spent, δ_target)

═══════════════════════════════════════════════════════════════════════
PHASE 1: KHỞI TẠO (INITIALIZATION)
═══════════════════════════════════════════════════════════════════════

 1: Server khởi tạo tham số mô hình θ^0
 2: Server khởi tạo Rényi DP accountant M
       M ← RDPAccountant(orders=[2, 3, 5, 8, 16, 32, 64, 128])
 3: FOR each user k ∈ {1, 2, ..., K} DO
 4:     w_k ← min(n_k / ŵ, 1)        // Tính trọng số user (cap tại 1)
 5: END FOR
 6: W ← Σ_{k=1}^{K} w_k                // Tổng trọng số toàn hệ thống
 7: Server broadcast θ^0 đến tất cả user

═══════════════════════════════════════════════════════════════════════
PHASE 2: VÒNG LẶP HUẤN LUYỆN CHÍNH (MAIN TRAINING LOOP)
═══════════════════════════════════════════════════════════════════════

 8: FOR t = 0, 1, 2, ..., T-1 DO

       ───────────────────────────────────────────────────────────────
       BƯỚC 2.1: USER SAMPLING (Poisson Subsampling)
       ───────────────────────────────────────────────────────────────
 9:    C^t ← ∅
10:    FOR each user k ∈ {1, 2, ..., K} DO
11:        Sample u_k ~ Uniform(0, 1)
12:        IF u_k < q THEN
13:            C^t ← C^t ∪ {k}        // User k được chọn vào vòng t
14:        END IF
15:    END FOR
       // Lưu ý: Poisson sampling rất quan trọng cho privacy amplification
       // |C^t| là biến ngẫu nhiên với E[|C^t|] = qK

       ───────────────────────────────────────────────────────────────
       BƯỚC 2.2: LOCAL TRAINING (Parallel cho mỗi user trong C^t)
       ───────────────────────────────────────────────────────────────
16:    Server broadcast θ^t đến tất cả user k ∈ C^t
17:    FOR each user k ∈ C^t IN PARALLEL DO
18:        Δ_k^{t+1} ← UserUpdate(k, θ^t, ClipFn)
19:    END FOR

       ───────────────────────────────────────────────────────────────
       BƯỚC 2.3: WEIGHTED AGGREGATION TẠI SERVER
       ───────────────────────────────────────────────────────────────
20:    // Sử dụng fixed-denominator estimator f̃_f
21:    Δ^{t+1} ← (Σ_{k ∈ C^t} w_k · Δ_k^{t+1}) / (q · W)

       // GIẢI THÍCH MẪU SỐ qW:
       // - Phép chia cho qW (KHÔNG phải Σw_k thực tế) là chìa khóa của CDP
       // - qW = E[Σ_{k∈C^t} w_k] là kỳ vọng của tổng trọng số mẫu
       // - Mẫu số CỐ ĐỊNH này giúp sensitivity bị bound rõ ràng
       // - Trade-off: estimator có bias nhỏ nhưng controllable

       ───────────────────────────────────────────────────────────────
       BƯỚC 2.4: GAUSSIAN NOISE INJECTION
       ───────────────────────────────────────────────────────────────
22:    σ ← (z · S) / (q · W)
       // Phân tích sensitivity:
       // - Mỗi user k đóng góp tối đa w_k · S ≤ 1 · S = S vào tử số
       // - Sensitivity của Δ^{t+1} đối với việc thêm/bớt 1 user là:
       //     L2-sensitivity = S / (qW)
       // - Để đạt Gaussian Mechanism với noise multiplier z:
       //     σ = z × sensitivity = zS/(qW)

23:    n^{t+1} ~ N(0, σ² · I)         // Sample Gaussian noise vector
24:    Δ̃^{t+1} ← Δ^{t+1} + n^{t+1}    // Thêm nhiễu vào aggregate update

       ───────────────────────────────────────────────────────────────
       BƯỚC 2.5: GLOBAL MODEL UPDATE
       ───────────────────────────────────────────────────────────────
25:    θ^{t+1} ← θ^t + η_s · Δ̃^{t+1}
       // η_s = 1 trong FedAvg gốc, có thể tune cho server learning rate

       ───────────────────────────────────────────────────────────────
       BƯỚC 2.6: PRIVACY ACCOUNTING (Rényi DP)
       ───────────────────────────────────────────────────────────────
26:    M.accumulate_privacy(noise_multiplier=z, sample_rate=q, steps=1)

27:    (ε_spent, δ_used) ← M.get_privacy_spent(target_delta=δ_target)

28:    PRINT "Round t: θ^{t+1} updated, ε_spent = {ε_spent}, δ = {δ_target}"

       ───────────────────────────────────────────────────────────────
       BƯỚC 2.7: EARLY STOPPING (TÙY CHỌN)
       ───────────────────────────────────────────────────────────────
29:    IF ε_spent ≥ ε_target THEN
30:        PRINT "Privacy budget exhausted at round t. Stopping."
31:        BREAK
32:    END IF

33: END FOR

34: RETURN θ^{t+1}, (ε_spent, δ_target)

═══════════════════════════════════════════════════════════════════════
```

---

## 4. Thủ tục con: UserUpdate (Local Training tại Client)

```
═══════════════════════════════════════════════════════════════════════
PROCEDURE: UserUpdate(k, θ^t, ClipFn)
═══════════════════════════════════════════════════════════════════════

INPUT:
  • User index k
  • Tham số toàn cục hiện tại θ^t
  • Hàm clipping ClipFn (FlatClip hoặc PerLayerClip)
  • Local epochs E, batch size B, local learning rate η_l

OUTPUT:
  • Clipped user update Δ_k^{t+1}

───────────────────────────────────────────────────────────────────────

 1: θ_k ← θ^t                          // Khởi tạo tham số cục bộ
 2: 
 3: FOR epoch e = 1, 2, ..., E DO
 4:     // Chia D_k thành các batch ngẫu nhiên
 5:     B_k ← random_batches(D_k, batch_size=B)
 6:     FOR each batch b ∈ B_k DO
 7:         g ← ∇_{θ_k} F_k(b, θ_k)   // Tính gradient cục bộ
 8:         θ_k ← θ_k - η_l · g        // SGD update
 9:     END FOR
10: END FOR
11:
12: // Tính raw user update
13: Δ_k_raw ← θ_k - θ^t
14:
15: // Clipping: bound L2-norm của user update bởi S
16: Δ_k^{t+1} ← ClipFn(Δ_k_raw, S)
17:
18: RETURN Δ_k^{t+1}

═══════════════════════════════════════════════════════════════════════
```

### 4.1. FlatClip — Cắt theo toàn bộ vector

```
PROCEDURE: FlatClip(Δ, S)
───────────────────────────────────────────────────────────────────────
 1: norm ← ||Δ||_2
 2: scale ← min(1, S / norm)
 3: RETURN Δ · scale
───────────────────────────────────────────────────────────────────────
```

**Đặc điểm:** Toàn bộ vector $\Delta$ được scale chung. Đảm bảo $\|\Delta\|_2 \leq S$ chính xác.

### 4.2. PerLayerClip — Cắt theo từng lớp

```
PROCEDURE: PerLayerClip(Δ, S)
───────────────────────────────────────────────────────────────────────
 1: Tách Δ thành các lớp: Δ = [Δ_1, Δ_2, ..., Δ_L]
 2: Phân bổ ngân sách clipping: S_l = S · √(d_l / d_total)
       trong đó d_l là số tham số của lớp l
 3: FOR l = 1, 2, ..., L DO
 4:     Δ_l ← FlatClip(Δ_l, S_l)
 5: END FOR
 6: RETURN [Δ_1, Δ_2, ..., Δ_L]
───────────────────────────────────────────────────────────────────────
```

**Đặc điểm:** Mỗi lớp được clip riêng. Thường cho kết quả tốt hơn FlatClip vì các lớp có scale khác nhau.

---

## 5. Phân tích Privacy với Rényi DP Accountant

### 5.1. Rényi DP cho Subsampled Gaussian Mechanism

Với mỗi vòng $t$, cơ chế của chúng ta là:
$$\mathcal{M}_t(D) = \frac{1}{qW}\sum_{k \in \mathcal{C}^t} w_k \Delta_k + \mathcal{N}(0, \sigma^2 I)$$

Đây là **Subsampled Gaussian Mechanism (SGM)** với:
- Sampling rate: $q$
- Noise multiplier: $z = \sigma \cdot qW / S$
- Sensitivity: $\Delta = S/(qW)$

### 5.2. RDP Bound cho SGM (Mironov et al., 2019)

Với order $\alpha > 1$, RDP của SGM thỏa:
$$\epsilon_\alpha^{SGM}(z, q) \leq \frac{1}{\alpha - 1} \ln\left(\sum_{j=0}^{\alpha} \binom{\alpha}{j} (1-q)^{\alpha - j} q^j \exp\left(\frac{j(j-1)}{2z^2}\right)\right)$$

### 5.3. Composition Sau T Vòng

RDP composes **tuyến tính** theo order $\alpha$:
$$\epsilon_\alpha^{total} = T \cdot \epsilon_\alpha^{SGM}(z, q)$$

### 5.4. Chuyển từ RDP sang $(\epsilon, \delta)$-DP

Với $\delta_{target}$ cho trước, tìm:
$$\epsilon = \min_{\alpha > 1} \left( \epsilon_\alpha^{total} + \frac{\ln(1/\delta_{target})}{\alpha - 1} \right)$$

Tối ưu hóa qua tập các order $\alpha \in \{2, 3, 5, 8, 16, 32, 64, 128\}$ để tìm $\epsilon$ nhỏ nhất.

---

## 6. Implementation tham khảo bằng Python (sử dụng Opacus)

```python
"""
CDP-FedAvg Implementation với Rényi DP Accountant
Sử dụng thư viện Opacus (PyTorch) cho privacy accounting
"""

import torch
import numpy as np
from opacus.accountants import RDPAccountant


class CDPFedAvg:
    def __init__(self, model, num_users, T, q, w_hat, z, S,
                 target_epsilon, target_delta, eta_local, eta_server=1.0,
                 local_epochs=1, batch_size=32, clip_fn='flat'):
        """
        Khởi tạo CDP-FedAvg.

        Args:
            model: Mô hình PyTorch ban đầu
            num_users: Tổng số user K
            T: Số communication rounds
            q: User sampling probability
            w_hat: Per-user example cap
            z: Noise multiplier (noise scale)
            S: Clipping threshold
            target_epsilon, target_delta: Privacy budget mục tiêu
            eta_local, eta_server: Learning rates
            local_epochs, batch_size: Tham số huấn luyện cục bộ
            clip_fn: 'flat' hoặc 'per_layer'
        """
        self.model = model
        self.K = num_users
        self.T = T
        self.q = q
        self.w_hat = w_hat
        self.z = z
        self.S = S
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.eta_local = eta_local
        self.eta_server = eta_server
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.clip_fn = clip_fn

        # Khởi tạo Rényi DP accountant
        self.accountant = RDPAccountant()

    def compute_user_weights(self, user_datasets):
        """Tính trọng số w_k và W theo công thức w_k = min(n_k/ŵ, 1)."""
        self.user_weights = {}
        for k, dataset in enumerate(user_datasets):
            n_k = len(dataset)
            self.user_weights[k] = min(n_k / self.w_hat, 1.0)
        self.W = sum(self.user_weights.values())
        return self.user_weights, self.W

    def poisson_sample_users(self):
        """Poisson subsampling: mỗi user được chọn độc lập với xác suất q."""
        sampled = []
        for k in range(self.K):
            if np.random.uniform(0, 1) < self.q:
                sampled.append(k)
        return sampled

    def user_update(self, k, theta, dataset_k):
        """Local training cho user k và clip update."""
        # Sao chép tham số toàn cục
        local_model = self._clone_model(theta)
        optimizer = torch.optim.SGD(local_model.parameters(), lr=self.eta_local)

        # Local SGD
        for epoch in range(self.local_epochs):
            loader = torch.utils.data.DataLoader(
                dataset_k, batch_size=self.batch_size, shuffle=True
            )
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = self._compute_loss(local_model, batch_x, batch_y)
                loss.backward()
                optimizer.step()

        # Tính raw update
        delta_k_raw = self._param_diff(local_model, theta)

        # Clip update
        if self.clip_fn == 'flat':
            delta_k = self._flat_clip(delta_k_raw, self.S)
        else:  # per_layer
            delta_k = self._per_layer_clip(delta_k_raw, self.S)

        return delta_k

    def aggregate_and_perturb(self, deltas, sampled_users):
        """Aggregate user updates với fixed denominator và thêm Gaussian noise."""
        # Weighted sum
        weighted_sum = None
        for k in sampled_users:
            w_k = self.user_weights[k]
            if weighted_sum is None:
                weighted_sum = self._scale(deltas[k], w_k)
            else:
                weighted_sum = self._add(weighted_sum, self._scale(deltas[k], w_k))

        # Chia cho qW (fixed denominator estimator f̃_f)
        delta_aggregate = self._scale(weighted_sum, 1.0 / (self.q * self.W))

        # Tính sigma và thêm Gaussian noise
        sigma = (self.z * self.S) / (self.q * self.W)
        noise = self._sample_gaussian_noise(delta_aggregate, sigma)
        delta_perturbed = self._add(delta_aggregate, noise)

        return delta_perturbed

    def train(self, user_datasets):
        """Vòng lặp huấn luyện chính của CDP-FedAvg."""
        self.compute_user_weights(user_datasets)
        theta = self._clone_model(self.model)

        for t in range(self.T):
            # Bước 2.1: User sampling (Poisson)
            sampled_users = self.poisson_sample_users()

            if len(sampled_users) == 0:
                # Vẫn cần accumulate privacy ngay cả khi không có user
                self.accountant.step(noise_multiplier=self.z, sample_rate=self.q)
                continue

            # Bước 2.2: Local training song song
            deltas = {}
            for k in sampled_users:
                deltas[k] = self.user_update(k, theta, user_datasets[k])

            # Bước 2.3 + 2.4: Aggregate + thêm nhiễu
            delta_perturbed = self.aggregate_and_perturb(deltas, sampled_users)

            # Bước 2.5: Cập nhật mô hình toàn cục
            theta = self._update_model(theta, delta_perturbed, self.eta_server)

            # Bước 2.6: Privacy accounting với Rényi DP
            self.accountant.step(noise_multiplier=self.z, sample_rate=self.q)
            epsilon_spent = self.accountant.get_epsilon(delta=self.target_delta)

            print(f"Round {t+1}/{self.T}: ε_spent = {epsilon_spent:.4f}, "
                  f"δ = {self.target_delta}")

            # Bước 2.7: Early stopping nếu hết privacy budget
            if epsilon_spent >= self.target_epsilon:
                print(f"Privacy budget exhausted at round {t+1}. Stopping.")
                break

        return theta, epsilon_spent

    # ─── Các hàm phụ trợ (helper functions) ───────────────────────────
    def _clone_model(self, model):
        """Sao chép mô hình."""
        import copy
        return copy.deepcopy(model)

    def _param_diff(self, model_a, model_b):
        """Tính hiệu các tham số: model_a - model_b."""
        diff = {}
        for (name_a, p_a), (name_b, p_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            diff[name_a] = p_a.data - p_b.data
        return diff

    def _flat_clip(self, delta, S):
        """Flat clipping: scale toàn bộ vector."""
        total_norm_sq = sum((v ** 2).sum().item() for v in delta.values())
        total_norm = total_norm_sq ** 0.5
        scale = min(1.0, S / (total_norm + 1e-12))
        return {name: v * scale for name, v in delta.items()}

    def _per_layer_clip(self, delta, S):
        """Per-layer clipping với phân bổ ngân sách theo căn bậc hai số tham số."""
        d_total = sum(v.numel() for v in delta.values())
        clipped = {}
        for name, v in delta.items():
            d_l = v.numel()
            S_l = S * (d_l / d_total) ** 0.5
            norm_l = (v ** 2).sum().item() ** 0.5
            scale = min(1.0, S_l / (norm_l + 1e-12))
            clipped[name] = v * scale
        return clipped

    def _scale(self, delta, factor):
        """Nhân vô hướng: factor · delta."""
        return {name: v * factor for name, v in delta.items()}

    def _add(self, delta_a, delta_b):
        """Cộng hai dict tham số."""
        return {name: delta_a[name] + delta_b[name] for name in delta_a}

    def _sample_gaussian_noise(self, delta, sigma):
        """Sample Gaussian noise với cùng shape như delta."""
        return {
            name: torch.randn_like(v) * sigma for name, v in delta.items()
        }

    def _update_model(self, theta, delta, eta_s):
        """θ ← θ + η_s · Δ"""
        with torch.no_grad():
            for name, param in theta.named_parameters():
                param.data.add_(delta[name], alpha=eta_s)
        return theta

    def _compute_loss(self, model, x, y):
        """Hàm loss tùy chỉnh tùy task."""
        criterion = torch.nn.CrossEntropyLoss()
        output = model(x)
        return criterion(output, y)


# ────────────────────────────────────────────────────────────────────
# CÁCH SỬ DỤNG
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Khởi tạo mô hình
    model = YourPyTorchModel()  # Thay bằng mô hình thực tế

    # Khởi tạo CDP-FedAvg
    cdp_fedavg = CDPFedAvg(
        model=model,
        num_users=1000,
        T=100,
        q=0.01,           # Sample 1% user mỗi vòng
        w_hat=100,        # Per-user example cap
        z=1.0,            # Noise multiplier
        S=1.0,            # Clipping threshold
        target_epsilon=2.0,
        target_delta=1e-5,
        eta_local=0.01,
        eta_server=1.0,
        local_epochs=1,
        batch_size=32,
        clip_fn='flat',
    )

    # Train
    user_datasets = [...]  # List of K user datasets
    final_model, epsilon_spent = cdp_fedavg.train(user_datasets)

    print(f"Training complete. Final ε = {epsilon_spent}")
```

---

## 7. Các điểm then chốt cần lưu ý khi báo cáo trong paper

### 7.1. Điểm khác biệt chính giữa CDP-FedAvg và UDP

| Tiêu chí | UDP | CDP-FedAvg |
|----------|-----|-----------|
| **Threat model** | Server không đáng tin | Server đáng tin |
| **Vị trí nhiễu** | Client (local) | Server (central) |
| **Đối tượng nhiễu** | Tham số $w_i^{t+1}$ | Aggregate update $\Delta^{t+1}$ |
| **Privacy granularity** | Per-user $(\epsilon_i, \delta_i)$ | System-wide $(\epsilon, \delta)$ |
| **Sensitivity calculation** | $\Delta\ell$ của local training | $S/(qW)$ của aggregate |
| **Accounting** | RDP / Strong Composition | RDP (chuẩn modern) |
| **Utility** | Thấp hơn (nhiễu nhân theo K) | Cao hơn (1 lần nhiễu) |

### 7.2. Tại sao baseline này hợp lý

1. **Khác threat model với UDP** — không trùng lặp về bản chất
2. **Cùng level bảo vệ (user-level)** — so sánh fair với phương pháp DFL của bạn
3. **State-of-the-art accounting (RDP)** — không bị reviewer phê bình về tính hiện đại
4. **Established trong literature** — McMahan et al. 2017 là paper kinh điển

### 7.3. Cấu trúc baseline tổng thể đề xuất

```
┌─────────────────────────────────────────────────────────────┐
│  BASELINE STRUCTURE FOR YOUR DFL PAPER                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Group A: Non-private upper bound                            │
│    • FedAvg (centralized, no privacy)                        │
│                                                              │
│  Group B: Privacy baselines in centralized FL                │
│    • CDP-FedAvg (Central DP)  ← BASELINE 1                   │
│    • UDP (Local DP)            ← BASELINE 2                   │
│                                                              │
│  Group C: Decentralized (your contribution)                  │
│    • YOUR DFL METHOD                                         │
│                                                              │
│  Logic:                                                      │
│  • B vs A: Cost of privacy in centralized                    │
│  • CDP vs UDP: Trust assumption trade-off                    │
│  • C vs B: Benefit of decentralization                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Các tham số khuyến nghị (Hyperparameters)

| Tham số | Giá trị khuyến nghị | Ghi chú |
|---------|---------------------|---------|
| $q$ | 0.001 - 0.05 | Càng nhỏ càng tốt cho privacy (privacy amplification) |
| $\hat{w}$ | $\approx$ median$(n_k)$ | Cap nên gần median của số sample |
| $z$ | 0.5 - 2.0 | Noise multiplier; lớn = privacy mạnh |
| $S$ | 0.1 - 10 | Tune dựa trên grad norm thực tế |
| $T$ | 100 - 10,000 | Tùy task; nhiều round cần $z$ lớn hơn |
| $\delta$ | $\ll 1/K$ | Thường $10^{-5}$ hoặc $10^{-6}$ |
| $\alpha$ orders | $\{2, 3, 5, 8, 16, 32, 64, 128\}$ | Cho RDP accounting |

---

## 9. Tham khảo

1. **McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2017).** "Learning Differentially Private Recurrent Language Models." *ICLR 2018*. — Paper gốc của thuật toán này.

2. **Mironov, I. (2017).** "Rényi Differential Privacy." *CSF 2017*. — Định nghĩa Rényi DP.

3. **Mironov, I., Talwar, K., & Zhang, L. (2019).** "Rényi Differential Privacy of the Sampled Gaussian Mechanism." *arXiv:1908.10530*. — RDP bound cho SGM.

4. **Wang, Y. X., Balle, B., & Kasiviswanathan, S. P. (2019).** "Subsampled Rényi Differential Privacy and Analytical Moments Accountant." *AISTATS 2019*.

5. **Abadi, M., et al. (2016).** "Deep Learning with Differential Privacy." *CCS 2016*. — Moments Accountant gốc.

6. **Opacus Documentation.** https://opacus.ai/ — Thư viện implementation.

---

*Tài liệu này được viết để phục vụ baseline experiment trong nghiên cứu Decentralized Federated Learning với Differential Privacy. Bạn nên cite cả paper gốc McMahan 2017 và paper RDP của Mironov khi báo cáo kết quả CDP-FedAvg trong publication của mình.*