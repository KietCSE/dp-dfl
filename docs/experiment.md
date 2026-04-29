# Plan Thực nghiệm — Trust-Aware DFL (Final Minimal)

**Phạm vi**: Phần 1 (Privacy-Utility, no attack) + Phần 2 (Robustness)
**Datasets**: MNIST, FEMNIST
**Mục tiêu**: Báo cáo nhanh — minimal metrics, focused outputs.

---

## Setup chung

| Tham số | Value |
|---------|-------|
| Nodes | 30 (MNIST), 50 (FEMNIST) |
| Topology | Erdős-Rényi p=0.3 |
| Rounds | 50 |
| Local epochs | 1 |
| Seeds | 3 (1, 42, 2024) |
| Privacy granularity | User-level Local DP |
| δ | 1e-5 |
| Clipping bound C | 1.0 (MNIST), 2.0 (FEMNIST) |
| Privacy accountant | Opacus RDP + tight conversion |

---

## PHẦN 1: Privacy-Utility (No Attack)

### 1.1 Mục tiêu

So sánh accuracy của Trust-Aware với baselines (DP-FedAvg-LDP, UDP-DFL) trong môi trường LDP user-level + DFL **không có Byzantine attack**.

### 1.2 Methods

| # | Method | Reference |
|---|--------|-----------|
| 1 | DP-FedAvg-LDP | McMahan 2018, adapted to LDP |
| 2 | UDP-DFL | Wei 2022, adapted to DFL |
| 3 | **Trust-Aware (Ours)** | This work |

### 1.3 Setup

| Tham số | Values |
|---------|--------|
| ε sweep | {2, 8} |
| Byzantine fraction | 0% |
| Datasets | MNIST, FEMNIST |
| Seeds | 3 |
| **Total runs** | 3 × 2 × 2 × 3 = **36 runs** |

### 1.4 Outputs

#### Table 1: Final Accuracy

```
─────────────────────────────────────────────────────────────────
Method           | MNIST ε=2 | MNIST ε=8 | FEMNIST ε=2 | FEMNIST ε=8
─────────────────────────────────────────────────────────────────
DP-FedAvg-LDP    | xx.x±x.x  | xx.x±x.x  | xx.x±x.x    | xx.x±x.x
UDP-DFL          | xx.x±x.x  | xx.x±x.x  | xx.x±x.x    | xx.x±x.x
**Ours**         | xx.x±x.x  | xx.x±x.x  | xx.x±x.x    | xx.x±x.x
─────────────────────────────────────────────────────────────────
```

**Metric**: Final test accuracy (mean ± std across 3 seeds).

#### Figure 1: Accuracy Convergence over Rounds

- **Type**: Line chart
- **X-axis**: Communication round t (1 → 50)
- **Y-axis**: Test accuracy (%)
- **Lines**: 3 methods (DP-FedAvg-LDP, UDP-DFL, Ours)
- **Setup**: ε=4 fixed (chọn 1 ε để chart sạch sẽ)
- **Subplots**: 2 panels (a) MNIST, (b) FEMNIST
- **Expected**: Ours line trên cùng, converge nhanh hơn, accuracy cao hơn cuối training.

---

## PHẦN 2: Robustness against Byzantine

### 2.1 Mục tiêu

So sánh tính robust của Trust-Aware với defense baselines (Krum-LDP, Median-LDP) và no-defense baseline (DP-FedAvg-LDP) under Sign-flip attack với **multiple Byzantine fractions**.

### 2.2 Methods

| # | Method | Detection? |
|---|--------|------------|
| 1 | DP-FedAvg-LDP (no defense) | ❌ |
| 2 | Krum + LDP | ✅ |
| 3 | Median + LDP | ❌ (implicit) |
| 4 | **Trust-Aware (Ours)** | ✅ |

### 2.3 Setup

| Tham số | Values |
|---------|--------|
| ε | 4 (fixed) |
| Byzantine fraction f | **{0%, 10%, 20%, 30%}** |
| Attack type | Sign-flip only |
| Datasets | MNIST, FEMNIST |
| Seeds | 3 |
| **Total runs** | 4 × 4 × 2 × 3 = **96 runs** |

### 2.4 Outputs

#### Table 2: Accuracy under Sign-flip Attack

```
MNIST (ε=4):
─────────────────────────────────────────────────
Method            | f=0%      | f=10%     | f=20%     | f=30%
─────────────────────────────────────────────────
DP-FedAvg-LDP     | xx.x±x.x  | xx.x±x.x  | xx.x±x.x  | xx.x±x.x
Krum + LDP        | xx.x±x.x  | xx.x±x.x  | xx.x±x.x  | xx.x±x.x
Median + LDP      | xx.x±x.x  | xx.x±x.x  | xx.x±x.x  | xx.x±x.x
**Ours**          | xx.x±x.x  | xx.x±x.x  | xx.x±x.x  | xx.x±x.x
─────────────────────────────────────────────────

FEMNIST (ε=4):
─────────────────────────────────────────────────
Method            | f=0%      | f=10%     | f=20%     | f=30%
─────────────────────────────────────────────────
DP-FedAvg-LDP     | xx.x±x.x  | xx.x±x.x  | xx.x±x.x  | xx.x±x.x
Krum + LDP        | xx.x±x.x  | xx.x±x.x  | xx.x±x.x  | xx.x±x.x
Median + LDP      | xx.x±x.x  | xx.x±x.x  | xx.x±x.x  | xx.x±x.x
**Ours**          | xx.x±x.x  | xx.x±x.x  | xx.x±x.x  | xx.x±x.x
─────────────────────────────────────────────────
```

**Metric**: Final test accuracy under various Byzantine fractions.

#### Figure 2: Accuracy Drop under Attack (Bar Chart)

- **Type**: Grouped bar chart
- **X-axis**: 4 methods (DP-FedAvg, Krum, Median, Ours)
- **Y-axis**: Accuracy (%)
- **Bars per method**: 2 bars — light (clean f=0%) vs dark (under attack f=20%)
- **Setup**: MNIST, ε=4 (chính); có thể thêm subplot FEMNIST nếu muốn
- **Insight**: Gap giữa 2 bars = mức độ bị attack ảnh hưởng. Ours có gap nhỏ nhất.

#### Figure 3: Detection Rate (TPR Bar Chart)

- **Type**: Grouped bar chart
- **X-axis**: 2 datasets (MNIST, FEMNIST)
- **Y-axis**: TPR (0 → 1.0)
- **Bars per dataset**: 2 bars — Krum + LDP vs Trust-Aware (Ours)
- **Setup**: f=20%, ε=4
- **Note**: Median + LDP và DP-FedAvg không có detection mechanism → loại khỏi figure này.
- **Insight**: Ours catch malicious nhiều hơn Krum.

---

## Tóm tắt Outputs

| Phần | Tables | Figures | Metrics |
|------|--------|---------|---------|
| Phần 1 | Table 1 (Final Accuracy) | Figure 1 (Convergence line chart) | Final accuracy |
| Phần 2 | Table 2 (Accuracy under attack, sweep f) | Figure 2 (Acc drop bars) + Figure 3 (TPR bars) | Acc + TPR |
| **Total** | **2 tables** | **3 figures** | **2 metrics** |

---

## Tóm tắt Runs

| Phần | Runs |
|------|------|
| Phần 1 | 36 |
| Phần 2 | 96 |
| **Total** | **132 runs** |

**Time estimate**: ~4-6 ngày với 1-2 GPUs.

---

## Cấu trúc báo cáo gợi ý

```
3. Experiments
   3.1 Setup (1 paragraph)
   3.2 Privacy-Utility (no attack)
       - Table 1
       - Figure 1
   3.3 Byzantine Robustness
       - Table 2
       - Figure 2 (Accuracy drop)
       - Figure 3 (Detection TPR)
```

Khoảng 2-3 trang results section là đủ.

---

## Logging requirements (mỗi run)

Để generate được tables + figures, mỗi run cần log:

| Phần | Log per round | Log final |
|------|---------------|-----------|
| Phần 1 | accuracy(t), ε(t) | final_accuracy, final_ε |
| Phần 2 | accuracy(t) | final_accuracy, TPR (cho methods có detection) |

Save format: CSV mỗi run với columns `[round, accuracy, epsilon]`, plus summary JSON với `[final_acc, final_eps, tpr]`.

---

## Definition của TPR (cho Figure 3)

**TPR (True Positive Rate)** = số malicious nodes được flag đúng / tổng malicious nodes.

### Cho Krum + LDP

Tại mỗi round, Krum chọn 1 neighbor "best" → các neighbors khác = "rejected/flagged".

```
TPR = #(rounds j được Krum reject AND j thực sự malicious) / (T × #malicious)
```

Average qua tất cả rounds [1, T].

### Cho Trust-Aware (Ours)

Trust score `T_{i,j}^(t)` cập nhật qua EMA. Threshold `T_min = 0.3`:

```
TPR = #(j malicious AND T_{i,j} < T_min tại t = T_final) / #malicious
```

Đo tại round cuối T (sau khi trust scores ổn định).
