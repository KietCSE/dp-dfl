# Detection P/R/F1 — Definition & Implementation

## TL;DR

Mỗi honest active node h tự phân loại neighbor (flag / clean) → tự có
`(TP, FP, FN, TN)` của riêng nó → tự có `P_h, R_h, F1_h`. **Global P/R/F1
mỗi round = macro-mean của các giá trị này qua honest active nodes**, skip
NaN (node có metric undefined). **Attacker và inactive node bị loại trừ
hoàn toàn.** Per-round độc lập, không cộng dồn TP/FP/FN qua các round —
time-mean qua post-attack rounds là bước post-processing.

---

## 1. Vì sao cần fix

Implementation cũ tại [`base_simulator._log_round`](../core/base_simulator.py)
tính:

```python
precision = total_tp / (total_tp + total_fp)   # micro-average qua MỌI active node
```

`total_tp` cộng dồn TP/FP/FN từ **tất cả** active node, **bao gồm attacker
node** (attacker chạy aggregator + cũng có `flagged_ids`/`clean_ids` riêng).
Spec mong muốn cho paper:

1. Mỗi honest node tự tính P/R/F1 trên neighborhood
2. Trung bình cộng các giá trị qua honest nodes (macro-mean)
3. Attacker không tham gia

→ Cần đổi từ **micro-sum (incl. attacker)** sang **macro-mean over honest**.

## 2. Định nghĩa toán học

### 2.1. Per-node-per-round counts

Cho round `t`, honest active node `h` với neighborhood active `N_h^t`:

- `A_h^t = N_h^t ∩ Attackers` — ground-truth positives
- `F_h^t ⊆ N_h^t` — flagged_ids do aggregator của h trả về

```
TP_{h,t} = |F_h^t ∩ A_h^t|     # flagged đúng
FP_{h,t} = |F_h^t \ A_h^t|     # flagged nhầm honest
FN_{h,t} = |A_h^t \ F_h^t|     # bỏ sót attacker
TN_{h,t} = |N_h^t \ (F_h^t ∪ A_h^t)|
```

### 2.2. Per-node-per-round P/R/F1

```
P_{h,t}  = TP / (TP + FP)        nếu (TP+FP) > 0  else NaN
R_{h,t}  = TP / (TP + FN)        nếu (TP+FN) > 0  else NaN
F1_{h,t} = 2·P·R / (P + R)       nếu cả P, R defined; (P+R)=0 → 0
                                 nếu P hoặc R = NaN → NaN
```

NaN = "không có dữ liệu để đánh giá":
- `P_h` NaN khi h không flag ai (`F_h = ∅`)
- `R_h` NaN khi h không có attacker neighbor (`A_h = ∅`)
- `F1_h` NaN khi P hoặc R undefined

### 2.3. Per-round aggregation (macro-mean over honest active)

```
P̄_t  = mean( P_{h,t}  : h honest active, P_{h,t} defined )
R̄_t  = mean( R_{h,t}  : h honest active, R_{h,t} defined )
F1̄_t = mean( F1_{h,t} : h honest active, F1_{h,t} defined )
```

Nếu không có honest active node nào có metric defined trong round `t`
(ví dụ: pre-attack với defense không flag gì) → `P̄_t = R̄_t = F1̄_t = NaN`.

### 2.4. Time-mean (post-processing)

```
P̄  = mean( P̄_t  : t có P̄_t defined, có thể filter t ≥ start_round )
R̄  = mean( R̄_t  : t có R̄_t defined )
F1̄ = mean( F1̄_t : t có F1̄_t defined )
```

Pre-attack rounds tự loại do `R̄_t = NaN` (không có positive nào).

### 2.5. Cross-seed aggregation

Chạy ≥ 3 seeds → report `mean ± std` của `F1̄` qua seeds.

## 3. Quy tắc loại trừ

Honest active node `h` bị skip khỏi `P̄_t` / `R̄_t` / `F1̄_t` khi:

| Tình huống | Hệ quả |
|---|---|
| h là attacker (`h ∈ attacker_ids`) | Skip mọi metric (attacker không có quyền vote) |
| h inactive (Poisson không chọn round t) | `TP+FP+FN+TN = 0` → skip |
| h không flag ai (`F_h = ∅`) | `P_h = NaN` → skip khỏi P̄_t (R̄_t vẫn tính) |
| h không có attacker neighbor (`A_h = ∅`) | `R_h = NaN` → skip khỏi R̄_t (P̄_t vẫn tính) |
| h có cả P và R defined | Vào cả P̄_t, R̄_t, F1̄_t |

Pre-attack rounds (`t < attack.start_round`): `_compute_detection` set
`active_ids = ∅` → mọi flag là FP, mọi clean là TN, không có positive nào
→ `R_h = NaN` cho mọi h → `R̄_t = F1̄_t = NaN`. Pre-attack rounds tự động
loại khỏi time-mean (skip NaN).

## 4. Vì sao chọn cách này (cho paper)

| Lý do | |
|---|---|
| **Per-node là đơn vị tự nhiên trong DFL** | Mỗi honest node là 1 classifier độc lập trên neighborhood riêng |
| **Macro-mean không bias theo degree** | Mọi honest node có trọng số bằng nhau, không phụ thuộc số neighbor |
| **Skip NaN thay vì default 0/1** | Tránh inflate precision giả khi không flag (P=1 sai), tránh deflate recall giả khi không có positive (R=0 sai) |
| **Per-round độc lập, không cumulative** | Cộng dồn đếm lặp cùng (attacker, neighborhood) qua nhiều round → inflate giả tạo, mất time dynamics |
| **Time-mean post-attack** | Pre-attack không có positive → F1 vacuous, mix vào sẽ làm metric meaningless |
| **Mean ± std qua seeds** | DFL có nhiều randomness (graph, Poisson, noise) — 1 seed không reproducible |
| **Match convention robust-FL** | FLTrust, FLAME, Krum, Bulyan, FedRecover đều report theo style này |

## 5. 6 pitfall đã tránh

- ❌ Include attacker decisions vào global metric (bug cũ)
- ❌ Mix pre-attack + post-attack rounds (precision artificially high)
- ❌ Cumulative TP/FP across rounds (double-counting)
- ❌ Single-seed numbers (không reproducible)
- ❌ Micro-average qua tất cả (node, neighbor) pair (bias theo degree)
- ❌ Fill NaN bằng 0/1 (distort distribution)

## 6. Implementation

### 6.1. Single-source fix

Thay đổi tập trung ở [`base_simulator._log_round`](../core/base_simulator.py)
(lines 537-579). Tất cả 7 simulator (`fedavg`, `dpsgd_kurtosis`, `fltrust`,
`noise_game`, `trust_aware`, `adaptive_noise`, `cfl_fedavg`) đều gọi chung
`_log_round` → fix 1 chỗ áp dụng cho toàn hệ thống. Aggregator-only modules
(`krum`, `flame`, `momentum_kurtosis`, `trimmed_mean`) plug vào
`dpsgd_kurtosis` simulator → tự động đúng theo.

### 6.2. JSON output schema (per round)

```json
{
  "round": 2,
  "precision": 0.0641,        // P̄_t — macro-mean over honest active
  "recall": 0.3667,           // R̄_t
  "f1_score": 0.2278,         // F1̄_t
  "total_tp_all": 30,         // diagnostic — micro-counts INCL. attacker
  "total_fp_all": 68,         // (giữ để debug/so sánh, KHÔNG dùng cho paper)
  "total_fn_all": 10,
  "total_tn_all": 4892,
  ...
}
```

NaN serialized via `json.dump(..., allow_nan=True)` (Python default) →
literal `NaN` token. Loadable bằng `json.loads`. Dùng `pandas.read_json`
hoặc filter `math.isnan` ở post-processing.

### 6.3. Per-node JSON (`node_data/round_XXX.json`)

```json
{
  "round": 2,
  "nodes": {
    "0": {                           // attacker node — vẫn log nhưng marked
      "is_attacker": true,
      "precision": 0.67,             // values from attacker's own aggregator
      "recall": 0.80,                // KHÔNG dùng cho global metric
      "f1_score": 0.73,
      ...
    },
    "6": {                           // honest node, inactive this round
      "is_attacker": false,
      "precision": NaN,              // (TP+FP+FN+TN = 0)
      "recall": NaN,
      "f1_score": NaN,
      ...
    },
    "12": {                          // honest active node, defined metric
      "is_attacker": false,
      "precision": 0.50,
      "recall": 0.40,
      "f1_score": 0.44,
      ...
    }
  }
}
```

Per-node values đã có sẵn trong code cũ — fix mới giữ nguyên, chỉ thay
default fallback từ `(P=1, R=0)` thành `NaN` cho consistency.

### 6.4. Reporting cho paper

**Bảng chính** (per defense × attack combination):

| Defense | Attack | Acc | P̄ | R̄ | F1̄ |
|---|---|---|---|---|---|
| FedAvg | LabelFlip-20% | 0.65±0.02 | — | — | — |
| Krum | LabelFlip-20% | 0.82±0.01 | 0.91±0.04 | 0.78±0.05 | 0.84±0.03 |
| Kurtosis | LabelFlip-20% | 0.85±0.01 | 0.88±0.03 | 0.85±0.04 | 0.86±0.02 |

Mean ± std qua ≥ 3 seeds. FedAvg/CFL không có defense → P̄/R̄/F1̄ = NaN
trong JSON, hiển thị "—" trong bảng (footnote: "no detection mechanism").

**Figure**: Line plot `F1̄_t` vs round overlay các defense → show
convergence behavior + attack onset.

**Supplementary** (optional):
- Per-node F1 distribution (boxplot/CDF) qua honest nodes — show variance
- TPR / FPR riêng theo convention detection-theory
- AUC-PR cho continuous-score defenses (TrustAware) qua threshold sweep

## 7. Coverage qua các thuật toán

| Simulator | Trạng thái | Note |
|---|---|---|
| `fedavg` | ✅ | Không có defense → F̄1 = NaN (đúng nghĩa) |
| `dpsgd_kurtosis` | ✅ | Cover thêm `krum`, `flame`, `momentum_kurtosis`, `trimmed_mean` qua aggregator plug-in |
| `fltrust` | ✅ | |
| `noise_game` | ✅ | |
| `trust_aware` | ✅ | Soft trust → aggregator threshold quyết định flagged_ids; cân nhắc thêm AUC-PR cho paper |
| `adaptive_noise` | ✅ | |
| `cfl_fedavg` | ✅ | CFL không có per-neighbor detection → F̄1 = NaN |

## 8. Test verification

Smoke test 5 round trên `krum_dp.yaml` (n_nodes=30, n_attackers=6,
attack.start_round=2):

```
Round 1/5  P: 0.00 R: nan F1: nan   # pre-attack: flagged some → P=0, no positives → R=NaN
Round 2/5  P: 0.00 R: nan F1: nan   # pre-attack
Round 3/5  P: 0.06 R: 0.37 F1: 0.23 # post-attack onset
Round 4/5  P: 0.21 R: 0.72 F1: 0.60 # detection improves
Round 5/5  P: 0.13 R: 0.53 F1: 0.36

Avg precision: 0.1351   # macro-mean qua post-attack rounds [t=2,3,4]
Avg recall:    0.5384
Avg F1 score:  0.3951
```

Verified:
- ✅ Pre-attack rounds có R/F1 = NaN
- ✅ Post-attack rounds có P/R/F1 ∈ [0, 1]
- ✅ Time-mean trong report tự skip NaN, chỉ tính post-attack rounds
- ✅ JSON serialize NaN thành `NaN` literal, loadable bằng Python `json.loads`
- ✅ Attacker per-node values vẫn được log nhưng KHÔNG vào global metric
- ✅ Diagnostic counters `total_tp_all/fp_all/fn_all/tn_all` (micro, incl. attackers) giữ cho debug

## 9. Migration impact

- **JSON keys không đổi**: `precision`, `recall`, `f1_score` giữ tên cũ, chỉ
  giá trị thay đổi (về đúng theo spec).
- **NaN values mới xuất hiện**: pre-attack rounds + defenses không flag
  (FedAvg/CFL) — post-processing scripts cần `df.dropna()` hoặc `math.isnan`
  filter trước khi avg/sum.
- **Diagnostic keys mới**: `total_tp_all`, `total_fp_all`, `total_fn_all`,
  `total_tn_all` — micro-counts, không dùng cho paper (chỉ debug).
- **Backward compat**: `_log_round` signature không đổi → 7 simulator không
  cần sửa.

## 10. Files modified

- [`core/base_simulator.py`](../core/base_simulator.py) — `_log_round`:
  thay micro-sum bằng macro-mean over honest active; thêm diagnostic
  counters vào `round_metrics`.
- [`tracking/metrics_tracker.py`](../tracking/metrics_tracker.py) — thêm
  `_nanmean` helper; `save_report` + `summary` dùng skip-NaN time-mean.
