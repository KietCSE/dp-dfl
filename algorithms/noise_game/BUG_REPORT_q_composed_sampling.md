# Bug Report: REPORT.md §5 "Bug #6" — Công thức `q_composed = q_client · q_batch` Không Valid Cho Noise Game

**File ảnh hưởng**: [algorithms/noise_game/REPORT.md](REPORT.md)
**Sections ảnh hưởng**: §4 "Privacy Accounting", §5 "Bug Fixes Timeline" (Bug #6), bảng impact §5
**Mức độ**: Critical — gây mâu thuẫn doc-code, hiểu nhầm về tính rigorous của ε reported
**Status code**: Code ĐÃ correctly không áp dụng (commit `5d39ca7`); chỉ REPORT.md cần update

---

## 1. Tóm tắt

REPORT.md §5 Bug #6 đề xuất composing client-level và batch-level Poisson sub-sampling:

> q_composed = q_client · q_batch

và pass vào outer Rényi accountant để gain ~4× privacy amplification, với bảng impact:
> | Trước fix | 0.053 | ε round 4 ≈ 19 |
> | Sau fix | 0.0265 | ε round 4 ≈ 4–6 (~4× amplification) |

**Vấn đề**: Công thức này **chỉ valid cho 1-step DP-SGD per round** (Abadi et al. 2016 setup gốc — 1 batch sample + 1 noise inject mỗi step). noise_game là **DP-FedAvg multi-batch local training** (mỗi round, client chạy local SGD nhiều batch rồi inject noise MỘT LẦN ở cuối) — không thỏa mãn điều kiện áp dụng của sub-sampling amplification lemma (Mironov-Talwar 2019).

Áp dụng công thức này → ε reported nhỏ hơn ~14× so với true ε → **under-report privacy cost** → privacy claim **mất tính rigorous**, không publishable.

---

## 2. Bằng chứng lý thuyết

### 2.1 Điều kiện áp dụng sub-sampling amplification lemma

Mironov-Talwar 2019 (Sampled Gaussian Mechanism, định lý nền tảng cho Opacus RDP accountant) yêu cầu:

1. Mechanism `M(D')` áp dụng lên **MỘT** Poisson sample `D' ~ Poisson(D, q)` của data
2. Noise inject **NGAY SAU** sample trong cùng atomic step
3. Composition giữa các step phải tuần tự, mỗi step là một (sample → noise) đơn nhất

Khi compose hierarchical Poisson layers (client-level + batch-level), điều này yêu cầu mỗi atomic step là: 1 client → 1 batch → 1 gradient → 1 noise inject. Đây là setup **DP-FedSGD** (Abadi-style 1-step per round).

### 2.2 Setup thực tế của noise_game vi phạm điều kiện

Trích từ [algorithms/noise_game/simulator.py:53-86](simulator.py#L53-L86):

```python
# Phase 1: Local training (no noise — game handles it)
raw_updates, all_steps = self._train_all_nodes(apply_noise=False, round_t=t)
# ↑ Multi-batch local SGD KHÔNG inject noise per batch

# L2-norm clip each update
C = self.config.dp.clip_bound
clipped = {nid: upd * min(1.0, C / (upd.norm() + 1e-12))
           for nid, upd in raw_updates.items()}

# Phase 2: Inject noise MỘT LẦN per round
noise, metrics = self.game_mechanism.compute_total_noise(g, ...)
g_hat = g + noise
```

→ Mỗi round, mechanism áp dụng cho mỗi client là:
> M(client_data) = Clip(SGD_multi_batch(client_data)) + Gaussian_noise

→ Local SGD output là **deterministic function của TOÀN BỘ data** trên client (cho fixed init), không phải hàm của 1 batch random.

→ Batch-level Poisson sampling **bên trong** local SGD KHÔNG có amplification effect vì:
1. Không có noise inject sau mỗi batch → không có per-batch "amplified mechanism" để compose
2. Output cuối cùng (sau nhiều batch) phụ thuộc tất cả batches, không phải 1 batch random
3. Sub-sampling lemma yêu cầu Gaussian Mechanism áp dụng tight ngay sau sample — điều kiện này không match local SGD multi-batch

→ Layer Poisson sampling DUY NHẤT có ý nghĩa amplification là **client-level** (q_client), inject noise đúng một lần sau khi sample client active.

### 2.3 Setup nào thì công thức REPORT.md valid?

Công thức `q_composed = q_client · q_batch` chỉ valid trong **DP-FedSGD**:
- Mỗi round = 1 client active sample 1 batch
- Compute gradient trên batch đó
- Inject noise ngay
- Send to server

Đây KHÔNG phải setup noise_game.

---

## 3. Bằng chứng từ code: developer đã correctly nhận diện

Comment chính xác tại [algorithms/noise_game/simulator.py:165-171](simulator.py#L165-L171):

```python
# DP unit: node-level (user-level Local DP). Local SGD runs with
# apply_noise=False — no per-step DP guarantee inside training.
# Noise is injected ONCE per round post-aggregation, so
# accountant composition is per-round (steps=1), not per-minibatch.
# Sampling amplification comes only from Poisson client subsampling
# (q_client) — NOT from per-batch sampling, because the local update
# is a deterministic function of all data seen given fixed init.
```

Commit `5d39ca7` ("feat: implement per-node RDP accounting and baseline configuration updates for fair DP-FedAvg comparisons") chính thức **bỏ q_batch composition**.

Code hiện tại tại [simulator.py:185, 212](simulator.py#L185):

```python
q_client = max(min(float(self.config.dp.sampling_rate), 1.0), 0.0)
...
acc.step(1, q_client, eff_mult_node)  # CHỈ q_client, KHÔNG nhân q_batch
```

→ Code rigorous, đúng lý thuyết. Chỉ REPORT.md đang lạc hậu.

---

## 4. Tác động thực tế nếu áp dụng công thức REPORT.md

Default MNIST config (50 nodes, batch_size=64, samples=1200/node, sampling_rate=0.5):

| Số liệu | Code hiện tại (q=q_client=0.5) | Apply công thức REPORT.md (q_composed=0.0265) |
|---|---|---|
| q effective | 0.5 | 0.0265 |
| RDP/step ∝ α·q²/(2σ²) (q nhỏ approx) | ∝ 0.25 | ∝ 0.0007 |
| ε round 50 (báo cáo) | ~25-50 (rigorous) | ~2-4 (optimistic) |
| Tỉ lệ ε báo cáo | 1× | ~14× nhỏ hơn |
| True ε (privacy thực) | ~25-50 | **~25-50** (giống nhau!) |
| Privacy guarantee | ✓ Rigorous, publish-quality | ✗ Under-estimate, **không đảm bảo (ε,δ)-DP** |

**Kết luận**: Áp dụng công thức REPORT.md chỉ làm **báo cáo** ε nhỏ đi, KHÔNG làm **true privacy** tốt hơn. Đây là dạng "privacy laundering" — tạo cảm giác guarantee tốt mà không có cơ sở.

---

## 5. Khuyến nghị: Sửa REPORT.md (KHÔNG sửa code)

### 5.1 §4 "Privacy Accounting" — đổi paradigm sang node-level Local DP

**Hiện tại**:
```
threat model: item-level DP (mỗi data record là 1 unit)
q_composed = q_client · q_batch
```

**Đề xuất**:
```
Threat model: NODE-LEVEL (user-level) Local DP — mỗi client là 1 unit
privacy. Local SGD multi-batch là deterministic function của all
data on client → KHÔNG có amplification per-batch. Sub-sampling
chỉ áp dụng tại CLIENT layer (q_client = config.dp.sampling_rate).

q = q_client only
```

### 5.2 §5 Bug #6 — đổi thành REVERTED + giải thích lý thuyết

```markdown
### Bug #6 (REVERTED in commit 5d39ca7) — Item-level sampling composition không valid

**Claim ban đầu**: `q_composed = q_client · q_batch` để gain batch-level amplification.

**Status**: REVERTED. Lý do lý thuyết:

1. Sub-sampling amplification lemma (Mironov-Talwar 2019, Sampled Gaussian Mechanism)
   chỉ valid khi mechanism inject Gaussian noise NGAY SAU mỗi Poisson sample atomic step.

2. Setup noise_game là DP-FedAvg multi-batch:
   - Phase 1: local SGD chạy nhiều batch với apply_noise=False
   - Phase 2: inject noise MỘT LẦN per round sau khi train xong
   → Local update là deterministic function của all data on client
   → Batch-level Poisson sampling không có amplification effect

3. Composition q_client · q_batch chỉ đúng cho DP-FedSGD (Abadi 2016 1-step setup):
   - 1 round = 1 client × 1 batch × 1 gradient × 1 noise inject
   - noise_game KHÔNG match setup này

**Áp dụng công thức**:
- ε reported giảm ~14× (do q² amplification giả)
- True privacy KHÔNG thay đổi
- → under-report ε, vi phạm rigorous DP claim, không publishable

**Sau REVERT** (commit 5d39ca7):
- q = q_client only (node-level Local DP)
- Per-node RenyiAccountant track RDP độc lập cho mỗi client
- ε reported = max(per_node_eps) — worst-case honest node
- Trade-off: cần tăng σ_0, sigma_total, hoặc epsilon_max ~2-4×
  so với spec gốc để cùng đạt acc target

**Files**: [simulator.py:165-212](simulator.py#L165-L212), [renyi_accountant.py](../../core/renyi_accountant.py)
```

### 5.3 §6.4 Tuning Guide — thêm caveat

```markdown
**Lưu ý sau Bug #6 REVERT**: Vì không có q_batch amplification, sigma_0
và sigma_total cần lớn hơn ~2-4× so với cấu hình DP-FedSGD truyền thống
để cùng đạt epsilon_max budget với same accuracy target. Đây là price
của rigorous node-level Local DP guarantee.
```

### 5.4 §7 Quyết định thiết kế — thêm §7.6

```markdown
### 7.6 Node-level Local DP paradigm (commit 5d39ca7)

Quyết định: chuyển từ item-level DP (claim gốc) sang node-level
(user-level) Local DP. Lý do: local SGD multi-batch không satisfy
1-step DP-SGD assumption của sub-sampling amplification lemma
(Mironov-Talwar 2019). Per-node RenyiAccountant tích RDP độc lập
cho mỗi client; ε báo cáo là max(per_node_eps) — worst-case node.

DP unit: mỗi client/node, không phải mỗi data record.
```

---

## 6. Tham chiếu

1. **Mironov, I., & Talwar, K. (2019)**. *Rényi Differential Privacy of the Sampled Gaussian Mechanism*. arXiv:1908.10530.
   → Định lý chính thức cho RDP của subsampled Gaussian, áp dụng trong Opacus.

2. **Abadi, M., et al. (2016)**. *Deep Learning with Differential Privacy*. CCS 2016.
   → Setup DP-SGD gốc: 1 batch Poisson sample + 1 noise inject per step.

3. **McMahan, H. B., et al. (2017)**. *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS 2017.
   → FedAvg multi-batch local training paradigm.

4. **Mironov, I. (2017)**. *Rényi Differential Privacy*. CSF 2017.
   → RDP composition theorem và RDP→(ε,δ)-DP conversion.

5. **Balle, B., & Wang, Y.-X. (2018)**. *Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising*. ICML 2018.
   → Tight σ cho Gaussian Mechanism với mọi ε > 0 (đã áp dụng đúng trong code, REPORT.md §1 Bug #8).

---

## 7. Tóm tắt một dòng

> REPORT.md §5 Bug #6 đề xuất `q_composed = q_client · q_batch` KHÔNG valid cho noise_game vì setup là DP-FedAvg multi-batch, không phải 1-step DP-SGD. Code đã correctly không apply công thức này từ commit 5d39ca7. Cần update REPORT.md để match code và tránh hiểu lầm về tính rigorous của ε reported.
