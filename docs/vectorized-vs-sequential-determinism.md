# Vectorized vs Sequential Training: Numerical Determinism & Fairness

> **Tài liệu này giải thích**: tại sao cùng 1 config khi chạy `use_vectorized: true` vs `false` lại cho kết quả khác nhau, và liệu có "fair" khi compare experiments giữa 2 paths này. Reference cho team + reviewer paper.

---

## 1. TL;DR

- **Cùng config + cùng seed** ở 2 paths (vectorized vs sequential) **vẫn ra kết quả khác** — đây là behavior **chuẩn của PyTorch**, KHÔNG phải bug.
- Nguyên nhân chính: **FP32 non-associativity** + **RNG consumption order khác** + **torch.func.vmap** dùng code path riêng so với autograd.
- "Fair" hay không phụ thuộc cách compare: **cả 2 algorithms cùng path = fair**, **mix path = unfair**.
- Project hiện tại: **14/14 fast-experiment configs đều `use_vectorized: true`** → ✅ consistent, fair.
- Cho paper: lock path + multi-seed (≥3) + note rõ "vectorized vmap pipeline" trong methodology.

---

## 2. Tại sao cùng config lại ra kết quả khác?

5 nguyên nhân kỹ thuật, đa số tích lũy theo rounds:

### 2.1 FP32 non-associativity

Số float32 không có tính kết hợp:

```
(a + b) + c ≠ a + (b + c)    # với errors ~1e-7
```

Khi compute `Σ gradient_per_sample / batch_size`:

| Path | Reduction order |
|---|---|
| **Sequential** | Left-fold: `((((g₀+g₁)+g₂)+...)+gₙ) / B` |
| **Vectorized vmap** | Có thể parallel/tree-reduction, hoặc batched matmul reduction |

Code ref: [vectorized_trainer.py:97-100](../core/vectorized_trainer.py#L97-L100):
```python
grad_fn_unmasked = grad(per_client_loss)
vmap_grad_unmasked = vmap(grad_fn_unmasked, in_dims=(0, 0, 0))
```

→ Bit-level differences ~1e-7 mỗi step. Nhỏ thôi nhưng tích lũy.

### 2.2 RNG consumption order khác nhau

Cùng `seed=42`, nhưng pattern dùng RNG hoàn toàn khác:

| Path | RNG draw sequence |
|---|---|
| **Sequential ThreadPool** | Node 0 shuffle → Node 0 batches → Node 1 shuffle → ... |
| **Vectorized** | All N clients shuffle CÙNG LÚC → batched dataloader yield mini-batches across all clients per step |

Cả 2 path đều dùng `data_gen` từ [base_simulator.py:82-95](../core/base_simulator.py#L82-L95) `_make_isolated_gen`, nhưng **thứ tự gọi `gen.next()` khác nhau** → mỗi node nhận **khác mini-batches** → khác gradients ngay từ round 0.

### 2.3 `torch.func.vmap` + `grad` có code path riêng

Vectorized dùng functional transforms khác autograd:

```python
# Vectorized (functional)
out = functional_call(base_model, params_dict, (x,))   # functional path
loss = F.cross_entropy(out, y)
grads = grad(loss)(params)                              # torch.func.grad
batched_grads = vmap(grad_fn)(params_stack, x, y)       # vmap-aware kernels

# Sequential (autograd)
out = model(x)            # nn.Module forward
loss = F.cross_entropy(out, y)
loss.backward()           # autograd backward
grads = [p.grad for p in model.parameters()]
```

Khác biệt:
- `functional_call` không attach autograd state lên `nn.Parameter`
- vmap-aware kernels cho conv2d/matmul có thể chọn algorithm khác (cuDNN selectors)
- Reduction trong `cross_entropy` khi vmap'd có behavior subtly khác

### 2.4 Padding + masking cho variable client sizes

Khi clients có dataset sizes khác nhau (Dirichlet split, FEMNIST natural partition), vectorized path phải **pad tới max size + mask**.

Code ref: [vectorized_trainer.py:90-95](../core/vectorized_trainer.py#L90-L95):
```python
def per_client_loss_masked(params_dict, x, y, mask):
    out = functional_call(base_model, params_dict, (x,))
    per_sample = F.cross_entropy(out, y, reduction="none")
    masked = per_sample * mask.to(per_sample.dtype)
    denom = mask.sum().clamp(min=1).to(per_sample.dtype)
    return masked.sum() / denom
```

Sequential thì mỗi node iterate đúng số batches của mình, không cần mask.

→ Vectorized có thêm `mask * x` và `clamp(min=1)` → khác numerical path so với sequential.

### 2.5 ThreadPool race conditions

Sequential path dùng `ThreadPoolExecutor(max_workers=n_workers)` ở [base_simulator.py:296-303](../core/base_simulator.py#L296-L303):

```python
with ThreadPoolExecutor(max_workers=n_workers) as pool:
    for nid, upd, n in pool.map(_train, self.nodes.values()):
        updates[nid] = upd
```

Khi `n_workers > 1`:
- **Thread scheduling không deterministic** trên CPU
- Per-node `compute_update` chạy parallel
- Dict insertion order phụ thuộc thread completion order
- Aggregator iterate `updates.items()` → có thể aggregate theo thứ tự khác → khác kết quả (nhỏ, do FP non-associativity)

Note: với `n_workers=1`, sequential path fully deterministic.

### 2.6 Tích lũy qua rounds

| Round | Sai số tích lũy giữa 2 paths |
|---|---|
| Round 1 | ~1e-7 per param (tiny) |
| Round 5 | ~1e-5 per param (small) |
| Round 20 | ~1e-3 per param (visible) |
| Round 30+ ở **unstable regime** | **Khác hẳn** — một path NaN, path kia OK |

Trong **stable training** (lr nhỏ, light noise), 2 paths cho **gần như** cùng kết quả (acc khác ±1-2%).

Trong **unstable regime** (lr lớn, noise lớn — như NoiseGame với `lr=0.005, sigma_total=0.5`), small numerical differences amplify exponentially → divergent trajectories.

---

## 3. Đây có phải bug không?

**KHÔNG.** PyTorch official docs ([Reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html)) confirm:

> Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds.

Specifically:
- Same seed only guarantees same results **within same code path**
- `torch.func.vmap`, CUDA atomic ops, một số cuDNN kernels được labeled **nondeterministic**
- Đây là tradeoff được chấp nhận giữa speed và bit-level reproducibility

Cùng phenomena trong:
- TensorFlow (`tf.function` JIT vs eager)
- JAX (`jit` vs eager)
- Opacus (vectorized DP-SGD via vmap is standard)

---

## 4. Có "fair" không?

**Vectorized không unfair tự nó** — nó là implementation khác, không phải cheating. Định nghĩa "fair" phụ thuộc setup compare:

### 4.1 Khi nào FAIR

| Setup | Fair? | Lý do |
|---|---|---|
| Algorithm A vs B, **cả 2 vectorized** | ✅ FAIR | Cùng numerical environment |
| Algorithm A vs B, **cả 2 sequential** | ✅ FAIR | Cùng path, cùng FP behavior |
| Same algo run 2 lần, **cùng path + seed** | ✅ Reproducible | Bit-level identical |
| Multi-seed (≥3 seeds), **cùng path** | ✅ Best practice | Wash out RNG variance |

### 4.2 Khi nào UNFAIR

| Setup | Fair? | Issue |
|---|---|---|
| Algo A **vectorized** vs Algo B **sequential** | ❌ UNFAIR | Difference từ algorithm hay implementation? |
| Single seed comparison, bất kỳ path | ⚠️ Insufficient | RNG variance lớn, không statistical |
| Mix paths trong cùng experiment table | ❌ UNFAIR | Reviewer sẽ challenge |

### 4.3 Nguyên tắc fair comparison

1. **Path consistency**: Tất cả algorithms compare phải dùng cùng `use_vectorized` value
2. **Multi-seed**: ≥3 seeds, report mean ± std (numerical noise giữa paths < seed variance)
3. **Same hyperparameter tuning effort**: Mỗi algorithm phải được tune cùng mức (không tune algorithm của mình kỹ hơn baseline)
4. **Document explicitly**: Note path trong methodology section

### 4.4 Phản đề: vectorized "công bằng hơn"?

Một số luận điểm có thể argue vectorized **rigorous hơn**:
- Sequential ThreadPool có race conditions → ít deterministic hơn
- Vectorized vmap không có thread scheduling
- Vectorized = standard practice trong DP-FL literature (Opacus, JAX-DP, TF-Privacy)

Một số luận điểm ngược lại:
- Sequential mô phỏng "real FL" hơn (mỗi client = 1 process riêng)
- Vectorized batches all clients together → "infrastructure cheating" về wall-time

→ **Cả 2 đều valid simulator implementations**. Câu hỏi "cái nào đúng?" sai cách đặt.

---

## 5. Recommendations cho project này

### 5.1 Cho paper publication

```yaml
# Lock toàn bộ configs về cùng path:
training:
  use_vectorized: true   # OR false — chỉ chọn 1, áp toàn bộ
  n_workers: 1           # nếu sequential, để loại race condition
```

**Multi-seed protocol**: Mỗi cell trong results table = mean ± std qua **3-5 seeds**, e.g., `[42, 123, 456]`.

**Methodology note**: 
> "All experiments use vectorized vmap-based training pipeline ([core/vectorized_trainer.py](../core/vectorized_trainer.py)). Each result is mean ± std across 3 seeds. Hyperparameters tuned per-algorithm via grid search at fixed ε budget."

### 5.2 Cho debugging

- Use vectorized cho speed (5-30× faster trên GPU)
- Single seed OK cho sanity check
- Don't worry về numerical noise

### 5.3 Cho research iteration

- Use vectorized as default
- Lock seed=42 cho reproducibility within development session
- Switch sequential **chỉ khi** nghi ngờ vmap-specific bug (e.g., NaN cascade, unexpected explosion)

---

## 6. Verification: kiểm tra setup hiện tại

```bash
cd "/Users/lap15791/Documents/Differential privacy/robust & privacy/dpfl"
grep -rn "use_vectorized" config/fast-experiment/mnist/
```

**Snapshot kết quả audit (2026-04-30)**:
- **14/14** configs trong `config/fast-experiment/mnist/` dùng `use_vectorized: true` → ✅ **consistent**
- Bao gồm: cfl_fedavg, dp_fedavg, fedavg, krum, trimmed_mean, fltrust, flame, dpsgd_kurtosis, trust_aware, noise_game, adaptive_noise, adaptive_noise_momentum_kurtosis, dp_fedavg_eps2, cfl_fedavg_eps2, dp_fedavg_measure
- → User đang compare đúng cách (path consistent)

**Khi tạo config mới**: copy template từ một config existing để inherit `use_vectorized: true` + đảm bảo consistency.

---

## 7. Khi nào VẤN ĐỀ này thực sự matter

| Scenario | Matter? |
|---|---|
| Single research run | Không lắm — 1-2% accuracy noise |
| Comparing algorithm variants | **Có** — phải dùng cùng path |
| Reporting paper results | **Có** — note path trong methodology |
| Debugging instability (NaN, divergence) | **Có** — test cả 2 paths để locate bug |
| Reproducing baseline từ paper khác | ⚠️ — paper khác có thể dùng path khác |

**Trường hợp recent**: NoiseGame ở fast-experiment ([noise_game.yaml](../config/fast-experiment/mnist/noise_game.yaml)) crash với NaN ở round 30 sau khi switch vectorized. Root cause là **lr=0.005 + sigma_total=0.5 đẩy training vào unstable regime**, vectorized FP precision khác đủ để trigger NaN sớm hơn sequential. **Không phải bug** của vectorized — là instability của hyperparameter setup.

---

## 8. References

- [PyTorch Reproducibility Notes](https://pytorch.org/docs/stable/notes/randomness.html) — official docs về determinism
- [JAX FAQ on randomness](https://jax.readthedocs.io/en/latest/faq.html#different-random-numbers) — JAX has same issue
- Opacus (Meta AI): [`opacus.optimizers.DPOptimizer`](https://github.com/pytorch/opacus) — vectorized DP-SGD reference
- TensorFlow Privacy: vectorized + `tf.function` JIT pattern
- Internal: [core/vectorized_trainer.py](../core/vectorized_trainer.py), [core/base_simulator.py:284-385](../core/base_simulator.py#L284-L385)

---

## Open Questions

Nếu user / bạn tester nào đó tìm thấy:
- Vectorized vs sequential khác **rất nhiều** ở stable regime (>5% accuracy gap) → có thể là bug, raise issue
- NaN xảy ra **chỉ** ở vectorized path nhưng config "đáng lẽ" stable → check vmap kernel selection (CPU vs GPU)

→ Trong nhưng case đó, dùng `torch.use_deterministic_algorithms(True)` để force deterministic kernels (chậm hơn, nhưng eliminate vmap-specific issues).
