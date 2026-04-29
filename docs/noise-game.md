# Noise Game — Implementation Report

## Tổng quan

Báo cáo này mô tả chi tiết cách hiện thực hóa chiến lược Noise Game theo nội dung `noise-game.md` bằng kiến trúc 4 lớp. Mục tiêu chính là kết hợp:

- Differential Privacy để bảo vệ gradient,
- Renyi DP để điều phối mức noise theo ngân sách privacy,
- strategic noise để tăng robust với model poisoning attack,
- recovery để lấy lại tín hiệu gradient sau khi thêm noise.

Toàn bộ pipeline được triển khai trong `algorithms/noise_game/mechanism.py` và được liên kết với simulator bằng các tham số DP/adaptive. Các phần nội dung bên dưới giải thích rõ từng lớp, công thức, ý nghĩa và quyết định thiết kế.

---

## Layer 1 — Differential Privacy Layer

### Mục tiêu

Layer 1 chịu trách nhiệm bảo đảm privacy cơ bản. Nó giới hạn sensitivity của gradient và thêm Gaussian noise chuẩn hoá.

### Cơ chế thực hiện

1. Clipping gradient:
   - Với gradient g, thực hiện:
     ```math
     g_t \leftarrow \frac{g_t}{\max\left(1, \frac{\|g_t\|}{C}\right)}
     ```
   - Điều này đảm bảo mỗi gradient sau clipping có chuẩn không vượt quá `C`.
   - Clipping là bước quan trọng để DP guarantee có ý nghĩa; nếu không giới hạn `\|g_t\|`, Gaussian mechanism không thể tính toán privacy cost chính xác.

2. Gaussian noise DP:
   - Sinh noise:
     ```math
     n_{DP,t} \sim \mathcal{N}(0, \sigma_{DP,t}^2 I)
     ```
   - Noise này được thêm vào sau khi clipping: `\hat{g}_t = g_{bar,t} + n_{DP,t} + n_{strategic,t}`.

3. Lower bound chuẩn cho sigma_DP (Balle-Wang 2018 analytic Gaussian — Bug #8 fix):
   - Trước đây dùng công thức Dwork-Roth `σ ≥ C·√(2·ln(1.25/δ))/ε` — chỉ valid với `ε ∈ (0, 1)` (proof Taylor truncation).
   - Default config dùng `epsilon_max = 50` nằm ngoài regime valid → Dwork-Roth UNDER-estimates required σ.
   - Hiện thực hóa bằng Balle-Wang 2018 analytic Gaussian Mechanism, tight bound cho mọi `ε > 0`:
     ```math
     \Phi\!\left(\frac{\Delta}{2\sigma} - \frac{\varepsilon\sigma}{\Delta}\right) - e^{\varepsilon} \cdot \Phi\!\left(-\frac{\Delta}{2\sigma} - \frac{\varepsilon\sigma}{\Delta}\right) = \delta
     ```
     trong đó `Φ` là CDF chuẩn `N(0,1)`, `Δ` là L2 sensitivity (= clip bound `C`).
   - Solve numerical qua bisection (`scipy.optimize.brentq`) — implementation [`analytic_gaussian_sigma()`](../algorithms/noise_game/mechanism.py).
   - Reference: [Balle & Wang 2018 — "Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising"](https://arxiv.org/abs/1805.06530), Theorem 9.

### File liên quan

- `algorithms/noise_game/mechanism.py` → hàm `compute_dp_noise()` và logic clipping

### Ý nghĩa

Layer 1 tách biệt rõ ràng phần privacy noise: nó đảm bảo guarantee DP độc lập với phần strategic noise và các cơ chế những lớp sau thêm.

---

## Layer 2 — RDP-based Noise Control

### Mục tiêu

Layer 2 quản lý ngân sách privacy và chất lượng noise theo thời gian. Nó khai thác RDP accounting để điều chỉnh `\sigma_{DP}` một cách adaptative dựa trên trạng thái trust/attack và còn lại budget.

### Cơ chế thực hiện

1. RDP accounting nội bộ:
   - Sử dụng công thức heuristic với `\alpha = 2`:
     ```math
     \varepsilon_t(\alpha) = \frac{\alpha}{2 \sigma_{DP,t}^2}
     ```
   - Tích luỹ privacy cost nội bộ: `eps_spent += C^2 / (2 * sigma_DP^2)`.
   - Lưu ý: đây chỉ là tracker để điều phối scheduler, còn `RenyiAccountant` chính thức trong simulator chịu trách nhiệm privacy auditing.

2. Adaptive sigma scheduler:
   - `\sigma_{DP,t}` được điều chỉnh theo ba yếu tố chính:
     - `epsilon_remain` (ngân sách chưa dùng hết)
     - `trust_t` (độ ổn định gradient)
     - `attack_t` (tín hiệu bất thường/tấn công)
   - Công thức heuristics đã áp dụng tương tự:
     ```math
     \sigma_{DP,t} = \sigma_0 \exp(-\kappa t) \cdot \max(0.1, 1 - \frac{eps_spent}{eps_max}) \cdot (1 + attack_signal)
     ```
   - Khi model ổn định và budget còn lại thấp, sigma giảm dần để bảo toàn privacy; khi attack tín hiệu tăng, sigma được tăng lên để tăng robustness.

3. Strategic noise coupling:
   - `\sigma_{strat,t}` được lấy theo tỷ lệ với `\sigma_{DP,t}`:
     ```math
     \sigma_{strat,t} = \beta_t \cdot \sigma_{DP,t}
     ```
   - Tham số `beta_strat` cho phép điều chỉnh mức noise chiến lược so với noise privacy.

### Logic điều khiển chính

- trust thấp → tăng `\sigma_{DP,t}` và `\sigma_{strat,t}`
- attack cao → tăng noise để tăng robustness
- budget thấp → giảm `\sigma_{DP,t}` để tiết kiệm privacy

### File liên quan

- `algorithms/noise_game/mechanism.py` → hàm `compute_sigma_dp()`

### Ý nghĩa

Layer 2 là “bộ điều tiết” trung tâm: nó xác định mức noise dùng cho cả privacy và strategic noise, cân bằng giữa bảo mật, robust và budget privacy.

---

## Layer 3 — Strategic Noise Design

### Mục tiêu

Layer 3 xây dựng phần noise chiến lược `n_{strategic}` sao cho vừa chống lại attack, vừa giữ lại cấu trúc thông tin quan trọng của gradient.

### Thành phần chính

1. Directional noise:
   - Dựa trên biến thiên gradient `v_t = g_t - g_{t-1}`.
   - Nếu trust thấp, `v_t` có thể chứa tín hiệu tấn công bất thường.
   - Noise hướng attack được tạo ra theo dạng:
     ```math
     n_{attack} = \alpha_t \cdot v_t
     ```
   - Tham số `\alpha_t` điều chỉnh cường độ noise theo độ bất thường.

2. Orthogonal noise:
   - Giữ noise không làm lệch hướng descent chính:
     ```math
     n_{orth} = z - \frac{z \cdot g_t}{\|g_t\|^2} g_t
     ```
   - `z` là vector ngẫu nhiên; thành phần này thuần túy ở không gian vuông góc với gradient.
   - Mục đích là tăng entropy noise nhưng không phá hoại hướng chính của gradient.

3. Reshape-decomposed noise (a.k.a. "spectrum"):
   - Dựng noise theo SVD của **2D reshape nhân tạo** của gradient vector. Lưu ý: KHÔNG phải spectrum thật của model weights/layers.
   - Pipeline thực tế (theo [mechanism.py:132–158](../algorithms/noise_game/mechanism.py#L132-L158)):
     ```math
     \begin{aligned}
     & \tilde g = [g_t \;\|\; 0_{\text{pad}}], \quad M = \mathrm{reshape}(\tilde g, \; (k, c)) \\
     & M \approx U \Sigma V^\top, \quad U \in \mathbb{R}^{k \times r}, \; V \in \mathbb{R}^{c \times r}, \; r = \min(R, k, c) \\
     & w = (\Sigma + \epsilon_{\text{reg}})^{-1}, \quad \epsilon_{\text{reg}} = 10^{-8} \quad \text{(regularization tránh blow-up)}\\
     & \rho \sim \mathcal{N}(0, \sigma^2 I_r) \\
     & n_{spec} = \mathrm{flatten}\big(U \cdot \mathrm{diag}(w \odot \rho) \cdot V^\top\big)[:D]
     \end{aligned}
     ```
   - Tham số: `svd_reshape_k = k` (mặc định 64), `svd_rank = R` (mặc định 16).
   - **Caveat**: reshape là tuần tự arbitrary (`g[i·c + j] → M[i,j]`), KHÔNG theo layer boundaries — params từ multiple layers bị trộn vào row/col mà không có cấu trúc semantic. Singular vectors $U, V$ phản ánh structure của artificial reshape, không phải spectrum thật của model. Tên "spectrum-aware" giữ để backward-compat; honest naming sẽ là **"low-rank-shaped structured noise"**.
   - Tính chất kỹ thuật: $w_i = 1/(\Sigma_{ii} + 10^{-8})$ — singular value càng nhỏ → weight càng lớn → noise tập trung vào mode yếu của reshape matrix, đảm bảo không Inf/NaN khi $\Sigma_{ii} \to 0$.
   - Edge case: SVD fail (matrix degenerate) → fallback `n_spec = randn_like(g) * sigma` ([mechanism.py:148–149](../algorithms/noise_game/mechanism.py#L148-L149)).

4. Kết hợp và chuẩn hoá noise:
   - Tổng hợp các thành phần:
     ```math
     \hat{n}_t = \frac{n_{attack} + n_{orth} + n_{spec}}{\|n_{attack} + n_{orth} + n_{spec}\|}
     ```
   - Sau đó scale:
     ```math
     n_{strategic} = \sigma_{strat,t} \cdot \hat{n}_t
     ```
   - Điều này đảm bảo độ lớn noise chiến lược được kiểm soát bởi `\sigma_{strat,t}` và hướng noise được chuẩn hoá.

### Privacy của n_strategic (Bug #9 — Lipschitz inflation, đã fix auto-inflate)

`n_strategic` phụ thuộc trực tiếp vào `g` (raw gradient) qua các thành phần directional, orthogonal, spectrum. Strict Gaussian Mechanism RDP bound `α·C²/(2σ²)` chỉ valid khi noise độc lập với input → ở đây bound bị inflate.

**Lipschitz analysis:**
- Sau normalize: `‖n_strat(g)‖ = σ_strat` constant.
- Worst-case `‖n_strat(g) - n_strat(g')‖ ≤ 2·σ_strat` (direction flip).
- Effective Lipschitz: `L_strat = 2·σ_strat / C = 2·β_strat·z` (với `z = σ_DP/C`).

**Effective sensitivity:**
```math
C_{total} = C \cdot (1 + L_{strat}) = C \cdot (1 + 2 \beta_{strat} z)
```

**Inflation factor cho RDP:**
```math
\text{inflation} = (1 + L_{strat})^2 = (1 + 2 \beta_{strat} z)^2
```

Bảng (β=0.3): z=0.5 → 1.69×, z=1.0 → 2.56×, z=3.0 → 7.84×.

**Implementation (sau Bug #9 fix):** simulator.py phase 4 tự động pass `effective_mult = σ_DP / C_total` (thay vì `σ_DP / C`) vào Opacus → ε reported đã RIGOROUS, include inflation. **Không cần manual correction khi publish paper.** Xem `algorithms/noise_game/REPORT.md` §4.1 và Bug #9 để chi tiết.

**Lý do CHỌN auto-inflate (Option A') thay vì post-processing (Option B):**
- Compute `n_strat` từ noisy `Y = g + n_DP` (Option B) phá vỡ robustness: `Y` có SNR `~ 1/(z·√D)` → trong heavy-noise regime, attack signal indistinguishable from noise → Layer 3 vô hiệu hoàn toàn (default config: SNR ≈ 0.6%).
- Auto-inflate (Option A') giữ utility/robustness từ raw-g version, chỉ inflate accountant ε rigorously.

**Caveat về Lipschitz formal:** True Lipschitz unbounded tại origin (normalize step). Bound effective dựa trên max gradient distance `‖g-g'‖ ≤ C` — đủ chặt cho practical, cần guard `‖g‖ ≥ δ_min` cho paper-quality proof formal.

### File liên quan

- `algorithms/noise_game/mechanism.py` → `directional_noise()`, `orthogonal_noise()`, `spectrum_noise()`

### Ý nghĩa

Layer 3 là phần thuần “robustness”: nó thêm noise có cấu trúc để giảm tác động attack mà vẫn lưu giữ tín hiệu chính.

---

## Layer 4 — Recovery and Optimization

### Mục tiêu

Layer 4 xử lý gradient đã nhiễu để khôi phục phần tín hiệu hữu ích và duy trì cập nhật mô hình ổn định.

### Thành phần chính

1. EMA smoothing:
   - Dùng bộ lọc EMA để làm mượt gradient noisy:
     ```math
     \tilde{g}_t = \gamma \tilde{g}_{t-1} + (1-\gamma) \hat{g}_t
     ```
   - `\gamma` quyết định trọng số lịch sử.

2. Momentum:
   - Ổn định cập nhật bằng momentum:
     ```math
     m_t = \beta m_{t-1} + (1-\beta) \tilde{g}_t
     ```
   - Giúp giảm dao động do noise.

3. Gradient filtering:
   - Nếu gradient noisy lệch quá xa lịch sử, có thể reject:
     ```math
     \cos(\hat{g}_t, \tilde{g}_{t-1}) < \tau \Rightarrow \text{reject}
     ```
   - Khi đó dùng EMA hoặc gradient trước để tránh cập nhật gây hại.

4. Trust-aware learning rate:
   - Learning rate được điều chỉnh theo trust score:
     ```math
     final = m_t \cdot \max(trust_t, 0)
     ```
   - Mục đích giảm ảnh hưởng của cập nhật khi trust thấp.

5. Variance reduction / two-track model:
   - Có cơ chế tương tự SCAFFOLD:
     ```math
     g' = g - c_i + c_{global}
     ```
   - Ngoài ra có ý tưởng hợp nhất hai đường đi:
     ```math
     w = \lambda w^{clean} + (1-\lambda) w^{robust}
     ```
   - Điều này giúp cân bằng giữa mô hình chính xác và mô hình bền vững.

### Files liên quan

- `algorithms/noise_game/node.py`
- `algorithms/noise_game/simulator.py`

### Ý nghĩa

Layer 4 đảm bảo rằng dù đã thêm noise mạnh, quá trình tối ưu hóa vẫn giữ được khả năng học và không bị nhiễu mất phương hướng.

---

## Budget Constraint và NSR

### Constraint tổng năng lượng noise

<<<<<<< HEAD
- Tổng năng lượng noise phải thỏa:
  ```math
  \|n_{DP}\|^2 + \|n_{strategic}\|^2 \le \sigma_{total}^2
  ```
- Nếu tổng độ lớn noise vượt, cả hai thành phần sẽ được scale xuống tỉ lệ tương ứng.
- Điều này đảm bảo hệ thống không thêm noise quá lớn gây mất tín hiệu.

### NSR monitoring

- Khai báo chỉ số noise-to-signal ratio:
  ```math
  NSR = \frac{\|n_{DP} + n_{strategic}\|}{\|g\|}
  ```
- Nếu NSR vượt ngưỡng `nsr_warn`, hệ thống ghi log warning để cảnh báo noise quá cao.
- Các chỉ số `avg_nsr` và `avg_sigma_dp` được lưu xuống round logs để theo dõi xu hướng.

### Ý nghĩa

Constraint và NSR giúp cân bằng giữa privacy/robustness và chất lượng gradient. Khi noise quá lớn, system biết phải giảm cường độ hoặc điều chỉnh tham số.
=======
Privacy cost tại vòng `t` (Mironov 2017, Gaussian Mechanism với sensitivity sau clipping = `C = clip_bound`):

```math
\varepsilon_t(\alpha) = \frac{\alpha \cdot C^2}{2 \, \sigma_{DP,t}^2}
```

Tương đương dạng noise multiplier `z = σ_DP / C`:

```math
\varepsilon_t(\alpha) = \frac{\alpha}{2 \, z^2}
```

Tổng RDP cost sau `T` vòng (composition Mironov 2017 Theorem 5):

```math
\mathrm{RDP}_{total}(\alpha) = \sum_{t=1}^T \varepsilon_t(\alpha)
```

## 5.2 Ngân sách privacy

Để so sánh với `ε_max` (đơn vị (ε, δ)-DP) phải convert RDP → (ε, δ)-DP qua Mironov 2017 Theorem 8:

```math
\varepsilon(\alpha, \delta) = \mathrm{RDP}_{total}(\alpha) + \frac{\log(1/\delta)}{\alpha - 1}
```

Tối ưu over α:

```math
\varepsilon^* = \min_{\alpha > 1} \varepsilon(\alpha, \delta) \le \varepsilon_{max}
```

> **Implementation note**: heuristic scheduler trong `mechanism.py` track `rdp_spent` (RDP_α với α cố định) rồi gọi `compute_eps_dp()` để convert trước khi compare với `epsilon_max`. Privacy claim cuối cùng (trong `report.txt`) dùng Opacus `compute_rdp` + `get_privacy_spent` (tight SGM bounds) — xem `core/renyi_accountant.py`.

Layer 2 chịu trách nhiệm cân bằng giữa mức noise bảo mật và ngân sách privacy có hạn.

## 5.3 Adaptive scheduler

Một bộ điều phối mẫu có dạng:

```math
\sigma_{DP,t} = f(\varepsilon_{remain}, trust_t, attack_t)
```

Trong đó:

- `\varepsilon_{remain}` là ngân sách privacy còn lại,
- `trust_t` đo độ ổn định của gradient,
- `attack_t` là tín hiệu chỉ báo tấn công.

## 5.4 Chiết tách strategic noise

Phần noise chiến lược được chọn theo tỷ lệ:

```math
\sigma_{strat,t} = \beta_t \cdot \sigma_{DP,t}
```

Trong đó `\beta_t` điều chỉnh phần noise dành cho robustness so với privacy.

## 5.5 Logic điều khiển

Các quy tắc chung:

- `trust thấp` → tăng `\sigma_{DP,t}` và `\sigma_{strat,t}`
- `attack cao` → tăng noise để tăng robustness
- `budget thấp` → giảm `\sigma_{DP,t}` để tiết kiệm privacy

Layer 2 là bộ điều tiết trung tâm của hệ thống.
>>>>>>> 78726ecd3673b3984535b4e7fdec1d06b5ff1fed

---

## Thay đổi cấu hình và tham số

### Thêm trường trong `NoiseGameConfig`

```python
beta_strat: float = 0.5      # Tỷ lệ chuyển từ sigma_DP sang sigma_strat
sigma_total: float = 3.0     # Năng lượng noise tối đa cho tổng noise
nsr_warn: float = 5.0        # Ngưỡng cảnh báo NSR
```

### Loại bỏ trường cũ

```yaml
noise_mult: 1.1 # Đã loại bỏ vì noise_game tự quản lý mức noise, không cần multiplier ngoài
```

### Tệp đã chỉnh sửa

- `config.py` — thêm trường cấu hình mới
- `config/noise_game.yaml` — cập nhật tham số, loại bỏ `noise_mult`
- `config/cifar10_noise_game.yaml` — cập nhật tương tự
- `algorithms/noise_game/mechanism.py` — triển khai pipeline 4 lớp và các hàm noise
- `algorithms/noise_game/simulator.py` — cập nhật phases liên quan đến DP, trust và logging
- `run.py` — truyền thông số DP/adaptive sang `mechanism.py`

### Tệp không thay đổi

- `algorithms/noise_game/node.py` — buffer và trạng thái đã đủ cho pipeline hiện tại
- `algorithms/noise_game/simple_avg_aggregator.py` — không cần sửa đổi cho noise game

---

## Quyết định thiết kế chính

### 1. `attack_signal` chọn là `(1 - trust)`

- Lý do: đơn giản, trực quan và phù hợp với mô tả trong `noise-game.md`.
- Nếu trust giảm tức là gradient không ổn định → tăng noise.
- Giữ KISS, tránh phức tạp hoá bằng nhiều chỉ báo.

### 2. `sigma_total` cố định

- Chọn giữ fixed cap `sigma_total = 3.0` như trong kế hoạch ban đầu.
- Nhận xét: với các mô hình lớn, constraint này có thể làm noise hiệu dụng bị scale xuống nhiều.
- Khuyến nghị tương lai: nếu cần scale theo kích thước model, có thể chuyển sang base theo `sqrt(param_dim)`.

### 3. Scheduler DP adaptative

- Áp dụng giảm dần noise theo thời gian (`sigma_0 e^{-\kappa t}`) và giảm thêm khi gần cạn budget privacy.
- Mục đích: giữ accuracy khi model ổn định và không lãng phí epsilon.

### 4. RDP accounting nội bộ chỉ để điều phối

- Không dùng tracker này như phương pháp accounting duy nhất.
- `RenyiAccountant` trong simulator vẫn đóng vai trò chính xác hơn cho privacy auditing.
- Nội bộ tracker chỉ giúp scheduler ổn định.

### 5. Strategic noise cấu trúc nhiều thành phần

- Kết hợp hướng tấn công, noise vuông góc và spectrum-aware để vừa robust vừa giữ tín hiệu.
- Đây là điểm khác biệt chính so với DP Gaussian noise chuẩn.

---

## Các lựa chọn đã cân nhắc nhưng chưa dùng

### 1. Scale `sigma_total` theo kích thước model

- Lý do không dùng: giữ đơn giản theo yêu cầu hiện tại.
- Hậu quả: với D lớn, noise tổng có thể bị giới hạn quá chặt.

### 2. Attack signal phức tạp hơn

- Có thể dùng `\|g_t - g_{t-1}\| / \|g_t\|` hoặc kết hợp nhiều chỉ số.
- Hiện tại dùng trust đơn giản để giữ nguyên lý KISS.

### 3. Full RDP min-alpha accounting trong mechanism

- Lý do không dùng: quá phức tạp cho mục đích scheduler.
- Giữ `RenyiAccountant` chính thức làm việc chính xác.

### 4. Per-dimension noise cap

- Ý tưởng tốt nhưng không cần thiết trong phiên bản hiện tại.
- Đã chọn fixed global noise budget để dễ giám sát.

---

## Kết luận

Triển khai hiện tại đáp ứng đầy đủ kiến trúc 4 lớp trong `noise-game.md`:

- Layer 1 định nghĩa DP qua clipping và Gaussian noise.
- Layer 2 điều phối noise bằng RDP heuristics và budget privacy.
- Layer 3 xây noise chiến lược có cấu trúc hướng attack, vuông góc và phổ-aware.
- Layer 4 duy trì recovery và cập nhật mô hình ổn định.

<<<<<<< HEAD
Báo cáo này đã làm rõ từng bước thực hiện, lý do chọn tham số và các trade-off chính. Nếu cần, bước tiếp theo có thể là đánh giá thực nghiệm với metrics `avg_nsr`, `avg_sigma_dp`, và so sánh accuracy/trust trên CIFAR-10/MNIST.
=======
Trong đó `n_{DP,t}` bảo vệ quyền riêng tư và `n_{strategic,t}` chống tấn công.

## 9.2 Ràng buộc tổng noise

```math
\|n_{DP,t}\|^2 + \|n_{strategic,t}\|^2 \le \sigma_{total}^2
```

Ràng buộc này giữ tổng năng lượng noise trong giới hạn chấp nhận được.

> **Privacy accounting note (CRITICAL)**: khi cap kích hoạt (energy > σ_total²), `_enforce_budget` scale cả `n_DP` và `n_strategic` xuống bằng cùng một factor. Sau cap, σ thực tế của `n_DP` là `σ_eff = ‖n_DP_post‖ / √D`, **không phải** σ_DP pre-cap từ scheduler. Privacy accountant **bắt buộc** phải nhận `σ_eff` (post-cap) — dùng pre-cap σ sẽ under-report ε rất nhiều (factor có thể ~10⁴ hoặc hơn tùy config). Xem `simulator.py` Phase 4 implementation.

---

# 10. Hệ thống điều khiển noise

## 10.1 Tỷ lệ noise/tín hiệu

```math
NSR = \frac{\|n\|}{\|g\|}
```

NSR đo mức độ nhiễu so với tín hiệu gradient.

## 10.2 Hành vi thích nghi

| Giai đoạn | Mức noise  |
| --------- | ---------- |
| Early     | Cao        |
| Mid       | Trung bình |
| Late      | Thấp       |

Chiến lược là dùng nhiều noise khi hệ thống mới bắt đầu hoặc khi tấn công mạnh, rồi giảm dần khi mô hình ổn định.

---

# 11. Đóng góp chính

1. Công thức game-theoretic để thiết kế noise.
2. Tách biệt rõ ràng giữa noise privacy và noise chiến lược.
3. Điều phối noise adaptative bằng RDP.
4. Noise cấu trúc đa thành phần (theo hướng, vuông góc, phổ-aware).
5. Tích hợp khôi phục tín hiệu (EMA, momentum, filtering).
6. Kiến trúc DFL vừa robust vừa giữ độ chính xác.
>>>>>>> 78726ecd3673b3984535b4e7fdec1d06b5ff1fed
