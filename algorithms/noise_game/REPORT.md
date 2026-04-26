# Noise Game — Implementation Report

## Tổng quan

Báo cáo này mô tả chi tiết cách hiện thực hóa chiến lược Noise Game theo nội dung `noise-game.md` bằng kiến trúc 4 lớp. Mục tiêu chính là kết hợp:

- Differential Privacy để bảo vệ gradient,
- Renyi DP để điều phối mức noise theo ngân sách privacy,
- strategic noise để tăng robust với model poisoning attack,
- recovery để lấy lại tín hiệu gradient sau khi thêm noise.

Toàn bộ pipeline được triển khai trong `algorithms/noise_game/mechanism.py` và được liên kết với simulator bằng các tham số DP/adaptive. Các phần nội dung bên dưới giải thích rõ từng lớp, công thức, ý nghĩa và quyết định thiết kế.


## Parameter Specification & Tuning Guide

### 1. Mục tiêu

Tài liệu này tổng hợp **toàn bộ tham số điều chỉnh** trong Noise Game, giải thích vai trò toán học, tác động khi thay đổi, lý do chọn công thức, và hướng dẫn tuning thực tế cho từng kịch bản (MNIST, CIFAR-10, DFL).

---

### 2. Tổng quan công thức & ý nghĩa

Noise Game là hệ điều khiển đóng (closed-loop) cho adaptive noise, với pipeline:
$$
g^{final}_t = \mathcal{R}\left(g_t + n_{DP,t} + n_{strat,t}\right)
$$
Trong đó:
- $g_t$: gradient sau clipping
- $n_{DP,t}$: Gaussian noise đảm bảo DP
- $n_{strat,t}$: strategic noise tăng robustness
- $\mathcal{R}$: recovery (EMA, momentum, filter)

**Mục tiêu:**  
- Đảm bảo privacy (DP)
- Tăng robustness chống attack
- Giữ chất lượng gradient (accuracy)

---

### 3. Giải thích công thức & lý do

#### Layer 1 — Differential Privacy
- **Clipping:** $g \leftarrow \frac{g}{\max(1, \|g\|/C)}$  
  → Giới hạn sensitivity, đảm bảo DP meaningful.
- **DP noise:** $n_{DP} \sim \mathcal{N}(0, \sigma_{DP}^2 I)$  
  → Đảm bảo $(\epsilon, \delta)$-DP.
- **Sigma floor:** $\sigma_{DP} \geq \frac{C\sqrt{2\ln(1.25/\delta)}}{\epsilon}$  
  → Đảm bảo không giảm noise dưới mức cần thiết.

#### Layer 2 — Adaptive Control (RDP-based)
- **RDP tracker:** $\varepsilon_t(\alpha) = \frac{\alpha C^2}{2\sigma_{DP}^2}$  
  → Theo dõi ngân sách privacy, điều phối noise.
- **Adaptive scheduler:**  
  $\sigma_{DP,t} = \sigma_0 e^{-\kappa t} \cdot \max(\text{floor}, 1 - \frac{\epsilon_{spent}}{\epsilon_{max}}) \cdot (1 + \text{attack})$  
  → Giảm noise khi model ổn định, tăng khi bị attack hoặc còn nhiều budget.

#### Layer 3 — Strategic Noise
- **Directional:** $n_{attack} = \alpha (g_t - g_{t-1})$  
  → Tăng noise theo hướng bất thường khi trust thấp.
- **Orthogonal:** $n_{orth} = z - \frac{z \cdot g_t}{\|g_t\|^2}g_t$  
  → Thêm nhiễu vuông góc, không phá hướng descent.
- **Spectrum-aware:** $n_{spec} = U \cdot \mathrm{diag}(\lambda^{-1}) \cdot r$  
  → Tăng entropy ở mode yếu, giảm mất mát thông tin.
- **Kết hợp:** $n_{strat} = \sigma_{strat} \cdot \frac{n_{attack} + n_{orth} + n_{spec}}{\|...\|}$

#### Global Constraint & NSR
- **Energy cap:** $\|n_{DP}\|^2 + \|n_{strat}\|^2 \leq \sigma_{total}^2$  
  → Không để noise quá lớn làm mất tín hiệu.
- **NSR:** $\text{NSR} = \frac{\|n_{DP} + n_{strat}\|}{\|g\|}$  
  → Theo dõi, cảnh báo khi noise quá cao.

#### Layer 4 — Recovery
- **EMA:** $\tilde{g}_t = \gamma \tilde{g}_{t-1} + (1-\gamma)\hat{g}_t$
- **Momentum:** $m_t = \beta m_{t-1} + (1-\beta)\tilde{g}_t$
- **Cosine filter:** Nếu $\cos(\hat{g}_t, \tilde{g}_{t-1}) < \tau$ thì reject.
- **Trust scaling:** $g = trust \cdot m$

---

### 4. Ý nghĩa & tác động tham số

| Tham số      | Vai trò toán học | Tăng lên           | Giảm xuống         |
|--------------|------------------|--------------------|--------------------|
| C            | Clip norm        | Ít bias, robust    | Mất thông tin      |
| epsilon      | Privacy strength | Noise ↓, acc ↑     | Noise ↑, privacy ↑ |
| delta        | Privacy slack    | Ít ảnh hưởng       | Noise ↑ nhẹ        |
| sigma_0      | Noise ban đầu    | Robust ↑           | Acc ↑              |
| kappa        | Decay speed      | Converge nhanh     | Noise giữ lâu      |
| beta_strat   | Strat/DP ratio   | Robust ↑           | DP dominate        |
| alpha_t      | Attack noise     | Robust ↑           | Ít bảo vệ          |
| sigma_total  | Energy cap       | Linh hoạt          | Dễ clamp           |
| nsr_warn     | NSR threshold    | Cảnh báo muộn      | Cảnh báo sớm       |
| gamma        | EMA smoothing    | Mượt, ổn định      | Nhạy, dao động     |
| beta         | Momentum         | Ổn định            | Dao động           |
| tau          | Cosine filter    | Reject nhiều       | Nhận nhiều         |

---

### 5. Bộ tham số mẫu & hướng dẫn tuning

#### MNIST (simple, low noise)
| Tham số     | Giá trị đề xuất |
|-------------|----------------|
| C           | 1.0            |
| epsilon     | 5 – 10         |
| delta       | 1e-5           |
| sigma_0     | 1.0            |
| kappa       | 0.01           |
| beta_strat  | 0.2 – 0.4      |
| alpha_t     | 0.1            |
| sigma_total | 2.0            |
| gamma       | 0.9            |
| beta        | 0.9            |
| tau         | 0.2            |

#### CIFAR-10 (medium complexity)
| Tham số     | Giá trị đề xuất |
|-------------|----------------|
| C           | 1.0 – 2.0      |
| epsilon     | 3 – 6          |
| delta       | 1e-5           |
| sigma_0     | 1.5 – 2.0      |
| kappa       | 0.02           |
| beta_strat  | 0.4 – 0.6      |
| alpha_t     | 0.2 – 0.3      |
| sigma_total | 3.0            |
| gamma       | 0.95           |
| beta        | 0.9            |
| tau         | 0.3            |

#### DFL (adversarial)
| Tham số     | Giá trị đề xuất |
|-------------|----------------|
| C           | 1.0            |
| epsilon     | 1 – 3          |
| delta       | 1e-5           |
| sigma_0     | 2.0 – 3.0      |
| kappa       | 0.005          |
| beta_strat  | 0.6 – 1.0      |
| alpha_t     | 0.3 – 0.6      |
| sigma_total | 3.0 – 5.0      |
| gamma       | 0.98           |
| beta        | 0.95           |
| tau         | 0.4 – 0.6      |

---

### 6. Hướng dẫn tuning thực tế

1. **Fix constraint:** Chọn epsilon, delta, C phù hợp yêu cầu privacy.
2. **Set noise scale:** Điều chỉnh sigma_0, sigma_total để cân bằng robust/acc.
3. **Tune robustness:** Tăng alpha_t, beta_strat nếu gặp attack.
4. **Tune recovery:** Tăng gamma, beta nếu gradient nhiễu.

---

### 7. Lý do chọn công thức & độ phù hợp

- **Công thức DP & RDP:** Chuẩn lý thuyết, đảm bảo privacy auditing.
- **Adaptive scheduler:** Cho phép trade-off động giữa privacy, robust, acc.
- **Strategic noise:** Kết hợp nhiều hướng để vừa chống attack vừa giữ tín hiệu.
- **Constraint & NSR:** Đảm bảo hệ không bị noise quá lớn, cảnh báo kịp thời.
- **Recovery:** Giúp mô hình không bị trôi khi noise mạnh.

---

### 8. Failure modes & cảnh báo

- **Noise quá lớn:** NSR > 1, gradient nhiễu như random.
- **Noise quá nhỏ:** Dễ bị poisoning.
- **Constraint quá chặt:** Adaptive mất tác dụng.

---

### 9. Kết luận

Noise Game là hệ adaptive noise control, cho phép điều chỉnh linh hoạt giữa privacy, robustness, accuracy. Việc tuning đúng tham số là chìa khóa để đạt hiệu quả tối ưu cho từng kịch bản.

Nếu cần nâng cấp, nên bổ sung ablation study, sensitivity analysis, stability proof.


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

3. Lower bound chuẩn cho sigma_DP:
   - Với một mục tiêu `(\varepsilon, \delta)`-DP, công thức chuẩn là:
     ```math
     \sigma_{DP,t} \ge \frac{C \sqrt{2 \ln(1.25/\delta)}}{\varepsilon}
     ```
   - Bản thực hiện giữ nguyên nguyên lý này như base floor để không giảm quá thấp sigma_DP.

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

3. Spectrum-aware noise:
   - Dựng noise theo phân tích phổ của gradient.
   - Dạng triển khai dùng giả SVD/truncated SVD và nghịch đảo giá trị riêng để tạo noise trên không gian phổ:
     ```math
     n_{spec} = U \cdot \mathrm{diag}(\lambda^{-1}) \cdot r
     ```
   - `r` là vector ngẫu nhiên, `\lambda` là các giá trị riêng.
   - Tính năng này giúp noise chú trọng tới các thành phần phổ yếu hơn, làm giảm thiểu tổn thất thông tin ở các không gian có tín hiệu lớn.

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

Báo cáo này đã làm rõ từng bước thực hiện, lý do chọn tham số và các trade-off chính. Nếu cần, bước tiếp theo có thể là đánh giá thực nghiệm với metrics `avg_nsr`, `avg_sigma_dp`, và so sánh accuracy/trust trên CIFAR-10/MNIST.
