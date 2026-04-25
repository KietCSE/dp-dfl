# Chèn Noise Chiến lược với Quyền riêng tư RDP cho DFL Bền vững

---

# 1. Giới thiệu

Federated Learning phân tán (Decentralized Federated Learning - DFL) cho phép nhiều client hợp tác huấn luyện mô hình mà không cần máy chủ trung tâm. Tuy nhiên, kiến trúc phân tán cũng mở ra các rủi ro bảo mật lớn:

- Tấn công nhiễu gradient (model poisoning) làm sai lệch quá trình tối ưu hóa.
- Rò rỉ thông tin qua gradient (inference attack).

Differential Privacy (DP) có thể giảm rò rỉ bằng cách thêm noise vào gradient. Tuy nhiên, các phương pháp DP tiêu chuẩn thường:

- thêm noise đồng nhất, không xét đến cấu trúc của gradient,
- yếu trước các cập nhật độc hại,
- làm giảm độ chính xác khi noise quá lớn.

Tài liệu này trình bày một khung 4 lớp kết hợp:

- **DP để bảo vệ quyền riêng tư**,
- **RDP để điều khiển cường độ noise**,
- **noise chiến lược để tăng robust và giữ accuracy**.

---

# 2. Ý tưởng cốt lõi: Thiết kế noise theo lý thuyết game

## 2.1 Mô hình tấn công và phòng vệ

Cho:

- gradient sạch: `g`,
- perturbation độc hại: `\delta`,
- noise thêm vào: `n`.

Gradient quan sát được là:

```math
\hat{g} = g + \delta + n
```

Mục tiêu là chọn `n` để giảm thiểu tác động của `\delta` mà vẫn duy trì privacy.

## 2.2 Bài toán Minimax

Bài toán này có thể biểu diễn như sau:

```math
\min_n \max_\delta \|\delta + n\|
```

Nghĩa là defender muốn chọn noise sao cho khi attacker tối đa hoá perturbation, tổng nhiễu vẫn nhỏ.

## 2.3 Tách noise thành hai thành phần

Chúng ta tách noise thành:

```math
n = n_{DP} + n_{strategic}
```

Trong đó:

- `n_{DP}` đảm bảo **privacy** theo DP,
- `n_{strategic}` đảm bảo **robustness** và giảm thiểu tổn thất độ chính xác.

## 2.4 Nguyên lý điều khiển

Privacy và robustness được tách biệt chức năng, nhưng vẫn phải được điều phối chung qua ngân sách noise.

---

# 3. Kiến trúc hệ thống 4 lớp

## 3.1 Tổng quan các lớp

| Lớp   | Chức năng chính                            |
| ----- | ------------------------------------------ |
| Lớp 1 | Bảo đảm DP bằng clipping và Gaussian noise |
| Lớp 2 | Điều khiển cường độ noise theo RDP         |
| Lớp 3 | Thiết kế noise chiến lược với cấu trúc     |
| Lớp 4 | Khôi phục tín hiệu và cập nhật tối ưu hóa  |

## 3.2 Luồng xử lý tổng thể

```text
g_t
 ↓
[Layer 1] Clipping → gbar_t
 ↓
Lấy noise DP: n_DP ~ N(0, sigma_DP_t^2 I)
 ↓
[Layer 2] Tính sigma_strat_t
 ↓
[Layer 3] Tính \hat{n}_t
 ↓
n_strategic = sigma_strat_t · hat{n}_t
 ↓
\hat{g}_t = gbar_t + n_DP + n_strategic
 ↓
[Layer 4] Denoise + cập nhật
```

## 3.3 Mục tiêu thiết kế

Khung 4 lớp giúp tách rõ:

- **Privacy** ở tầng 1,
- **Control** ở tầng 2,
- **Geometry** ở tầng 3,
- **Recovery** ở tầng 4.

---

# 4. Lớp 1 — Differential Privacy Layer

## 4.1 Clipping gradient

Gradient đầu vào được chuẩn hoá để giới hạn sensitivity:

```math
g_t \leftarrow \frac{g_t}{\max\left(1, \frac{\|g_t\|}{C}\right)}
```

Việc này đảm bảo:

```math
\|g_t\| \le C
```

Trong đó `C` là ngưỡng clipping. Điều này ngăn gradient có độ lớn quá cao làm lệch bảo đảm privacy.

## 4.2 Cơ chế Gaussian

Noise DP được thêm vào bằng Gaussian:

```math
n_{DP,t} \sim \mathcal{N}(0, \sigma_{DP,t}^2 I)
```

Đây là phần noise chịu trách nhiệm chính cho tính chất DP.

## 4.3 DP guarantee

Để đạt $(\varepsilon, \delta)$-DP sau clipping, cần:

```math
\sigma_{DP,t} \ge \frac{C \sqrt{2 \ln(1.25/\delta)}}{\varepsilon}
```

Công thức này là điều kiện chuẩn của Gaussian mechanism.

## 4.4 Vai trò của Layer 1

Layer 1 bảo đảm mỗi gradient đầu vào có sensitivity có hạn và cung cấp nguồn noise DP. Các lớp sau không thể thay thế guarantee này.

---

# 5. Lớp 2 — RDP-based Noise Control

## 5.1 RDP accounting

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

---

# 6. Lớp 3 — Strategic Noise Design

## 6.1 Biến thiên gradient

Giữa hai vòng liên tiếp, sử dụng biến thiên gradient:

```math
v_t = g_t - g_{t-1}
```

`v_t` thường chứa tín hiệu về thay đổi bất thường hoặc tấn công.

## 6.2 Trust score

Độ tin cậy được xác định bằng cosine similarity:

```math
trust_t = \frac{g_t \cdot g_{t-1}}{\|g_t\| \cdot \|g_{t-1}\|}
```

Giá trị gần `1` nghĩa là gradient ổn định, giá trị thấp có thể báo hiệu tấn công.

## 6.3 Noise hướng attack

Một thành phần noise nhắm theo hướng attack:

```math
n_{attack} = \alpha_t \cdot v_t
```

`\alpha_t` điều khiển cường độ noise theo hướng biến thiên.

## 6.4 Noise vuông góc

Một thành phần noise được thiết kế để không làm lệch hướng descent chính:

```math
n_{orth} = z - \frac{z \cdot g_t}{\|g_t\|^2} g_t
```

Trong đó `z` là vector ngẫu nhiên.

## 6.5 Noise phổ-aware

Phân tích gradient theo phổ:

```math
g_t = U \Lambda V^T
```

Noise phổ-aware có thể được xây dựng như:

```math
n_{spec} = U \cdot \operatorname{diag}(\lambda^{-1}) \cdot r
```

Trong đó `r` là vector ngẫu nhiên và `\lambda` là các giá trị riêng.

## 6.6 Kết hợp hướng noise

Noise tổng hợp được chuẩn hoá:

```math
\hat{n}_t = \frac{n_{attack} + n_{orth} + n_{spec}}{\|n_{attack} + n_{orth} + n_{spec}\|}
```

## 6.7 Noise chiến lược cuối cùng

```math
n_{strategic} = \sigma_{strat,t} \cdot \hat{n}_t
```

Layer 3 làm nhiệm vụ chọn hướng noise sao cho vừa chống tấn công vừa giữ lại thông tin gradient quan trọng.

---

# 7. Lớp 4 — Tối ưu hóa và khôi phục tín hiệu

## 7.1 Làm mượt với EMA

Gradient noisy được làm mượt bằng EMA:

```math
\tilde{g}_t = \gamma \tilde{g}_{t-1} + (1-\gamma) \hat{g}_t
```

`\gamma` điều khiển mức trễ của bộ lọc.

## 7.2 Momentum

Sử dụng momentum để ổn định cập nhật:

```math
m_t = \beta m_{t-1} + (1-\beta) \tilde{g}_t
```

Momentum giúp giảm dao động do noise.

## 7.3 Lọc gradient bất thường

Một điều kiện loại bỏ gradient nếu nó lệch quá xa lịch sử:

```math
\cos(\hat{g}_t, \tilde{g}_{t-1}) < \tau \Rightarrow \text{reject}
```

Trong đó `\tau` là ngưỡng tương quan.

## 7.4 Cập nhật tham số

Sau khi xử lý, cập nhật mô hình:

```math
w_{t+1} = w_t - \eta m_t
```

`\eta` là learning rate.

## 7.5 Vai trò của Layer 4

Layer 4 giữ cho quá trình huấn luyện ổn định bằng cách khôi phục tín hiệu từ gradient đã nhiễu.

---

# 8. Cơ chế tăng độ chính xác

## 8.1 Giảm noise theo thời gian

Noise DP có thể giảm dần để giữ accuracy khi mô hình đã ổn định:

```math
\sigma_{DP,t} = \sigma_0 e^{-\kappa t}
```

## 8.2 Learning rate theo trust

Learning rate điều chỉnh theo độ tin cậy:

```math
\eta_i = \eta \cdot trust_i
```

## 8.3 Giảm phương sai

Biện pháp variance reduction:

```math
g_i' = g_i - c_i + c
```

## 8.4 Mô hình hai luồng

Kết hợp hai tham số:

```math
w = \lambda w^{clean} + (1-\lambda) w^{robust}
```

Giúp cân bằng giữa độ chính xác và độ bền.

---

# 9. Mô hình noise thống nhất

## 9.1 Gradient cuối cùng

```math
\hat{g}_t = g_t + n_{DP,t} + n_{strategic,t}
```

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
