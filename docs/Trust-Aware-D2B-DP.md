# Cấu trúc Thuật toán D2B - High Accuracy Pipeline (RMS & Softmax Version)

Thuật toán D2B (Decentralized to Byzantine-robust) giải quyết bài toán Học Liên kết Phi tập trung (DFL) trong môi trường có nhiễu Differential Privacy (DP) và dữ liệu phân bố Non-IID, đồng thời loại bỏ "Lời nguyền số chiều" (Curse of Dimensionality) bằng chuẩn hóa RMS.

## 1. Bảng Siêu tham số (Hyper-parameters) & Cấu hình Đề xuất

| Ký hiệu         | Ý nghĩa / Tên gọi                                               | Phạm vi                | Giá trị Đề xuất                            |
| :-------------- | :-------------------------------------------------------------- | :--------------------- | :----------------------------------------- |
| $k$             | Kích thước cửa sổ hàng đợi tính ngưỡng $Clip_l$                 | $[1, 20]$              | $5$ hoặc $10$                              |
| $\rho_{min}$    | Ngân sách quyền riêng tư (Privacy Budget) khởi điểm             | $(0, 1.0]$             | $0.1$                                      |
| $\rho_{max}$    | Ngân sách quyền riêng tư tối đa ở cuối kỳ                       | $[\rho_{min}, \infty)$ | $1.0$ hoặc $2.0$                           |
| $\beta$         | Tốc độ nới lỏng ngân sách DP theo thời gian                     | $(0, 0.1]$             | $0.01$                                     |
| $\theta$        | Hệ số dung sai DP (DP Tolerance Factor)                         | $[1.0, 1.5]$           | $1.1$                                      |
| $\gamma$        | Hệ số nới lỏng ngưỡng $D_{threshold}$ ban đầu                   | $[1.5, 5.0]$           | $2.5$                                      |
| $\kappa$        | Tốc độ suy giảm hàm mũ của ngưỡng khoảng cách                   | $[3.0, 10.0]$          | $5.0$                                      |
| $\alpha_T$      | Tham số nhớ (Memory Factor) của bộ lọc EMA                      | $[0.5, 0.99]$          | $0.85$                                     |
| $T_{min}$       | Ngưỡng điểm Trust tối thiểu để lọt vào tập an toàn $\mathbb{V}$ | $[0.1, 0.5]$           | $0.3$ hoặc $0.4$                           |
| $\beta_{soft}$  | Nhiệt độ Softmax (Softmax Temperature)                          | $[1.0, 10.0]$          | $3.0$                                      |
| $\beta_m$       | Tham số Momentum toàn cục (Global Momentum)                     | $[0.5, 0.99]$          | $0.9$                                      |
| $\eta_{global}$ | Tỷ lệ học của hệ thống (Global Learning Rate)                   | $> 0$                  | Thường chọn $\eta_{global} = \eta_{local}$ |

---

## 2. Chi tiết Quy trình Thuật toán tại Vòng $t$

Dưới đây mô tả các bước thực thi tại Node $i$ trong vòng giao tiếp (round) thứ $t$.

### Giai đoạn 1: Local Training (Huấn luyện Cục bộ)

**Bước 1: Cập nhật Model Delta**
Node $i$ thực hiện huấn luyện (ví dụ: dùng SGD) trên tập dữ liệu cục bộ, thu được trọng số mới $W_{trained}$. Vector khác biệt (Model Delta) chứa tri thức mới được tính bằng:
$$\Delta W^{(t)} = W_{trained} - W_i^{(t-1)}$$

- $W_i^{(t-1)}$: Trọng số của Node $i$ ở đầu vòng $t$.

---

### Giai đoạn 2: Outbound Processing (Xử lý Gói tin Đầu ra)

**Bước 2: Layer-wise Adaptive Clipping (Cắt tỉa theo Layer)**
Để tránh việc các layer có gradient lớn lấn át các layer nhỏ, ta giới hạn độ lớn cập nhật cho từng layer $l$:
$$Clip_l = \frac{1}{k} \sum_{m=1}^k H_l[m]$$
$$\Delta W'_{l} = \frac{\Delta W_{l}^{(t)}}{\max\left(1, \frac{\|\Delta W_{l}^{(t)}\|_2}{Clip_l}\right)}$$

- $H_l$: Hàng đợi lịch sử chứa chuẩn L2 của layer $l$ trong $k$ vòng gần nhất (Ở vòng 1, $Clip_l = \|\Delta W_{l}^{(1)}\|_2$).
- $\Delta W'_{l}$: Trọng số cập nhật của layer $l$ sau khi bị giới hạn (clipping).

**Bước 3: Tiêm Nhiễu DP Layer-wise**
Bơm nhiễu Gaussian được tinh chỉnh phương sai theo từng layer dựa trên ngân sách bảo mật:
$$\rho^{(t)} = \min\left( (1 + \beta t)\rho_{min}, \; \rho_{max} \right)$$
$$\sigma_{l, (t)}^2 = \frac{2 \cdot (Clip_l)^2}{\rho^{(t)}}$$
$$\tilde{W}_{i, l} = \Delta W'_{l} + B(x_l) \quad \text{với } x_l \sim \mathcal{N}(0, \sigma_{l, (t)}^2)$$

- $\sigma_{l, (t)}^2$: Phương sai nhiễu cấp phát cho layer $l$.
- $B(x_l)$: Hàm chặn giới hạn nhiễu (tránh giá trị ngoại lai).
  Vector $\tilde{W}_i$ gồm tất cả các layer được đóng gói thành gói tin $S_i$ và gửi cho hàng xóm.

---

### Giai đoạn 3: Inbound Evaluation (Đánh giá Đầu vào Chuẩn hóa RMS)

Gọi $S_j$ là gói tin nhận được từ hàng xóm $j$. Node $i$ so sánh $S_j$ với vector gốc cục bộ $\Delta W'_i$. Gọi $D_{total} = \sum_{l=1}^L d_l$ là tổng số lượng tham số của toàn bộ mô hình.

**Bước 4: Đánh giá Đa chiều (RMS \& Cosine)**
$$D_{i,j}^{(t)} = \frac{\|\Delta W'_i - S_j\|_2}{\sqrt{D_{total}}}$$
$$C_{i,j}^{(t)} = \frac{\langle \Delta W'_i, S_j \rangle}{\|\Delta W'_i\|_2 \cdot \|S_j\|_2} \quad \in [-1, 1]$$

- $D_{i,j}^{(t)}$: Khoảng cách chuẩn hóa RMS (khoảng cách trung bình trên mỗi tham số).
- $C_{i,j}^{(t)}$: Độ tương đồng Cosine (hướng của vector).

**Bước 5: Xác định Ngưỡng Dị thường RMS**
Xây dựng ngưỡng động để chấm điểm, tránh bị bùng nổ bởi số chiều:
$$C_{DP}^{(t)} = \theta \cdot \sqrt{\frac{1}{D_{total}} \sum_{l=1}^L d_l \cdot \sigma_{l, (t)}^2}$$
$$D_{threshold}^{(t)} = \max \left( \gamma \cdot \exp\left(-\kappa \cdot \frac{t}{T_{max}}\right) \cdot \frac{\|\Delta W'_i\|_2}{\sqrt{D_{total}}}, \; C_{DP}^{(t)} \right)$$

- $C_{DP}^{(t)}$: Mức sàn nhiễu DP (chuẩn hóa RMS).
- $D_{threshold}^{(t)}$: Ngưỡng khoảng cách suy giảm theo tiến trình huấn luyện, chốt chặn dưới tại $C_{DP}^{(t)}$.

---

### Giai đoạn 4: History & Aggregation (Tổng hợp Mềm \& Momentum)

**Bước 6: Quản lý Lịch sử Danh tiếng (Continuous Trust Score)**
Chấm điểm từng gói tin bằng hàm liên tục thay vì nhị phân Pass/Fail:
$$p_{dist} = \exp\left( - \frac{D_{i,j}^{(t)}}{D_{threshold}^{(t)}} \right) \quad \in (0, 1]$$
$$p_{cos} = \max\left(0, \; C_{i,j}^{(t)}\right) \quad \in [0, 1]$$
$$Q_{i,j}^{(t)} = p_{dist} \cdot p_{cos}$$
$$T_{i,j}^{(t)} = \alpha_T \cdot T_{i,j}^{(t-1)} + (1 - \alpha_T) \cdot Q_{i,j}^{(t)}$$

- $p_{dist}$: Điểm suy giảm hàm mũ theo khoảng cách.
- $p_{cos}$: Màng lọc ReLU triệt tiêu hoàn toàn các vector đi ngược hướng.
- $Q_{i,j}^{(t)}$: Điểm chất lượng tức thời của vòng hiện tại.
- $T_{i,j}^{(t)}$: Điểm Tin cậy (Trust Score) tổng hợp theo EMA. Khởi tạo $T_{i,j}^{(0)} = 1.0$.

**Bước 7: Softmax Aggregation \& Momentum Updates**
Lọc ra tập hàng xóm an toàn $\mathbb{V} = \{ j \mid T_{i,j}^{(t)} \ge T_{min} \}$. Chuyển đổi điểm Trust thành trọng số phần trăm để tổng hợp:
$$w_{i,j}^{(t)} = \frac{\exp(\beta_{soft} \cdot T_{i,j}^{(t)})}{\sum_{k \in \mathbb{V}} \exp(\beta_{soft} \cdot T_{i,k}^{(t)})}$$
$$S_{agg}^{(t)} = \sum_{j \in \mathbb{V}} w_{i,j}^{(t)} \cdot S_j$$

- $w_{i,j}^{(t)}$: Trọng số Softmax. Node có Trust cao sẽ chi phối bản cập nhật, node yếu bị cách ly mềm.

Cuối cùng, cập nhật vận tốc Momentum toàn cục và trọng số mô hình:
$$V_{agg}^{(t)} = \beta_m \cdot V_{agg}^{(t-1)} + (1 - \beta_m) \cdot S_{agg}^{(t)}$$
$$W_i^{(t)} = W_i^{(t-1)} - \eta_{global} \cdot V_{agg}^{(t)}$$

- $V_{agg}^{(t)}$: Quỹ đạo học (khởi tạo bằng $0$). Việc dùng Softmax ở trên đảm bảo không có nhiễu độc hại lọt vào làm hỏng quỹ đạo này.
