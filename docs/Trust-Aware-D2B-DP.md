## Tổng quan Thuật toán (Algorithm Overview)

Thuật toán **Trust Aware** là một 파ipeline học máy phân tán (Decentralized/Federated Learning) được thiết kế nhằm đạt được 3 mục tiêu đồng thời:

1.  **Bảo vệ quyền riêng tư (Privacy-Preserving):** Thông qua cơ chế Layer-wise Adaptive Clipping và thêm nhiễu Gaussian chuẩn (Differential Privacy - DP).
2.  **Kháng lỗi/Tấn công (Byzantine Robustness):** Loại bỏ các node độc hại (gửi gradient sai hoặc bị nhiễu quá nặng) thông qua hệ thống đánh giá đa chiều (Cosine, RMSE) và quản lý danh tiếng (Trust Score).
3.  **Hội tụ ổn định:** Sử dụng Softmax Aggregation và Global Momentum để mượt mà hóa quỹ đạo cập nhật mô hình.

---

## Bảng Chú thích Ký hiệu (Notations)

- $t$: Chỉ số vòng giao tiếp (Communication round).
- $W_i^{(t)}$: Trọng số mô hình cục bộ của node $i$ tại vòng $t$.
- $\Delta W^{(t)}$: Vector cập nhật (Model Delta) chứa tri thức mới.
- $L$: Tổng số lượng layer của mô hình.
- $l$: Chỉ số của một layer cụ thể ($l \in [1, L]$).
- $d_l$: Số lượng tham số của layer $l$.
- $D_{total}$: Tổng số lượng tham số của toàn bộ mô hình ($D_{total} = \sum d_l$).
- $Clip_l$: Ngưỡng cắt tỉa (clipping bound) động của layer $l$.
- $\sigma_{l, (t)}^2$: Phương sai nhiễu DP tiêm vào layer $l$ tại vòng $t$.
- $S_j$: Gói tin trọng số (đã thêm nhiễu) nhận được từ node hàng xóm $j$.
- $T_{i,j}^{(t)}$: Điểm tin cậy (Trust Score) tổng hợp của node $j$ trong mắt node $i$.

---

## Chi tiết Thuật toán Step-by-Step

### Giai đoạn 1: Local Training (Huấn luyện Cục bộ)

**Bước 1: Huấn luyện và tính Model Delta**
Node $i$ thực hiện quá trình huấn luyện mô hình (ví dụ: dùng SGD/Adam) trên tập dữ liệu cục bộ của mình để thu được trọng số mới $W_{trained}$. Sau đó, trích xuất lượng tri thức học được:
$$\Delta W^{(t)} = W_{trained} - W_i^{(t-1)}$$

---

### Giai đoạn 2: Outbound Processing (Xử lý trước khi gửi)

**Bước 2: Layer-wise Adaptive Clipping**
Thay vì gọt giũa (clip) toàn bộ mô hình cùng lúc (có thể làm mất thông tin của các layer nhỏ), thuật toán tách riêng từng layer. Dùng hàng đợi $H_l$ kích thước $k$ lưu lịch sử chuẩn L2 của layer $l$ để tính ngưỡng động:
$$Clip_l = \frac{1}{k} \sum_{m=1}^k H_l[m]$$
_(Lưu ý: Ở vòng đầu tiên, $Clip_l = \|\Delta W_{l}^{(1)}\|_2$)_

Tiến hành cắt tỉa layer $l$:
$$\Delta W'_{l} = \frac{\Delta W_{l}^{(t)}}{\max\left(1, \frac{\|\Delta W_{l}^{(t)}\|_2}{Clip_l}\right)}$$

**Bước 3: Bơm nhiễu Gaussian Chuẩn (DP)**
Tính phương sai nhiễu động, giảm dần theo thời gian (để ưu tiên hội tụ ở các vòng cuối):
$$\sigma_{l, (t)}^2 = \frac{2 \cdot (Clip_l)^2}{\rho^{(t)}}$$
Trong đó $\rho^{(t)} = \min\left( (1 + \beta t)\rho_{min}, \rho_{max} \right)$.

Sinh nhiễu và cộng trực tiếp vào layer:
$$x_l \sim \mathcal{N}(0, \sigma_{l, (t)}^2)$$
$$\tilde{W}_{i, l} = \Delta W'_{l} + x_l$$
Cuối cùng, đóng gói tất cả các $\tilde{W}_{i, l}$ thành vector $\tilde{W}_i$ và gửi cho các node hàng xóm.

---

### Giai đoạn 3: Inbound Evaluation (Đánh giá Gói tin đến)

**Bước 4: Đánh giá Đa chiều**
Khi nhận được vector $S_j$ từ node $j$, node $i$ so sánh nó với vector cục bộ $\Delta W'_i$ (trước khi bản thân thêm nhiễu) qua 2 khía cạnh:

1.  _Hướng (Cosine Similarity):_ Xem node $j$ có đang đi cùng hướng tối ưu không.
    $$C_{i,j}^{(t)} = \frac{\langle \Delta W'_i, S_j \rangle}{\|\Delta W'_i\|_2 \cdot \|S_j\|_2}$$
2.  _Khoảng cách RMSE (Normalized L2):_ Khắc phục lời nguyền chiều dữ liệu bằng cách chia cho $\sqrt{D_{total}}$.
    $$D_{i,j}^{(t)} = \frac{1}{\sqrt{D_{total}}} \|\Delta W'_i - S_j\|_2$$

**Bước 5: Xác định Ngưỡng Dị thường**
Tính mức sàn nhiễu DP (độ rung lắc hợp lệ tối thiểu do chính nhiễu Gaussian gây ra):
$$C_{DP}^{(t)} = \theta \cdot \sqrt{ \frac{1}{D_{total}} \sum_{l=1}^L d_l \cdot \sigma_{l, (t)}^2 }$$
Tính ngưỡng khoảng cách động (suy giảm theo thời gian nhưng không bao giờ thủng mức sàn $C_{DP}$):
$$D_{threshold}^{(t)} = \max \left( \gamma \cdot \exp(-\kappa \cdot \lambda(t)) \cdot \frac{\|\Delta W'_i\|_2}{\sqrt{D_{total}}}, \; C_{DP}^{(t)} \right)$$

---

### Giai đoạn 4: History & Aggregation (Tổng hợp & Cập nhật)

**Bước 6: Quản lý Lịch sử Danh tiếng**
Chấm điểm chất lượng tức thời của gói tin $j$ tại vòng $t$:
$$p_{dist} = \exp\left( - \frac{D_{i,j}^{(t)}}{D_{threshold}^{(t)}} \right)$$
$$p_{cos} = \max\left(0, \; C_{i,j}^{(t)}\right)$$
$$Q_{i,j}^{(t)} = p_{dist} \cdot p_{cos}$$
_(Bộ lọc ReLU ở $p_{cos}$ sẽ trừng phạt thẳng tay điểm $0$ cho bất kỳ node nào có dấu hiệu Sign-flipping attack).\_

Cập nhật hàm mũ danh tiếng (EMA - Exponential Moving Average) để có cái nhìn dài hạn, khởi tạo $T_{i,j}^{(0)} = 1.0$:
$$T_{i,j}^{(t)} = \alpha_T \cdot T_{i,j}^{(t-1)} + (1 - \alpha_T) \cdot Q_{i,j}^{(t)}$$

**Bước 7: Softmax Aggregation & Global Momentum**
Lọc tập hợp các node hợp lệ $\mathbb{V}$ thỏa mãn $T_{i,j}^{(t)} \ge T_{min}$. Dùng Softmax để tính trọng số đóng góp dựa trên độ tin cậy:
$$w_{i,j}^{(t)} = \frac{\exp(\beta_{soft} \cdot T_{i,j}^{(t)})}{\sum_{k \in \mathbb{V}} \exp(\beta_{soft} \cdot T_{i,k}^{(t)})}$$

Tổng hợp các vector:
$$S_{agg}^{(t)} = \sum_{j \in \mathbb{V}} w_{i,j}^{(t)} \cdot S_j$$

Áp dụng Global Momentum để làm mượt quỹ đạo, sau đó cập nhật thẳng vào mô hình gốc:
$$V_{agg}^{(t)} = \beta_m \cdot V_{agg}^{(t-1)} + (1 - \beta_m) \cdot S_{agg}^{(t)}$$
$$W_i^{(t)} = W_i^{(t-1)} + V_{agg}^{(t)}$$

---

## Gợi ý Cấu hình Siêu tham số (Hyperparameter Configurations)

Để thuật toán hội tụ mượt mà và chống lại nhiễu tốt, dưới đây là các mức cấu hình (config) thực nghiệm được đề xuất dựa trên các hệ thống học phân tán tiêu chuẩn:

### 1. Adaptive Clipping & Privacy (Giai đoạn 2)

- **$k$ (Kích thước cửa sổ lịch sử Clipping):** `5` đến `10`. Không nên để quá lớn vì mô hình thay đổi nhanh ở các vòng đầu, lịch sử cũ sẽ làm sai lệch ngưỡng.
- **$\rho_{min}$ (Ngân sách nhiễu ban đầu):** Phụ thuộc vào mức độ bảo mật yêu cầu (thường tỷ lệ thuận với $\epsilon$). Khuyến nghị ở mức `0.1 - 0.5`.
- **$\rho_{max}$ (Giới hạn nhiễu tối đa):** `5.0 - 10.0`. Chặn không cho mô hình bị thiếu nhiễu hoàn toàn ở cuối kỳ.
- **$\beta$ (Tốc độ giảm nhiễu DP):** `0.01 - 0.05`. Tùy vào tổng số vòng $T$.

### 2. Thresholds & Evaluation (Giai đoạn 3)

- **$\theta$ (Hệ số đệm của mức sàn DP):** `1.05 - 1.2` (Khuyến nghị: `1.1`). Tạo một biên độ an toàn $10\%$ để không phạt nhầm node trung thực bị rung lắc do chính thuật toán gây ra.
- **$\gamma$ (Scale ban đầu của ngưỡng động):** `3.0 - 5.0`. Ở những vòng đầu, đạo hàm rất lớn và biến động, cần nới lỏng ngưỡng này.
- **$\kappa$ (Tốc độ siết ngưỡng theo hàm mũ):** `0.05 - 0.1` (Nếu $\lambda(t)$ là epoch number). Đảm bảo ở khoảng $30\% - 50\%$ thời gian huấn luyện, ngưỡng tiệm cận về sát mức sàn $C_{DP}$.

### 3. Trust Score & Aggregation (Giai đoạn 4)

- **$\alpha_T$ (Hệ số bảo lưu EMA Trust):** `0.8 - 0.9`. Cực kỳ quan trọng. Trọng số $\alpha_T$ cao giúp hệ thống nhớ "nhân cách tốt" của node lâu hơn, tránh việc một node tốt bị block chỉ vì một vòng gửi gradient nhiễu (khuyên dùng `0.85`).
- **$T_{min}$ (Ngưỡng sinh tử - Cut-off Threshold):** `0.3 - 0.4`. Dưới mức này sẽ bị xem là Byzantine/Malignant và bị loại khỏi $\mathbb{V}$.
- **$\beta_{soft}$ (Nhiệt độ Softmax):** `5.0 - 10.0`. Nếu để $=1$, phân phối trọng số sẽ rất đều (ai cũng được đóng góp na ná nhau). Nâng $\beta_{soft}$ lên cao (ví dụ `10`) sẽ có tác dụng "The winner takes all", dồn quyền lực lớn cho những node có điểm trust $T$ cao nhất.
- **$\beta_m$ (Hệ số Global Momentum):** `0.85 - 0.95` (Khuyến nghị tiêu chuẩn: `0.9`). Giúp mô hình có quán tính vững vàng, triệt tiêu được các xung nhiễu DP còn sót lại sau khi aggregate.
