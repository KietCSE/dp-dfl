## 1. Vòng đời của một Node (Pipeline Overview)

Trong mạng phi tập trung (Decentralized Network), tại mỗi vòng giao tiếp $t$, một node trung thực $i$ sẽ thực thi tuần tự 4 giai đoạn cốt lõi:

1. **Local Training:** Khai phá tri thức từ dữ liệu cục bộ.
2. **Outbound Processing:** Định hình (clipping) và ẩn danh hóa (tiêm nhiễu DP) cập nhật trước khi phát sóng.
3. **Inbound Evaluation:** Sàng lọc và đánh giá độ tin cậy của các gói tin nhận được từ mạng lưới.
4. **History & Aggregation:** Cập nhật sổ xưng danh (Trust Score), tổng hợp có trọng số và áp dụng Momentum.

---

## 2. Chi tiết Thuật toán

### Giai đoạn 1: Local Training (Huấn luyện \& Tính Delta)

Node $i$ thực hiện huấn luyện mô hình cục bộ (ví dụ: bằng SGD) trên tập dữ liệu riêng rẽ, bắt đầu từ trọng số toàn cục của vòng trước $W_i^{(t-1)}$.

Sau khi huấn luyện, node thu được trọng số mới $W_{trained}$. Thay vì gửi toàn bộ trọng số, node chỉ trích xuất phần tri thức mới học được (Model Delta):
$$\Delta W^{(t)} = W_{trained} - W_i^{(t-1)}$$

### Giai đoạn 2: Outbound Processing (Bảo vệ Quyền riêng tư)

Để chống lại các cuộc tấn công khôi phục dữ liệu (Data Reconstruction Attacks) đồng thời tránh làm hỏng các layer có gradient nhỏ, hệ thống áp dụng **Layer-wise Adaptive Clipping** và tiêm nhiễu.

**Bước 2.1: Layer-wise Adaptive Clipping**
Với mỗi layer $l$, ngưỡng cắt tỉa động $Clip_l$ được tính dựa trên lịch sử trung bình chuẩn L2 của chính layer đó trong $k$ vòng gần nhất (sử dụng hàng đợi $H_l$):
$$Clip_l = \frac{1}{k} \sum_{m=1}^k H_l[m]$$
_(Lưu ý: Tại vòng 1, $Clip_l = \|\Delta W_{l}^{(1)}\|_2$)_

Vector cập nhật của layer $l$ sau đó được đưa về giới hạn chuẩn:
$$\Delta W'_{l} = \frac{\Delta W_{l}^{(t)}}{\max\left(1, \frac{\|\Delta W_{l}^{(t)}\|_2}{Clip_l}\right)}$$

**Bước 2.2: Tiêm Nhiễu Gaussian Chuẩn**
Phương sai nhiễu cho từng layer được hiệu chỉnh theo ngân sách $\rho^{(t)}$ (tăng dần theo thời gian để giảm dần nhiễu khi mô hình đã hội tụ):
$$\sigma_{l, (t)}^2 = \frac{2 \cdot (Clip_l)^2}{\rho^{(t)}}$$
Vớii $\rho^{(t)} = \min\left( (1 + \beta t)\rho_{min}, \rho_{max} \right)$.
Nhiễu Gaussian $x_l \sim \mathcal{N}(0, \sigma_{l, (t)}^2)$ được cộng trực tiếp vào trọng số đã cắt tỉa:
$$\tilde{W}_{i, l} = \Delta W'_{l} + x_l$$

**Bước 2.3: Đánh giá Rényi DP Accounting**
Mức độ rò rỉ quyền riêng tư (Privacy Loss) tại vòng $t$ với bậc $\alpha$ được kiểm soát chặt chẽ bởi layer chịu nhiễu yếu nhất so với độ nhạy (worst-case scenario):
$$\epsilon^{(t)}(\alpha) = \frac{\alpha}{2} \max_{l \in [1, L]} \left( \frac{Clip_l}{\sigma_{l, (t)}} \right)^2$$

Các layer được gộp lại thành vector cuối cùng $\tilde{W}_i$ và gửi đến các node láng giềng.

### Giai đoạn 3: Inbound Evaluation (Đánh giá Đa chiều)

Khi node $i$ nhận được gói tin $S_j$ từ node hàng xóm $j$, nó sẽ đối chiếu $S_j$ với vector cục bộ của mình (đã qua bước 2.1) là $\Delta W'_i$.

**Bước 3.1: Đo lường Hướng (Cosine Similarity)**
Kiểm tra xem node $j$ có đang tối ưu cùng hướng với node $i$ hay không:
$$C_{i,j}^{(t)} = \frac{\langle \Delta W'_i, S_j \rangle}{\|\Delta W'_i\|_2 \cdot \|S_j\|_2}$$

**Bước 3.2: Đo lường Khoảng cách Chuẩn hóa (RMSE)**
Khắc phục sự bùng nổ của nhiễu DP trong không gian nhiều chiều bằng cách chia cho tổng số lượng tham số $D_{total} = \sum d_l$:
$$D_{i,j}^{(t)} = \frac{1}{\sqrt{D_{total}}} \|\Delta W'_i - S_j\|_2$$

**Bước 3.3: Ngưỡng Dị thường Động (Dynamic Threshold)**
Hệ thống thiết lập một ranh giới khoảng cách an toàn. Ngưỡng này suy giảm theo hàm mũ để siết chặt dần khi mô hình hội tụ, nhưng không bao giờ rơi xuống dưới **Mức sàn DP Toán học ($C_{DP}^{(t)}$)** (vì nhiễu DP chắc chắn sẽ gây ra một độ lệch tối thiểu):
$$C_{DP}^{(t)} = \theta \cdot \sqrt{ \frac{1}{D_{total}} \sum_{l=1}^L d_l \cdot \sigma_{l, (t)}^2 }$$
$$D_{threshold}^{(t)} = \max \left( \gamma \cdot \exp(-\kappa \cdot \lambda(t)) \cdot \frac{\|\Delta W'_i\|_2}{\sqrt{D_{total}}}, \; C_{DP}^{(t)} \right)$$

### Giai đoạn 4: History \& Aggregation (Tổng hợp Tin cậy)

Thay vì loại bỏ cứng nhắc (hard threshold), hệ thống sử dụng điểm số liên tục để node trung thực bị lệch hướng do nhiễu không bị cấm vĩnh viễn, trong khi node Byzantine bị cô lập.

**Bước 4.1: Chấm điểm Tức thời (Instant Quality)**
$$p_{dist} = \exp\left( - \frac{D_{i,j}^{(t)}}{D_{threshold}^{(t)}} \right) \quad ; \quad p_{cos} = \max\left(0, \; C_{i,j}^{(t)}\right)$$
$$Q_{i,j}^{(t)} = p_{dist} \cdot p_{cos}$$

**Bước 4.2: Lịch sử Danh tiếng (EMA Trust Score)**
Cập nhật điểm tin cậy thông qua Trung bình Trượt Mũ (Exponential Moving Average):
$$T_{i,j}^{(t)} = \alpha_T \cdot T_{i,j}^{(t-1)} + (1 - \alpha_T) \cdot Q_{i,j}^{(t)}$$

**Bước 4.3: Softmax Aggregation \& Momentum Update**
Chỉ các node đạt ngưỡng tin cậy tối thiểu ($\mathbb{V}: T \ge T_{min}$) mới được đưa vào tổng hợp. Trọng số đóng góp được khuếch đại bằng Softmax với nhiệt độ $\beta_{soft}$:
$$w_{i,j}^{(t)} = \frac{\exp(\beta_{soft} \cdot T_{i,j}^{(t)})}{\sum_{k \in \mathbb{V}} \exp(\beta_{soft} \cdot T_{i,k}^{(t)})}$$
Vector tổng hợp an toàn:
$$S_{agg}^{(t)} = \sum_{j \in \mathbb{V}} w_{i,j}^{(t)} \cdot S_j$$
Tích hợp Momentum để tăng tốc độ hội tụ và làm phẳng quỹ đạo (ổn định hóa biến động do DP):
$$V_{agg}^{(t)} = \beta_m \cdot V_{agg}^{(t-1)} + (1 - \beta_m) \cdot S_{agg}^{(t)}$$
Cập nhật mô hình toàn cục:
$$W_i^{(t)} = W_i^{(t-1)} + V_{agg}^{(t)}$$

---

## 3. Bảng Chú thích Ký hiệu (Notation)

| Ký hiệu                  | Ý nghĩa                                                   | Ký hiệu      | Ý nghĩa                                          |
| :----------------------- | :-------------------------------------------------------- | :----------- | :----------------------------------------------- |
| $t$                      | Vòng giao tiếp (Communication round) hiện tại             | $L$          | Tổng số lượng layer trong mạng nơ-ron            |
| $W_{trained}$            | Trọng số sau khi huấn luyện cục bộ (Local SGD)            | $d_l$        | Số lượng tham số (kích thước) của layer $l$      |
| $\Delta W^{(t)}$         | Vector cập nhật (Model Delta) nguyên bản                  | $D_{total}$  | Tổng số tham số của toàn bộ mô hình ($\sum d_l$) |
| $H_l$                    | Hàng đợi lưu lịch sử chuẩn L2 của layer $l$               | $S_j$        | Gói tin chứa vector cập nhật nhận từ node $j$    |
| $Clip_l$                 | Ngưỡng cắt tỉa động cho layer $l$                         | $\lambda(t)$ | Hàm đếm số vòng hoặc epochs đã trôi qua          |
| $\sigma_{l, (t)}$        | Độ lệch chuẩn của nhiễu Gaussian cho layer $l$            | $p_{dist}$   | Điểm chất lượng dựa trên khoảng cách (RMSE)      |
| $x_l$                    | Vector nhiễu được lấy mẫu từ phân phối chuẩn              | $p_{cos}$    | Điểm chất lượng dựa trên góc (Cosine)            |
| $\tilde{W}_{i, l}$       | Vector layer $l$ sau khi cắt tỉa và tiêm nhiễu            | $\mathbb{V}$ | Tập hợp các node hàng xóm được cho là an toàn    |
| $\epsilon^{(t)}(\alpha)$ | Độ rò rỉ quyền riêng tư Rényi ở vòng $t$ với bậc $\alpha$ | $V_{agg}$    | Bộ đệm Momentum chứa quán tính cập nhật          |

---

## 4. Cấu hình Siêu tham số (Hyperparameters Tuning)

Để pipeline đạt **hiệu suất cao nhất** (cân bằng giữa độ chính xác của mô hình, khả năng chống nhiễu loạn Byzantine và bảo toàn Differential Privacy), các tham số dưới đây được khuyến nghị tối ưu dựa trên các thiết lập tiêu chuẩn của mạng Học sâu Phân tán:

### 4.1. Thông số Kỹ thuật Mạng (Network \& Momentum)

- **$\beta_m$ (Momentum Factor): `0.9`**
  - _Lý do:_ Giá trị kinh điển cho SGD with Momentum. Giúp bộ đệm $V_{agg}$ giữ lại 90% quán tính của quá trình tối ưu hóa trước đó, "san phẳng" các đỉnh nhiễu (variance) do DP gây ra và chống lại các vector độc hại lẻ tẻ.
- **$k$ (Kích thước hàng đợi Clipping - Queue Size): `5` đến `10`**
  - _Lý do:_ Đủ ngắn để bắt kịp xu hướng suy giảm tự nhiên của gradient khi mô hình hội tụ, đủ dài để không bị lệch bởi một vòng cập nhật nhiễu.

### 4.2. Thông số Bảo vệ Quyền riêng tư (Privacy Parameters)

- **$\rho_{min}$ (Initial Privacy Budget Allocation): `0.1`**
- **$\rho_{max}$ (Max Privacy Budget Allocation): `10.0`**
- **$\beta$ (Noise Decay Rate): `0.05`**
  - _Lý do:_ Tăng dần $\rho$ sẽ làm giảm phương sai nhiễu $\sigma^2$ ở các vòng sau (khi mô hình cần tinh chỉnh Fine-tuning). Ban đầu nhiễu lớn để bảo vệ các đặc trưng nhạy cảm, về sau nhiễu giảm để hội tụ mượt mà.
- **$\alpha$ (Rényi DP Orders): `[2, 4, 8, 16, 32, 64]`**
  - _Lý do:_ Tính toán trên một tập các bậc $\alpha$ để tìm ra bound chặt chẽ nhất khi chuyển đổi sang chuẩn $(\epsilon, \delta)$-DP bằng thư viện AutoDP hoặc Opacus.

### 4.3. Thông số Đánh giá \& Tin cậy (Evaluation \& Trust)

- **$\theta$ (DP Floor Multiplier): `1.2`**
  - _Lý do:_ Nhân thêm 20% dung sai so với kỳ vọng toán học của nhiễu Gaussian để tránh việc các node trung thực bị phạt lầm (False Positive) do phương sai ngẫu nhiên trong quá trình lấy mẫu nhiễu.
- **$\gamma$ (Initial Distance Scale): `3.0` đến `5.0`**
  - _Lý do:_ Ở những vòng đầu, không gian trọng số thay đổi rất dữ dội (định hướng lại bề mặt mất mát). Ngưỡng này cần đủ lớn để không kìm hãm sự học hỏi sớm.
- **$\kappa$ (Distance Threshold Decay): `0.2`**
  - _Lý do:_ Tốc độ siết chặt ranh giới an toàn. $\kappa = 0.2$ mang lại đường cong suy giảm từ tốn, song hành cùng việc suy giảm Learning Rate.
- **$\alpha_T$ (Trust EMA Factor): `0.85`**
  - _Lý do:_ Sổ danh tiếng cần tính "nhớ dai". $\alpha_T = 0.85$ nghĩa là danh tiếng hiện tại chiếm 85% trọng số, node bắt buộc phải có hành vi tốt liên tục (sustained good behavior) mới vực dậy được điểm số, tránh kiểu tấn công "vỗ béo rồi làm thịt" (tỏ ra tốt rồi bất ngờ gửi mã độc).
- **$T_{min}$ (Trust Cutoff Threshold): `0.4`**
  - _Lý do:_ Bất kỳ node nào có điểm danh tiếng rớt xuống dưới 0.4 lập tức bị loại khỏi Aggregation để cách ly hoàn toàn.
- **$\beta_{soft}$ (Softmax Temperature Inverse): `8.0`**
  - _Lý do:_ Hệ số khuếch đại sự khác biệt. Nhiệt độ thấp (giá trị $\beta_{soft}$ cao) khiến Softmax phân cực mạnh: Node có điểm tin cậy $0.9$ sẽ chiếm tỷ trọng áp đảo gần như tuyệt đối so với node ở mức $0.5$. Điều này bảo vệ tối đa độ chính xác của Global Model.
