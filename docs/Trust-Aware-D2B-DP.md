# Pipeline Xử lý Bảo mật Chủ động trong DFL (Trust-Aware D2B-DP)

Thuật toán Trust-Aware D2B-DP mô tả vòng đời của một Node $i$ trong vòng giao tiếp $t$ thuộc mạng lưới Học liên kết phi tập trung (Decentralized Federated Learning - DFL). Cấu trúc Pipeline được thực thi tuần tự qua 4 giai đoạn chính để đảm bảo tính riêng tư của dữ liệu và chống lại các cuộc tấn công độc hại (Byzantine/Inference Attacks).

## Tổng quan 4 Giai đoạn

1.  **Giai đoạn 1 (Local Train):** Huấn luyện cục bộ & Cắt tỉa Gradient thích ứng.
2.  **Giai đoạn 2 (Outbound):** Tra cứu Tín nhiệm $\rightarrow$ Cấp ngân sách $\rightarrow$ Bơm nhiễu $\rightarrow$ Gửi đi.
3.  **Giai đoạn 3 (Inbound):** Nhận Gradient $\rightarrow$ Đánh giá Dị thường (Z-Score) $\rightarrow$ Cập nhật Tín nhiệm.
4.  **Giai đoạn 4 (Aggregation):** Tính ngưỡng động $\rightarrow$ Lọc Node $\rightarrow$ Tổng hợp Mean.

---

## Giai đoạn 1: Local Training (Huấn luyện cục bộ)

### Bước 1: Huấn luyện & Tính Gradient

Tại bước này, Node $i$ sử dụng tập dữ liệu cục bộ của mình để huấn luyện và tính toán ngưỡng cắt tỉa động dựa trên lịch sử các vòng trước.

- **Thực thi:** Node $i$ sử dụng dữ liệu $D_i$ để huấn luyện và thu được vector gradient thô $\Delta w_e$ cho layer $e$.
- **Tính ngưỡng động (Adaptive Clipping Threshold):** Hệ thống sử dụng một hàng đợi để lấy trung bình chuẩn L2 của các gradient trong các vòng trước đó, giúp xác định ngưỡng cắt tỉa phù hợp cho vòng hiện tại:
  $$Clip_e = \frac{1}{k} \sum_{m=1}^k H[m]$$

**Chú thích ký hiệu:**

- $D_i$: Tập dữ liệu cục bộ của Node $i$.
- $\Delta w_e$: Gradient thô sinh ra từ quá trình huấn luyện tại layer $e$.
- $H$: Hàng đợi FIFO (First-In-First-Out) lưu trữ giá trị chuẩn L2 $\|\Delta w_e\|_2$ của các vòng gần nhất.
- $k$: Độ dài cửa sổ trượt của hàng đợi (ví dụ: $k=3$).
- _Lưu ý xử lý vòng 1:_ Nếu hàng đợi $H$ trống (vòng huấn luyện đầu tiên), ta khởi tạo $Clip_e = \|\Delta w_e\|_2$.

### Bước 2: Cắt tỉa Gradient cục bộ (Clipping)

Để kiểm soát độ nhạy (Sensitivity) của mô hình đối với các điểm dữ liệu cụ thể, gradient thô cần được cắt tỉa sao cho độ lớn (chuẩn L2) của nó không vượt quá ngưỡng $Clip_e$.

- **Công thức chuẩn hóa:**
  $$\Delta w'_e = \frac{\Delta w_e}{\max\left(1, \frac{\|\Delta w_e\|_2}{Clip_e}\right)}$$

**Chú thích ký hiệu:**

- $\Delta w'_e$: Gradient sau khi đã được cắt tỉa. Nếu $\|\Delta w_e\|_2 \le Clip_e$, gradient được giữ nguyên. Nếu vượt quá, nó sẽ được thu nhỏ lại bằng đúng độ dài $Clip_e$.
- $\|\cdot\|_2$: Chuẩn L2 (độ dài của vector trong không gian Euclid).

---

## Giai đoạn 2: Outbound Processing (Xử lý thông tin gửi đi)

### Bước 3: Tra cứu Tín nhiệm & Cấp Ngân sách

Trước khi gửi dữ liệu cho một node hàng xóm $j$, Node $i$ sẽ đánh giá độ tin cậy của $j$ và cấp một lượng "ngân sách riêng tư" tương ứng.

- **Tính ngân sách cơ sở (tăng dần theo thời gian):**
  $$\rho_{base}^{(t)} = \min\left( (1 + \beta t)\rho_{min}, \rho_{max} \right)$$
- **Cấp ngân sách cá nhân hóa cho Node $j$:**
  $$\rho_{i \rightarrow j}^{(t)} = \rho_{base}^{(t)} \times T_{i,j}^{(t-1)}$$

**Chú thích ký hiệu:**

- $T_{i,j}^{(t-1)}$: Điểm tín nhiệm của node hàng xóm $j$ do Node $i$ đánh giá, tính từ cuối vòng $t-1$. (_Khởi tạo Cold-Start tại vòng 1: $T_{i,j}^{(0)} = 0.5$ - mức trung lập\_).
- $\rho_{base}^{(t)}$: Ngân sách cơ sở tăng dần theo thời gian vòng lặp $t$.
- $\beta$: Hệ số gia tăng ngân sách theo thời gian.
- $\rho_{min}, \rho_{max}$: Giới hạn ngân sách tối thiểu và tối đa.
- $\rho_{i \rightarrow j}^{(t)}$: Ngân sách quyền riêng tư thực tế mà Node $i$ cấp cho Node $j$ tại vòng $t$.

### Bước 4: Xác định Cường độ Nhiễu Cá nhân hóa & Kế toán RDP

Quá trình này biến đổi ngân sách thành lượng nhiễu thực tế và kiểm soát mức độ rò rỉ thông tin theo chuẩn Rényi Differential Privacy (RDP).

- **Tính phương sai nhiễu ($\sigma^2$):** Dựa vào ngân sách đã cấp, Node $i$ quyết định lượng nhiễu sẽ bơm vào gradient:
  $$\sigma^2_{i \rightarrow j} = \frac{2 \cdot Clip_e^2}{|D_i|^2 \cdot \rho_{i \rightarrow j}^{(t)}}$$
  _Tính chất bảo mật cốt lõi:_ Có một mối quan hệ nghịch biến ở đây. Nếu Node $j$ **đáng tin** ($T$ cao $\rightarrow$ $\rho$ cao), lượng nhiễu $\sigma^2$ sẽ **nhỏ**, giúp mô hình học tốt hơn. Ngược lại, nếu Node $j$ **đáng ngờ**, lượng nhiễu sẽ **cực lớn**, che giấu hoàn toàn dữ liệu của Node $i$.

- **Kế toán Quyền riêng tư theo Rényi DP (RDP):** Tính toán mức độ hao hụt quyền riêng tư tại vòng $t$ (với bậc $\alpha > 1$):
  $$\epsilon_\alpha^{(t)} = \frac{\alpha \cdot Clip_e^2}{2\sigma_{i \rightarrow j}^2}$$
- **Tổng hao hụt tích lũy:**
  $$RDP_{total}(\alpha) = \sum_{t=1}^{t_{current}} \epsilon_\alpha^{(t)}$$

**Chú thích ký hiệu:**

- $|D_i|$: Kích thước tập dữ liệu (số lượng mẫu) của Node $i$.
- $\alpha$: Bậc của Rényi DP (thường được chọn tối ưu để quy đổi sang chuẩn DP thông thường).
- _Cơ chế tự bảo vệ:_ Node quy đổi $RDP_{total}$ sang chuẩn $(\epsilon, \delta)$-DP. Nếu tổng rò rỉ $\epsilon > \epsilon_{max}$ (ngưỡng tối đa), Node $i$ sẽ **từ chối gửi** cho Node $j$ để chống lại Inference Attack (Tấn công suy luận).

### Bước 5: Bơm Bounded Noise & Truyền tải

Gradient được thêm nhiễu ngẫu nhiên, nhưng nhiễu này được "gọt đuôi" để tránh làm hỏng hoàn toàn quá trình hội tụ.

- **Sinh nhiễu và hàm Bounding:** Sinh nhiễu $x \sim \mathcal{N}(0, \sigma^2_{i \rightarrow j})$ từ phân phối chuẩn và gọt đuôi qua hàm $B(x)$:
  $$B(x) = \max(-b, \min(x, b))$$
  Với biên giới hạn $b$ được tính bằng:
  $$b = \frac{e^\epsilon - \eta}{e^\epsilon + \eta}$$
- **Gói tin truyền tải:** Gradient cuối cùng được gửi đi:
  $$\tilde{w}_{i \rightarrow j} = \Delta w'_e + B(x)$$

**Chú thích ký hiệu:**

- $b$: Biên giới hạn nhiễu tuyệt đối (đảm bảo nhiễu không tiến tới vô cực).
- $\tilde{w}_{i \rightarrow j}$: Gói tin Gradient đã bảo mật hoàn chỉnh mà Node $i$ gửi đến Node $j$.

---

## Giai đoạn 3: Inbound Evaluation (Đánh giá dữ liệu nhận về)

_Cơ sở lý thuyết:_ Giai đoạn này sử dụng Z-Score để chống lại Tấn công khuếch đại (Scaling/Amplification Attack), nơi kẻ xấu nhân gradient của chúng lên nhiều lần (ví dụ $\times 100$) để chèn ép các node trung thực.

### Bước 6: Đánh giá Dị thường (Z-Score)

- **Kiểm tra Độ lớn (Magnitude Check):** Đánh giá mức độ lệch của gradient nhận được $S_j$ từ node $j$:
  $$Z(S_j) = \left| \frac{1}{d} \sum_{v=1}^d S_j[v] \right|$$
- **Chuyển đổi thành Điểm An toàn:**
  $$P_{safe} = \max\left(0, 1 - \frac{Z(S_j)}{Z_{threshold}^{(t)}}\right)$$

**Chú thích ký hiệu:**

- $S_j$: Vector gradient Node $i$ nhận được từ Node $j$.
- $d$: Số chiều (kích thước) của vector gradient.
- $P_{safe}$: Điểm đánh giá mức độ an toàn về mặt "độ lớn". Nếu $Z(S_j)$ vượt quá ngưỡng $Z_{threshold}^{(t)}$, điểm an toàn $P_{safe}$ lập tức bị kéo về $0$.

### Bước 6.1: Tính Ngưỡng Z (Robust Temporal Windowing)

Ngưỡng $Z_{threshold}$ được tính toán động dựa trên lịch sử mạng lưới để chống lại sự thao túng của các giá trị ngoại lai (Outliers).

- **Tính Trung vị tuyệt đối (MAD) có Sàn độ nhạy:** Gộp $W$ vòng lặp gần nhất vào bộ đệm $\mathcal{B}_Z^{(t)}$:
  $$MAD_{Z\_safe}^{(t)} = \max\left( \text{Median}\left( \left| \mathcal{B}_Z^{(t)} - \text{Med}_Z^{(t)} \right| \right), \sigma_{floor\_Z} \right)$$
- **Cập nhật Ngưỡng động:**
  $$Z_{threshold}^{(t)} = \text{Med}_Z^{(t)} + \gamma_Z \cdot MAD_{Z\_safe}^{(t)}$$

**Chú thích ký hiệu:**

- $\mathcal{B}_Z^{(t)}$: Bộ đệm lưu trữ các giá trị Z-Score trong $W$ vòng gần nhất.
- $\text{Med}_Z^{(t)}$: Giá trị trung vị của bộ đệm $\mathcal{B}_Z^{(t)}$.
- $\sigma_{floor\_Z}$: Sàn độ nhạy (dung sai tối thiểu, vd $10^{-4}$), đảm bảo ngưỡng không bị siết chặt về $0$ khi các node gửi giá trị quá giống nhau.
- $\gamma_Z$: Hệ số điều chỉnh ngưỡng Z.

### Bước 7: Đánh giá Đồng thuận & Tính điểm Vòng $t$

Ngoài kiểm tra độ lớn, hệ thống kiểm tra "hướng" học tập để xem node $j$ có đóng góp hữu ích hay không.

- **Kiểm tra Hướng học tập (Cosine Similarity):** So sánh $S_j$ với gradient tự tính của bản thân $\Delta w'_e$, chuẩn hóa về khoảng $[0, 1]$:
  $$P_{sim} = \frac{1}{2} \left( \frac{\Delta w'_e \cdot S_j}{\|\Delta w'_e\|_2 \times \|S_j\|_2} + 1 \right)$$
- **Tính điểm Hành vi vòng $t$ (Harmonic Mean):**
  $$R_{i,j}^{(t)} = \frac{2 \cdot P_{safe} \cdot P_{sim}}{P_{safe} + P_{sim} + \epsilon} \quad (\text{với } \epsilon \to 0)$$
  _Ý nghĩa:_ Dùng trung bình điều hòa đảm bảo rằng Node $j$ chỉ được điểm cao khi nó vừa **Sạch** ($P_{safe}$ cao) VÀ vừa **Hữu ích** ($P_{sim}$ cao). Một trong hai bằng $0$ sẽ kéo $R_{i,j}$ về $0$.

### Bước 8: Cập nhật Ma trận Tín nhiệm ($T_{i,j}$)

Sử dụng đường trung bình động hàm mũ (EMA) để cập nhật tín nhiệm lâu dài:
$$T_{i,j}^{(t)} = \lambda \cdot T_{i,j}^{(t-1)} + (1 - \lambda) \cdot R_{i,j}^{(t)}$$

**Chú thích ký hiệu:**

- $T_{i,j}^{(t)}$: Điểm tín nhiệm mới của Node $j$, sẽ được dùng làm cơ sở để cấp ngân sách cho vòng $t+1$.
- $\lambda \in [0, 1]$: Hệ số ghi nhớ quá khứ (ví dụ $\lambda = 0.8$ nghĩa là hệ thống tin tưởng 80% vào lịch sử và 20% vào hành vi hiện tại).

---

## Giai đoạn 4: Aggregation & End Round (Tổng hợp & Kết thúc Vòng)

### Bước 9: Xác định Ngưỡng loại bỏ $\tau_{drop}$

Sử dụng thuật toán Robust Temporal Windowing để tạo ra một ngưỡng loại bỏ động, không bị ảnh hưởng bởi không gian mẫu nhỏ hay Outliers.

- **Lấy mẫu bộ đệm:** Gộp $W$ vòng lặp gần nhất vào bộ đệm tín nhiệm $\mathcal{B}^{(t)}$.
- **Thống kê mạnh (có Sensitivity Floor):**
  $$\text{Med}^{(t)} = \text{Median}(\mathcal{B}^{(t)})$$
  $$MAD_{safe}^{(t)} = \max\left( \text{Median}\left( \left| \mathcal{B}^{(t)} - \text{Med}^{(t)} \right| \right), \sigma_{floor} \right)$$
- **Tính ngưỡng từ chối động:**
  $$\tau_{drop}^{(t)} = \text{Med}^{(t)} - \alpha \cdot MAD_{safe}^{(t)}$$

**Chú thích ký hiệu:**

- $\mathcal{B}^{(t)}$: Bộ nhớ đệm lưu trữ điểm tín nhiệm trong các vòng gần nhất.
- $\alpha$: Hệ số kiểm soát độ nhạy (ví dụ $\alpha=2$).
- $\sigma_{floor}$: Dung sai an toàn (vd $10^{-3}$) giúp duy trì tính linh hoạt khi mạng hội tụ.

### Bước 10: Reject & Penalty Node Độc hại

Đối chiếu điểm tín nhiệm $T_{i,j}^{(t)}$ với ngưỡng $\tau_{drop}^{(t)}$. Nếu $T_{i,j}^{(t)} < \tau_{drop}^{(t)}$, hệ thống sẽ thực thi hình phạt:

- **Reject (Loại bỏ):** Xóa vĩnh viễn gradient $S_j$ của node vi phạm khỏi danh sách tổng hợp $\mathbb{V}$.
- **Penalty (Phạt suy giảm):** Áp dụng hình phạt nhân suy giảm ngay lập tức để chống lại các "Sleeper Node" (node độc hại ngủ đông tích điểm):
  $$T_{i,j}^{(t)} = T_{i,j}^{(t)} \times \gamma_{penalty}$$

**Chú thích ký hiệu:**

- $\mathbb{V}$: Tập hợp các node hàng xóm "Sạch" đã vượt qua được bài kiểm tra tín nhiệm.
- $\gamma_{penalty} \in (0, 1)$: Hệ số hình phạt (ví dụ $\gamma_{penalty} = 0.5$ sẽ lập tức tước đi một nửa số điểm tín nhiệm tích lũy của node vi phạm).

### Bước 11: Aggregation & Cập nhật Model

Bước cuối cùng là tổng hợp các gradient "sạch" và cập nhật trọng số cho mô hình ở vòng tiếp theo.

- **Tổng hợp Mean:**
  $$S_{agg} = \frac{1}{|\mathbb{V}|} \sum_{j \in \mathbb{V}} S_j$$
- **Cập nhật trọng số:**
  $$W_i^{(t)} = W_i^{(t-1)} - \eta \cdot S_{agg}$$

**Chú thích ký hiệu:**

- $S_{agg}$: Vector gradient tổng hợp an toàn từ các node đáng tin cậy.
- $|\mathbb{V}|$: Số lượng node hàng xóm an toàn trong tập $\mathbb{V}$.
- $W_i^{(t)}, W_i^{(t-1)}$: Trọng số của mô hình cục bộ tại vòng hiện tại và vòng trước đó.
- $\eta$: Tốc độ học (Learning Rate).
