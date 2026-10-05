# BÁO CÁO BÀI TẬP THỰC HÀNH: POSE ESTIMATION & MODEL EVALUATION

- **Họ và tên học viên:** Trịnh Xuân Huy
- **Mã số sinh viên/Học viên:** 2A202602995
- **Link notebook đã chạy:** [Google Drive Notebook](https://drive.google.com/file/d/1z8OdhszvYpO1Or0d9qtQd_-RYuskx_WU/view?usp=sharing)


---

## 1. Tổng quan các nội dung đã hoàn thành trong bài Lab

Trong bài thực hành này, toàn bộ các hàm cốt lõi, câu hỏi lý thuyết và thí nghiệm thực nghiệm đã được cài đặt và vượt qua các bộ kiểm thử tự động (100% Passed):

1. **Cài đặt 6 hàm cốt lõi:**
   - `box_iou`: Tính chỉ số IoU giữa các bounding box dự đoán và nhãn Ground Truth.
   - `nms` & `batched_nms`: Thuật toán Non-Maximum Suppression đơn lẻ và phân loại theo lớp để loại bỏ các box trùng lặp.
   - `mask_iou`: Tính IoU trên ma trận nhị phân (mask segmentation).
   - `oks` (Object Keypoint Similarity): Tính độ tương đồng keypoint chuẩn COCO, chuẩn hóa theo kích thước đối tượng (s² = area) và hệ số độ lệch σ_i của từng khớp.
   - `joint_angle`: Tính góc giữa các khớp xương cơ thể từ 3 điểm mốc sử dụng tích vô hướng và hàm arccos.

2. **Các bài toán ứng dụng và phân tích:**
   - **Chuyển đổi Polygon sang Mask & YOLO Seg format:** Xây dựng pipeline tự động gán nhãn (`autolabel/bus.txt`).
   - **Phân tích Latency:** Đo và đánh giá thời gian suy luận trên CPU/GPU giữa các mô hình.
   - **Nhận diện ngã (Fall Detection):** Phân tích luật góc nghiêng thân người, các hạn chế khi người nằm/ngồi bệt, và đề xuất kiến trúc chuỗi thời gian video (temporal tracking) thực tế.
   - **Trả lời đầy đủ 12 câu hỏi lý thuyết (Q1 – Q12):** Bao gồm phân tích sai số pixel người lớn vs người nhỏ, ý nghĩa của σ (sigma), thiên lệch dữ liệu và bài toán gán nhãn.

3. **Cài đặt nâng cao (Bonus):**
   - Hoàn thành hàm tính `average_precision` (1D).
   - Triển khai thí nghiệm đánh giá trên tập validation lật gương (4C).

---

## 2. Báo cáo Bài tập về nhà: Thí nghiệm 4C và Thiết kế tập Validation

### 2.1. Bảng kết quả thực nghiệm

Mô hình YOLO-pose (12 keypoint) được đánh giá trên hai tập kiểm thử: tập validation gốc (100% hổ quay sang phải) và tập validation lật gương (mô phỏng 100% hổ quay sang trái với nhãn giải phẫu chuẩn):

| Mô hình | Pose mAP50-95 (Val gốc) | Pose mAP50-95 (Val lật gương) | Biến thiên |
| :--- | :---: | :---: | :---: |
| **`flip_idx` giải phẫu** | **0.457** | **0.439** | Giữ vững (-3.9%) |
| **`flip_idx` đồng nhất** | **0.417** | **0.298** | Tụt dốc thảm hại (-28.5%) |

### 2.2. Metric nào đã che giấu lỗi?
- **Pose mAP trên tập val gốc** là metric đã che đậy hoàn toàn lỗi logic nghiêm trọng của mô hình `flip_idx đồng nhất`. Do 100% ảnh val gốc đều là hổ quay sang phải, mô hình không hề bị thử thách trên trường hợp quay trái. Điểm số mAP vẫn đạt mức 0.417, tạo cảm giác sai lầm rằng mô hình hoạt động ổn định.
- Ngoài ra, việc dùng chung một hệ số σ cào bằng (σ = 1/12) cho mọi keypoint làm giảm độ nhạy phạt sai số ở các điểm mốc cực kỳ nhạy cảm (mắt, mũi) so với các điểm mốc rộng (gốc đuôi). Chỉ khi đưa vào tập kiểm thử **val lật gương**, lỗi hoán đổi chân trái/phải theo giải phẫu mới bị bộc lộ rõ rệt khi mAP tụt sâu từ 0.417 xuống 0.298.

### 2.3. Đề xuất quy trình thiết kế tập Validation trong thực tế
Để tránh bẫy thiên lệch dữ liệu (data bias) và đảm bảo mô hình sẵn sàng chạy thực tế:
1. **Phân tầng cân bằng thuộc tính (Stratified / Attribute-balanced Split):** Không chia train/val ngẫu nhiên theo ID ảnh đơn thuần mà phải cân bằng các thuộc tính then chốt: hướng quay (50% trái, 50% phải), các biến thể góc chụp camera (góc cao, góc ngang) và các dạng địa hình nền (cỏ cao, đất trống).
2. **Xây dựng bộ kiểm thử áp lực (Stress-test Sets):** Bắt buộc phải có các tập test đặc thù như tập lật gương, tập che khuất một phần (occlusion test) để phát hiện ngay các lỗi cấu hình augmentation hoặc hoán đổi chi.
3. **Ước lượng bộ `kpt_oks_sigmas` riêng cho bài toán custom:** Tiến hành quy trình gán nhãn lặp lại (redundant annotation) với 3–5 người gán độc lập trên tập mẫu, đo độ lệch chuẩn chuẩn hóa d_i / √(area) để tính σ_i chính xác cho từng bộ phận thay vì gán giá trị đều.