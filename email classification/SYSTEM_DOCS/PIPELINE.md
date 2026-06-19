# Quy trình chi tiết: Input → Output

Tài liệu này mô tả chi tiết các bước biến đổi dữ liệu và luồng SVM trong repository, từ dữ liệu thô đến kết quả/triển khai.

## 1. Dữ liệu đầu vào (Raw Input)
- File nguồn: `dataset/combined_data.csv` (cột chính: text, label).
- Kiểu: CSV với `text` (string) và `label` (nhãn spam/ham, ví dụ 0/1).

## 2. Lấy mẫu và phân chia
- Tạo các file mẫu: `dataset/sampled_dataset{N}.csv` (ví dụ 1000,1500,...). Script: `create_sample_dataset.py`.
- Chia train/test: thường 80/20 hoặc cross‑validation theo notebook.

## 3. Tiền xử lý văn bản
Các bước phổ biến (thực hiện trong các notebook):
- Lowercase, loại bỏ ký tự đặc biệt và punctuation.
- Tokenize.
- Loại stopwords (tùy chọn language-specific stoplist).
- (Tùy) Stemming / Lemmatization.
- Kết quả: danh sách token cho mỗi mẫu.

Kiểu dữ liệu: danh sách chuỗi → tiếp tục sang bước vector hóa.

## 4. Vector hóa (Feature Extraction)
- Phương pháp: `TfidfVectorizer` (sparse matrix).
- Tham số thường dùng trong repo: `max_features` = 1000,1500,... (các file kết quả trong `result/` cho thấy kích thước vocab khác nhau).
- Output: ma trận TF‑IDF `X` có shape `(n_samples, n_features)`; dtype float32/float64; lưu trữ dưới dạng sparse.

Ghi chú: giữ `TfidfVectorizer` đã fit trên tập huấn luyện để transform tập kiểm thử.

## 5. Chuẩn hóa nhãn
- Nhãn `y` có thể được mã hoá là `{0,1}` hoặc `{ -1, +1 }`.
- Một số thuật toán SVM (ví dụ Pegasos) hoạt động với `{ -1, +1 }` nên cần chuyển đổi nếu cần.

## 6. Huấn luyện SVM
Hai hướng chính trong repo:

- Pegasos (stochastic gradient descent trên hinge loss):
  - Mục tiêu tối ưu: min_{w} (λ/2)||w||^2 + (1/m)∑ max(0, 1 - y_i (w·x_i + b)).
  - Quy tắc cập nhật (mini-batch hoặc single-sample):
    - η_t = 1 / (λ * t)
    - Nếu y_i (w·x_i) < 1: w ← (1 - η_t * λ) w + η_t * y_i * x_i
    - Ngược lại: w ← (1 - η_t * λ) w
  - Tham số: `λ` (regularization), số epoch, batch size, khởi tạo `w`.
  - Ưu điểm: hiệu quả cho dữ liệu lớn, memory thấp.

- SVM tuyến tính (hard/soft margin) bằng giải pháp giản đơn hoặc thư viện: kiểm tra các notebook `chay-thu-thong-ke`.

Output huấn luyện: trọng số `w` (vector length = `n_features`), bias `b` (nếu có).

## 7. Đánh giá
- Dự đoán: `score = w·x + b`; `label_pred = sign(score)`.
- Metrics thường lưu: accuracy, precision, recall, f1, AUC, và thời gian huấn luyện.
- Kết quả lưu: các file CSV trong `result/` (ví dụ `svm_pegasos_1000.csv`, `base_svm_tfidf_1000.csv`).

## 8. Lưu model & artifacts
- Model có thể lưu bằng `joblib.dump()` hoặc `pickle` (định dạng: `w`, `b`, `vectorizer` để tái sử dụng).
- Lưu `TfidfVectorizer` cùng với `w` để đảm bảo transform input mới giống khi huấn luyện.

## 9. Triển khai
- File: `deployment/app.py` sử dụng model đã lưu để load `vectorizer` và `w,b` → nhận text mới → tiền xử lý giống training → TF‑IDF transform → `score` → trả về nhãn và score.

## 10. Luồng cuối cùng (tóm tắt từng bước)
1. Người dùng/Script cung cấp CSV raw.
2. (Optional) Lấy mẫu thành `sampled_datasetN`.
3. Tiền xử lý text → token.
4. Fit `TfidfVectorizer` trên train → transform train/test.
5. Encode nhãn phù hợp (±1 nếu cần).
6. Huấn luyện SVM (Pegasos / solver tuyến tính).
7. Dự đoán trên tập kiểm thử → tính metrics.
8. Lưu kết quả CSV vào `result/` và lưu model (pickle/joblib).
9. `deployment/app.py` load model để dự đoán realtime/batch.

## 11. Thông số thường chỉnh
- `max_features` (vocab size): 1000..9500 (thử nghiệm trong repo).
- `λ` (lambda) hoặc `C`: điều chỉnh bias-variance.
- `epochs`, `batch_size` cho Pegasos.

## 12. Kiểm tra nhanh (ví dụ lệnh)
```bash
# Fit và chạy notebook: mở `chay-thu-thong-ke/svm-pegasos.ipynb` hoặc chạy script tương ứng
```

---
File này mô tả luồng chung; nếu bạn muốn, tôi có thể: 1) trích xuất đoạn code chính từ notebook `chay-thu-thong-ke/svm-pegasos.ipynb` để chỉ ra nơi tiền xử lý/huấn luyện, hoặc 2) thêm ví dụ lệnh `python` để tái tạo pipeline.
