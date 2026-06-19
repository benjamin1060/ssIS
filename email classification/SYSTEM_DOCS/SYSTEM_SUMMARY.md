# Tổng hợp hệ thống: AI SVM Email Spam

Tệp này tổng hợp và mô tả cấu trúc, thành phần, cách thiết lập và cách chạy hệ thống phân loại email spam sử dụng SVM trong repo này. Mục tiêu: hỗ trợ việc viết tài liệu chính thức cho đồ án.

---

## 1. Tổng quan
- Mục tiêu: Phân loại email (tiếng Anh) thành spam / non-spam sử dụng SVM (cả cài đặt Pegasos/Hard-margin và triển khai SVM tuyến tính). Hệ thống có giao diện triển khai bằng Streamlit.
- Triển khai demo: https://ai-svm-deploy.onrender.com (được ghi trong README).

## 2. Cấu trúc project (tóm tắt các thư mục/tệp quan trọng)
- `create_sample_dataset.py`: Script tạo các tập mẫu theo kích thước khác nhau từ `dataset/combined_data.csv`.
- `dataset/`: Chứa dữ liệu gốc và các tập mẫu `sampled_dataset{N}.csv` (N = 1000,1500,...)
  - `dataset/combined_data.csv` — dữ liệu gốc (nhãn trong cột `label`).
- `chay-thu-thong-ke/` và `chay-model-final/`: Các notebook thực nghiệm, training và thu thập kết quả.
- `result/`: Các file CSV kết quả từ các thử nghiệm (kết quả SVM, Pegasos, hard-margin, v.v.).
- `xay-dung-model/`: Notebook phát triển mô hình (ví dụ `svm.ipynb`, `svm-ai.ipynb`).
- `deployment/`:
  - `app.py` — Streamlit app để demo phân loại (load `linear_svm.pkl` và `vectorizer.pkl`).
  - `Dockerfile` — tệp để đóng gói deployment.
- `requirements.txt` — thư viện cần thiết để chạy app và notebook.
- `setup_env.ps1` — script PowerShell hỗ trợ thiết lập môi trường (nếu có).

## 3. Dữ liệu
- File nguồn: `dataset/combined_data.csv`.
- Script tạo tập con: `create_sample_dataset.py`.
  - Hoạt động: đọc `combined_data.csv` từ URL GitHub, tính tỷ lệ nhãn, và sinh các file đồ mẫu `dataset/sampled_dataset{N}.csv` với các kích thước khác nhau.
  - Tham số: mảng kích thước `a = range(1000, 82000, 5000)` (theo mã nguồn).
- Các tập mẫu trong `dataset/` dùng cho huấn luyện, đánh giá thời gian/chất lượng.

## 4. Tiền xử lý & Vectorization
- Ứng dụng sử dụng `nltk` để xử lý:
  - Tải stopwords, tokenizer (ghi chú: code dùng `punkt_tab` — kiểm tra nếu đúng tên resource).
  - Loại ký tự đặc biệt, loại URL, tokenize, loại stopwords, stem bằng `SnowballStemmer`.
- Vectorizer: sử dụng `TfidfVectorizer` (được lưu thành `vectorizer.pkl` trong quá trình huấn luyện).

## 5. Mô hình
- Có nhiều cách tiếp cận trong repo:
  - Mô hình SVM custom: cài đặt class `SVM` trong `deployment/app.py` (cài đặt dạng Pegasos SGD với regularization `lambda_param`, epoch, batch_size, tol).
  - Notebook trong `chay-model-final/` và `xay-dung-model/` có các triển khai SVM (Pegasos, hard-margin, SVC từ scikit-learn).
- Lưu model dùng cho deployment: `linear_svm.pkl` (pickle) và `vectorizer.pkl`.
- Ghi chú kỹ thuật về `SVM.fit` (theo mã trong `app.py`):
  - Chuyển nhãn 0->-1 và 1->1 cho training nội bộ
  - Sử dụng learning rate eta = 1 / (lambda * t)
  - Áp cập nhật trọng số theo Pegasos với projection để đảm bảo ||w|| <= 1/sqrt(lambda)
  - Early stopping nếu thay đổi hàm mục tiêu < tol

## 6. Thí nghiệm và kết quả
- Các file kết quả nằm trong `result/` (nhiều CSV theo kích thước dữ liệu và phương pháp):
  - `svm_pegasos_*.csv`, `hardmargin_svm_tfidf_*.csv`, `base_svm_tfidf_*.csv`, v.v.
  - `combined_time_svm_after_pegasos.csv` chứa thống kê thời gian/chất lượng hợp nhất.
- Notebook `chay-thu-thong-ke/` chứa các phân tích thống kê, visualization, và đo thời gian huấn luyện.

## 7. Triển khai (Deployment)
- Streamlit app: `deployment/app.py`
  - Load model & vectorizer từ `linear_svm.pkl` và `vectorizer.pkl`.
  - Tiền xử lý văn bản đầu vào và dự đoán. Hiển thị kết quả trên giao diện web.
  - Phải đảm bảo các resource NLTK đã được tải trước khi chạy app (stopwords, punkt)
- Dockerfile: dùng để build image triển khai (kiểm tra nội dung `deployment/Dockerfile` để biết chi tiết build/runtime).

## 8. Cài đặt & Chạy
- Tạo virtualenv và cài dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

- Chạy Streamlit app (từ thư mục gốc hoặc `deployment/` nếu cấu hình file relative):

```bash
streamlit run deployment/app.py
```

- Nếu dùng Docker (từ `deployment/`):

```bash
docker build -t ai-svm-app .
docker run -p 8501:8501 ai-svm-app
```

## 9. Tệp cần chú ý (vị trí và vai trò)
- [README.md](README.md#L1) — Tổng quan ngắn gọn và link demo.
- [create_sample_dataset.py](create_sample_dataset.py#L1) — Script tạo tập mẫu.
- [deployment/app.py](deployment/app.py#L1) — Mã chạy Streamlit, SVM custom, preprocessing.
- [requirements.txt](requirements.txt#L1) — Gói cần cài.
- `dataset/combined_data.csv` — Dữ liệu gốc (không nhúng nội dung ở đây).
- `result/` — Kết quả thí nghiệm: nhiều file CSV.
- Notebooks: `xay-dung-model/`, `chay-model-final/`, `chay-thu-thong-ke/` — các bước phát triển + thử nghiệm.

## 10. Kiểm tra và ghi chú kỹ thuật
- Xác thực resource NLTK: mã trong `app.py` gọi `nltk.download('punkt_tab')` — kiểm tra xem tên resource có phải `punkt` không (khả năng là `punkt`), vì `punkt_tab` có thể gây lỗi tải.
- Đảm bảo file `linear_svm.pkl` và `vectorizer.pkl` tồn tại trong thư mục chạy app.
- Ở `create_sample_dataset.py`, script đọc file `combined_data.csv` từ GitHub raw URL; khi chạy offline, cần thay bằng đường dẫn cục bộ `dataset/combined_data.csv`.

## 11. Gợi ý cho tài liệu chính thức
- Bắt đầu bằng **Overview** + **Quickstart** (các lệnh cài và chạy app).
- Thêm **Architecture diagram**: luồng dữ liệu (data → preprocessing → vectorizer → model → deployment)
- Thêm **Bảng các file** với mô tả (có thể dùng phần 2 ở trên làm cơ sở).
- Ghi rõ **phiên bản môi trường** (Python, major package versions) — dùng `requirements.txt` làm nguồn.
- Ghi **quy trình tái tạo kết quả**: dùng `create_sample_dataset.py` để tạo tập mẫu, chạy các notebook trong `xay-dung-model/` và `chay-model-final/`, thu kết quả vào `result/`.

---

Nếu bạn muốn, tôi sẽ:
- Cập nhật và mở rộng `README.md` bằng nội dung tóm tắt từ file này.
- Tạo sơ đồ kiến trúc (Mermaid) và chèn vào tài liệu.
- Kiểm tra và sửa `app.py` để thay `punkt_tab` thành `punkt` nếu cần.

Vui lòng cho biết bạn muốn tôi thực hiện bước tiếp theo nào.