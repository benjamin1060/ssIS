# Chương 3 — Mô tả hệ thống và phương pháp thực nghiệm

Tài liệu này tóm tắt toàn bộ các thành phần của hệ thống phân loại email (email spam) trong repository, bao gồm dữ liệu, tiền xử lý, trình bày mô hình SVM (cài tay), pipeline huấn luyện, kết quả thí nghiệm và triển khai.

## 1. Tổng quan cấu trúc dự án
- `create_sample_dataset.py`: script tạo các tập mẫu theo kích thước khác nhau từ `combined_data.csv`.
- `dataset/`: chứa `combined_data.csv` (dữ liệu gốc lớn) và các file mẫu `sampled_dataset{N}.csv`.
- `chay-thu-thong-ke/`, `chay-model-final/`, `xay-dung-model/`: notebooks huấn luyện và thí nghiệm (vectorizer + SVM, so sánh thời gian, hard-margin, Pegasos).
- `result/`: các CSV kết quả thí nghiệm (ví dụ `svm_pegasos_1000.csv`, `hardmargin_svm_tfidf_1000.csv`, `base_svm_tfidf_1000.csv`, `combined_time_svm_after_pegasos.csv`, ...).
- `deployment/app.py`: ứng dụng Streamlit demo, load `vectorizer.pkl` và `linear_svm.pkl` để dự đoán.
- `requirements.txt`, `requirements-installed.txt`, `setup_env.ps1`, `.venv` (local) — môi trường Python.

## 2. Dữ liệu (Data)

- File gốc: `dataset/combined_data.csv` — mỗi hàng là một email. (Kích thước lớn, không đồng bộ trong repo vì quá lớn.)
- Schema (trực tiếp quan sát từ các file mẫu như `dataset/sampled_dataset1000.csv`):
  - `label`: nhãn nhị phân (0 hoặc 1). Theo code và notebooks, 1 là spam (ví dụ mẫu chứa nhiều spam), 0 là non-spam.
  - `text`: nội dung email nguyên thủy (chuỗi nhiều dòng, chứa HTML, URL, ký tự đặc biệt).

- Tạo tập mẫu: `create_sample_dataset.py` (đường dẫn: `create_sample_dataset.py`) — hành vi:
  - Tải `combined_data.csv` từ remote GitHub URL (script dùng pandas.read_csv trên URL). (Xem script: [create_sample_dataset.py](create_sample_dataset.py#L1-L20)).
  - Tính tỷ lệ của mỗi lớp (`proportions = df[target_column].value_counts(normalize=True)`).
  - Với danh sách kích thước `a = range(1000, 82000, 5000)`, script sample theo tỷ lệ lớp và lưu `dataset/sampled_dataset{N}.csv` (seed `random_state=42`).

## 3. Tiền xử lý văn bản (Text preprocessing)

Các bước tiền xử lý được áp dụng nhất quán trong notebooks và file deploy (`deployment/app.py`):

- Tài nguyên NLTK: `stopwords` và tokenizer (`punkt_tab` hoặc `punkt`) được tải.
- Hàm chính (triển khai tương đương trong `deployment/app.py`):
  1. `remove_special_characters(text)`: loại bỏ ký tự không phải chữ cái và khoảng trắng (regex `[^a-zA-Z\\s]`).
  2. `remove_url(text)`: loại bỏ URL bằng regex `http\\S+`.
  3. `word_tokenize(text)`: tách từ bằng NLTK.
  4. `remove_stop_words(tokens)`: loại stopwords tiếng Anh (`nltk.corpus.stopwords.words('english')`).
  5. `stem_text(text)`: stemming với `SnowballStemmer('english')` trên các token, sau đó nối thành chuỗi.

- Kết quả tiền xử lý: một chuỗi token đã stem, dùng để vector hóa bằng TF-IDF.

Ví dụ mã tiền xử lý: [deployment/app.py](deployment/app.py#L40-L120).

## 4. Biến đổi đặc trưng (Feature extraction)

- Sử dụng `sklearn.feature_extraction.text.TfidfVectorizer` (mặc định, không thấy tham số tuỳ chỉnh quan trọng trong notebooks). Thao tác:
  - `vectorizer = TfidfVectorizer()`
  - `X = vectorizer.fit_transform(df['text'])` (trên tập huấn luyện hoặc toàn bộ dataset tuỳ notebook)
  - Kết quả là ma trận TF-IDF dạng sparse (`csr_matrix`).

- Trong một số thí nghiệm (ví dụ hard-margin SVM), dữ liệu TF-IDF được chuyển sang dense (`toarray()`) nếu thuật toán yêu cầu.

## 5. Mô hình SVM (cài tay) — thiết kế và tham số

- Có hai hướng cài đặt chính trong project:
  1. SVM cài tay theo dạng Pegasos / SGD (class `SVM` trong `deployment/app.py` và tương tự trong notebooks).
  2. Sử dụng `scikit-learn` SVM (để so sánh/đặt baseline) trong một vài notebook.

- Chi tiết class `SVM` (theo `deployment/app.py`):
  - Tham số khởi tạo: `lambda_param=1e-4`, `epoch=1000`, `batch_size=256`, `tol=1e-4`, `random_state=42`.
  - Nội dung `fit(X,y)`:
    - Nếu `X` có `toarray` sẽ dùng `csr_matrix(X)` để bảo toàn sparse.
    - Chuyển nhãn: nếu nhãn là {0,1} thì ánh xạ 0 -> -1, 1 -> 1 cho tối ưu hinge loss.
    - Khởi tạo `w` (zeros) và `b` = 0.
    - Lặp `epoch` lần: xáo trộn chỉ số, chia batch theo `batch_size`, cập nhật theo công thức học Pegasos:
      - learning rate: `eta = 1.0 / (lambda_param * t)` với t tăng dần theo update step.
      - Tính margins `y_batch * (X_batch.dot(w) + b)`, chọn các mẫu vi phạm (`mask = margins < 1`).
      - Cập nhật: weight decay `w *= (1 - eta * lambda_param)`; nếu tồn tại vi phạm, cộng gradient của hinge loss qua `X_violate` và `y_violate` (khi dạng dense chuyển bằng `toarray()` nếu cần); cập nhật `b` bằng tổng `y_violate` nhân hệ số.
      - Áp dụng projection để giới hạn norm: factor = min(1, (1/sqrt(lambda)) / norm_w) và nhân `w *= factor`.
    - Tính objective = 0.5 * lambda * ||w||^2 + mean(hinge_losses). Dừng sớm nếu thay đổi objective < `tol`.
  - `predict(X)`: tính `decision = X.dot(w) + b` và trả nhãn `1` nếu decision >= 0, ngược lại `0`.

## 6. Huấn luyện & đánh giá

- Chuẩn bị dữ liệu:
  - Áp dụng tiền xử lý (stem, stopwords, loại ký tự, loại URL) trước khi TF-IDF.
  - Vectorizer fit trên tập dữ liệu (toàn bộ hoặc chỉ tập huấn luyện tuỳ notebook).
  - Chia train/test: `train_test_split(X, y, test_size=0.2, random_state=42)`.

- Huấn luyện: chạy `SVM.fit(X_train, y_train)` (với X sparse hoặc dense tuỳ thuật toán).

- Đánh giá: các notebook tính các metric tiêu chuẩn: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix` và `classification_report`.

- Thí nghiệm quy mô dữ liệu: project chạy thí nghiệm trên nhiều kích thước mẫu (ví dụ 1000, 1500, 2000, ... đến hàng chục nghìn) bằng script `create_sample_dataset.py` và notebooks. Kết quả lưu trong `result/` (các file CSV theo kích thước và phương pháp).

## 7. Kết quả & artifact

- Mô hình và vectorizer được lưu dưới dạng pickle: `linear_svm.pkl`, `vectorizer.pkl` (được load trong `deployment/app.py`).
- Các file kết quả thí nghiệm nằm trong `result/` (tên file mô tả phương pháp và kích thước dataset). Ví dụ: `svm_pegasos_1000.csv`, `base_svm_tfidf_1000.csv`, `hardmargin_svm_tfidf_1000.csv`.

## 8. Triển khai (Deployment)

- Ứng dụng demo: `deployment/app.py` (Streamlit). Quy trình:
  1. Tải NLTK resources (`stopwords`, `punkt_tab`).
  2. Định nghĩa hàm tiền xử lý như trên.
  3. Load `vectorizer.pkl` và `linear_svm.pkl` bằng `pickle` (hàm `load_model()`).
  4. Khi người dùng nhập text, tiền xử lý → `vectorizer.transform([preprocessed_text])` → `model.predict(features)` → hiển thị spam/non-spam.

- Chạy demo (ví dụ):
```
# trong môi trường ảo đã cài requirements
cd deployment
streamlit run app.py
```

## 9. Môi trường & phụ thuộc

- Các phụ thuộc quan trọng (xem `requirements.txt`): `pandas`, `numpy`, `nltk`, `scikit-learn`, `scipy`, `streamlit` (và các thư viện plotting nếu cần trong notebooks).
- Script `setup_env.ps1` hỗ trợ khởi tạo môi trường trên Windows; repo chứa `requirements-installed.txt` cho tham khảo.

## 10. Vị trí mã nguồn quan trọng (để trích dẫn trong báo cáo)
- `create_sample_dataset.py` — tạo tập mẫu.
- `dataset/combined_data.csv` — dữ liệu gốc (mô tả schema: `label,text`).
- `dataset/sampled_dataset{N}.csv` — mẫu dùng cho thí nghiệm.
- `deployment/app.py` — pipeline tiền xử lý + load model + predict.
- `xay-dung-model/svm-ai.ipynb`, `chay-thu-thong-ke/svm-pegasos.ipynb`, `chay-model-final/svm-pegasos-run.ipynb` — notebooks huấn luyện và thí nghiệm (vectorizer fit, huấn luyện SVM, so sánh phương pháp).
- `result/` — chứa các file CSV kết quả thí nghiệm.

## 11. Gợi ý cấu trúc nội dung Chương 3 (dựa trên code)
- 3.1 Dữ liệu: nguồn, schema, cách tạo mẫu (tham khảo `create_sample_dataset.py`).
- 3.2 Tiền xử lý: mô tả chi tiết từng bước và lý do (loại ký tự, URL, stopwords, stemming).
- 3.3 Biến đổi đặc trưng: TF-IDF (tham số mặc định, sparse representation).
- 3.4 Mô hình: mô tả chi tiết thuật toán Pegasos-SGD (công thức cập nhật, projection, learning rate), các tham số (`lambda_param`, `epoch`, `batch_size`, `tol`). Trích dẫn trực tiếp từ `deployment/app.py` hoặc notebooks.
- 3.5 Thiết kế thí nghiệm: cách chia tập, các kích thước mẫu, đánh giá theo metric (accuracy, precision, recall, f1), lưu kết quả.
- 3.6 Triển khai: mô tả `deployment/app.py`, cách load model/vectorizer, user flow.

## 12. Các lệnh thuận tiện để tái tạo (Quick commands)
```
# Tạo environment (Windows PowerShell)
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# Tạo các tập mẫu
python create_sample_dataset.py

# Chạy demo Streamlit
cd deployment
streamlit run app.py
```

## 13. Ghi chú và hạn chế
- Dữ liệu thô chứa nhiều HTML, URL, ký tự không chữ, và spam thương mại/chiêu trò; tiền xử lý hiện tại chỉ dựa trên loại ký tự/stopwords/stemming, không dùng lemmatization hay xử lý HTML nâng cao.
- Vectorizer sử dụng tham số mặc định; có thể cải thiện bằng tuning (ngram_range, max_features, min_df, max_df).
- Mô hình SVM cài tay yêu cầu hiệu chỉnh tham số và kiểm tra ổn định với dữ liệu sparse lớn; một số biến thể (chẳng hạn kernel-based SVM) không được triển khai.

---
Tài liệu này được sinh tự động từ phân tích mã nguồn trong repository; bạn có thể dùng trực tiếp nội dung phần trên để phát triển Chương 3 — Phương pháp nghiên cứu và xây dựng mô hình. Nếu muốn, tôi sẽ:
- Thêm trích dẫn code (đoạn code ngắn) vào từng mục (ví dụ công thức cập nhật trong Pegasos). 
- Sinh sơ đồ pipeline (Mermaid) hoặc bảng phân bố nhãn từ `combined_data.csv` (cần tải file hoặc chạy script cục bộ).
