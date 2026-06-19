# Demo nhận diện email tiếng Anh spam sử dụng SVM

Demo sử dụng thư viện Streamlit để xây giao diện web sử dụng model SVM để nhận diện

## Tính năng

- Giao diện web để nhập nội dung email
- Phát hiện email spam dựa trên mô hình sVM
- Tiền xử lý văn bản bao gồm tokenize,stemming và loại bỏ từ dừng
- Docker containerization

## Yêu cầu

- Tải sẵn docker
- Có các file nhị phân model (`linear_svm.pkl`) và vectorizer  (`vectorizer.pkl`)

## Bắt đầu

### Building the Docker Image

1. Đưa vào cùng đường dẫn:
   - `app.py` (Streamlit )
   - `Dockerfile`
   - `requirements.txt`
   - `linear_svm.pkl` (SVM model)
   - `hashing_vectorizer.pkl` (vectorizer)

2. Build  Docker image:
   ```bash
   docker build -t spam-detector .
   ```

3. Chạy  container:
   ```bash
   docker run -p 8501:8501 spam-detector
   ```

4. Truy cập đường dẫn web:
   ```
   http://localhost:8501
   ```

## Vấn đề gặp phải
Dương tính giả (false positive) --> đánh giá sai nhiều email bình thường là spam



https://www.theguardian.com/world/2025/apr/12/us-demands-control-from-ukraine-of-key-pipeline-carrying-russian-gas
### Not spam
#### Ex1
The US has demanded control of a crucial pipeline in Ukraine used to send Russian gas to Europe, according to reports, in a move described as a colonial shakedown.

US and Ukrainian officials met on Friday to discuss White House proposals for a minerals deal. Donald Trump wants Kyiv to hand over its natural resources as “payback” in return for weapons delivered by the previous Biden administration.

Talks have become increasingly acrimonious, Reuters said. The latest US draft is more “maximalist” than the original version from February, which proposed giving Washington $500bn worth of rare metals, as well as oil and gas.

Citing a source close to the talks, the news agency said the most recent document includes a demand that the US government’s International Development Finance Corporation take control of the natural gas pipeline.
#### Ex2
The report suggests that universities will need to look beyond their traditional international student recruitment markets, given that demand in India is slowing and higher education is improving in quality in east Asia. They may also need to consider offering more cost-effective options, the report notes.