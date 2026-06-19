FROM python:3.11-slim
WORKDIR /app

# Install system build dependencies required for some packages (e.g. scipy)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential \
	   gcc \
	   gfortran \
	   pkg-config \
	   libopenblas-dev \
	   liblapack-dev \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY linear_svm.pkl .
COPY vectorizer.pkl .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
