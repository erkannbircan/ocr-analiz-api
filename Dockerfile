# Dockerfile (güncel)
FROM python:3.9-slim

# EasyOCR / OpenCV için gerekli sistem paketleri
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyası
COPY api_server.py .

# Cloud Run port
ENV PORT=8080

# FastAPI + Uvicorn
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
