# Dockerfile
FROM python:3.9-slim

# Sistem bağımlılıklarını yükle (EasyOCR / OpenCV için gerekli)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /app

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod dosyasını kopyala
COPY api_server.py .

# Cloud Run PORT değişkeni
ENV PORT=8080

# Uvicorn ile FastAPI'yi 0.0.0.0:8080'de çalıştır
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
