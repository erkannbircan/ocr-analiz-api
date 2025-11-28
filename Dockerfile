# Python 3.9 Slim (Hafif ve Kararlı)
FROM python:3.9-slim

# OpenCV ve EasyOCR için Gerekli Linux Kütüphaneleri
# headless kullansak bile bu kütüphanelerin olması garanti sağlar.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Önce requirements kopyalanır (Cache avantajı için)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY api_server.py .

# Varsayılan port
ENV PORT=8080

# Uvicorn ile başlat (Tek worker, thread çakışması olmaması için)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]