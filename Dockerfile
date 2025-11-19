# Dockerfile
FROM python:3.9-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod dosyasını kopyala
COPY api_server.py .

# API'yi 8080 portunda çalıştır
ENV PORT 8080
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
