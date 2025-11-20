# api_server.py

import os
import io
import re

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse

import easyocr
import numpy as np
from PIL import Image
import cv2

# --- GÜVENLİK AYARI ---

# Gizli anahtarı ortam değişkeninden oku (Cloud Run'da ayarlanmalı)
API_SECRET_KEY = os.environ.get("MY_API_SECRET_KEY")

# Sistem başlarken API key hiç yoksa direkt hata fırlat (deployment hatasını daha net görürüz)
if not API_SECRET_KEY:
    raise ValueError(
        "MY_API_SECRET_KEY ortam değişkeni ayarlanmadı. "
        "Lütfen Cloud Run servisinde Environment Variables kısmına ekleyin."
    )

app = FastAPI()

# --- EasyOCR için LAZY LOAD (yavaş başlangıç sorununu çözer) ---

reader = None  # Başta None, ilk istek geldiğinde yüklenecek


def get_reader():
    """
    EasyOCR modelini ilk çağrıldığında yükler, sonrasında global reader'ı kullanır.
    Böylece container açılır açılmaz ağır model yüklemesi yapılmaz.
    """
    global reader
    if reader is None:
        try:
            # Sadece TR ve EN dilleri, GPU kapalı
            _reader = easyocr.Reader(['tr', 'en'], gpu=False)
            reader = _reader
        except Exception as e:
            # Model yüklenemezse istemciye 500 dönmek için exception fırlat
            raise RuntimeError(f"EasyOCR başlatılamadı: {e}")
    return reader


# --- Yardımcı Fonksiyonlar ---

def create_color_mask(hsv_image, lower_hsv, upper_hsv):
    """Belirtilen HSV aralığı için bir maske oluşturur."""
    return cv2.inRange(hsv_image, lower_hsv, upper_hsv)


def find_highlighted_area(image_np):
    """
    Görüntüdeki yaygın işaretleme renklerini (Kırmızı, Mavi, Yeşil, Sarı) arar
    ve en büyük kontürü (çerçeveyi) döndürür.
    """
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    # Yaygın İşaretleme Renklerinin HSV Aralıkları
    color_ranges = [
        # Kırmızı (Red wraps around HUE spectrum)
        ([0, 70, 50], [10, 255, 255]),
        ([170, 70, 50], [180, 255, 255]),
        # Mavi
        ([100, 70, 50], [130, 255, 255]),
        # Yeşil
        ([35, 70, 50], [75, 255, 255]),
        # Sarı
        ([15, 70, 50], [35, 255, 255]),
    ]

    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Tüm Renk Maskelerini Birleştir
    for lower, upper in color_ranges:
        lower_hsv = np.array(lower)
        upper_hsv = np.array(upper)
        combined_mask = combined_mask | create_color_mask(hsv, lower_hsv, upper_hsv)

    # Gürültüyü Azaltma ve Kontür Bulma
    kernel = np.ones((5, 5), np.uint8)
    processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    best_cnt = max(contours, key=cv2.contourArea) if contours else None

    if best_cnt is not None and cv2.contourArea(best_cnt) > 500:  # Minimum alan kontrolü
        x, y, w, h = cv2.boundingRect(best_cnt)
        pad = 15  # Padding

        # Sınırları kontrol et
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image_np.shape[1], x + w + pad), min(image_np.shape[0], y + h + pad)

        cropped = image_np[y1:y2, x1:x2]

        return cropped, (x1, y1, x2, y2)

    return None, None


def categorize_issue(text):
    """Metin içeriğine göre sorunu sınıflandırır."""
    text_lower = text.lower()

    if any(k in text_lower for k in ["500", "fatal", "sunucu hatası", "server error"]):
        return "SERVER_SIDE_FATAL"
    if any(k in text_lower for k in ["404", "bulunamadı", "link hatalı"]):
        return "CLIENT_SIDE_LINK_ERROR"
    if any(k in text_lower for k in ["iptal", "statü", "gönderilmedi", "gelmedi"]):
        return "BUSINESS_LOGIC_ISSUE"
    if any(k in text_lower for k in ["kullanıcı", "login", "şifre", "izin"]):
        return "AUTHENTICATION_ISSUE"

    return "UNCATEGORIZED"


def extract_key_variables(text):
    """Metinden temel değişkenleri (REQ ID, HTTP Code, Email) çeker."""
    variables = {}

    # REQ- ID çekme (REQ-466 gibi)
    req_match = re.search(r"(REQ[-\s]?\d+)", text, re.IGNORECASE)
    if req_match:
        variables["REQ_ID"] = req_match.group(1).replace(" ", "").replace("-", "")

    # HTTP Hata Kodu çekme (HTTP 500 gibi)
    http_match = re.search(r"(HTTP[\s]?\d{3}|\d{3}\s(Error|Hata))", text, re.IGNORECASE)
    if http_match:
        variables["HTTP_CODE"] = http_match.group(1)

    # Email adresi çekme
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    if email_match:
        variables["EMAIL"] = email_match.group(0)

    return variables


# --- Sağlık Kontrolü Endpoint'i (Cloud Run için faydalı) ---

@app.get("/")
def health_check():
    return {"status": "ok"}


# --- ANA ENDPOINT ---

@app.post("/analiz-et")
async def analiz_et(
    file: UploadFile = File(...),
    x_api_secret: str = Header(None),
):
    # GÜVENLİK KONTROLÜ (API Key)
    if x_api_secret != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API Secret")

    # EasyOCR modelini ilk istekte yükle
    try:
        ocr_reader = get_reader()
    except RuntimeError as e:
        # Model yüklenemezse 500 dön
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "detail": str(e)},
        )

    # 1. Resmi Oku
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # 2. İşaretli Alanı Tespit Et (Çoklu Renk)
    cropped_img_np, bbox = find_highlighted_area(image_np)

    highlight_text = ""
    highlighted_ocr_success = False

    if cropped_img_np is not None:
        # Sadece kırpılmış bölgeyi oku
        res = ocr_reader.readtext(cropped_img_np, detail=0)
        highlight_text = " ".join(res)
        highlighted_ocr_success = True

    # 3. Tüm Sayfayı Oku (Yedek veya ek bağlam için)
    raw_res = ocr_reader.readtext(image_np, detail=0)
    full_text = " ".join(raw_res)

    # 4. Agent Analizi
    analysis_context = highlight_text if highlighted_ocr_success else full_text

    issue_category = categorize_issue(analysis_context)
    key_vars = extract_key_variables(analysis_context)

    # 5. Cevabı Dön
    return {
        "status": "SUCCESS",
        "issue_category": issue_category,  # Hata sınıflandırması
        "extracted_variables": key_vars,  # Anahtar değişkenler (REQ ID, HTTP Code, Email)
        "highlighted_area": {
            "text": highlight_text
            if highlighted_ocr_success
            else "İşaretli alan tespit edilemedi.",
            "bbox_pixels": bbox if bbox else None,
            "ocr_success": highlighted_ocr_success,
        },
        "full_document_ocr": full_text,
    }
