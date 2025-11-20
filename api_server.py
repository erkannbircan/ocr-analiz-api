import os
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
import easyocr
import numpy as np
from PIL import Image
import io
import cv2
import re

# --- GÜVENLİK AYARI ---
# Gizli anahtarı ortam değişkeninden oku (Cloud Run'da ayarlanmalı)
API_SECRET_KEY = os.environ.get("MY_API_SECRET_KEY") 
# Hata Kontrolü (Sadece sistemin başlaması için gereklidir)
if not API_SECRET_KEY:
    raise ValueError("MY_API_SECRET_KEY ortam değişkeni ayarlanmadı. Lütfen Cloud Run'da tanımlayın.")

app = FastAPI()

# EasyOCR Modeli Yükleme
# Tr ve En dillerini yükle. Sadece bir kez yüklenir.
try:
    reader = easyocr.Reader(['tr', 'en'], gpu=False)
except Exception as e:
    raise RuntimeError(f"EasyOCR başlatılamadı: {e}")

# --- YENİ: ÇOKLU RENK (MULTI-COLOR) TESPİT FONKSİYONU ---

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
    
    # Yaygın İşaretleme Renklerinin HSV Aralıkları (Yüksek doygunluk ve parlaklık beklenir)
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
    
    # Tüm kontürleri bul
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # En büyük geçerli kontürü bul
    best_cnt = max(contours, key=cv2.contourArea) if contours else None

    if best_cnt is not None and cv2.contourArea(best_cnt) > 500: # Minimum alan kontrolü
        x, y, w, h = cv2.boundingRect(best_cnt)
        pad = 15 # Padding eklendi
        
        # Sınırları kontrol et
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image_np.shape[1], x + w + pad), min(image_np.shape[0], y + h + pad)
        
        cropped = image_np[y1:y2, x1:x2]
        
        return cropped, (x1, y1, x2, y2)
    
    return None, None

# --- YENİ: AJAN FONKSİYONLARI ---

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
    """Metinden temel değişkenleri (REQ ID, HTTP Code) çeker (Regex tabanlı)."""
    variables = {}
    
    # REQ- ID çekme (REQ-466 gibi)
    req_match = re.search(r'(REQ[-\s]?\d+)', text, re.IGNORECASE)
    if req_match:
        variables['REQ_ID'] = req_match.group(1).replace(' ', '').replace('-', '')
        
    # HTTP Hata Kodu çekme (HTTP 500 gibi)
    http_match = re.search(r'(HTTP[\s]?\d{3}|\d{3}\s(Error|Hata))', text, re.IGNORECASE)
    if http_match:
        variables['HTTP_CODE'] = http_match.group(1)

    # Email adresi çekme
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    if email_match:
        variables['EMAIL'] = email_match.group(0)

    return variables

# --- ANA ENDPOINT ---

@app.post("/analiz-et")
async def analiz_et(file: UploadFile = File(...), x_api_secret: str = Header(None)):
    
    # GÜVENLİK KONTROLÜ (API Key Yöntemi)
    if x_api_secret != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API Secret")
    
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
        res = reader.readtext(cropped_img_np, detail=0)
        highlight_text = " ".join(res)
        highlighted_ocr_success = True

    # 3. Tüm Sayfayı Oku (Yedek veya ek bağlam için)
    raw_res = reader.readtext(image_np, detail=0)
    full_text = " ".join(raw_res)

    # 4. Agent Analizi
    # İşaretli alan varsa öncelikli olarak o alana odaklan
    analysis_context = highlight_text if highlighted_ocr_success else full_text
    
    issue_category = categorize_issue(analysis_context)
    key_vars = extract_key_variables(analysis_context)

    # 5. Cevabı Dön
    return {
        "status": "SUCCESS",
        "issue_category": issue_category, # Yeni: Hata sınıflandırması
        "extracted_variables": key_vars, # Yeni: Anahtar değişkenler (REQ ID, HTTP Code, Email)
        "highlighted_area": {
            "text": highlight_text if highlighted_ocr_success else "İşaretli alan tespit edilemedi.",
            "bbox_pixels": bbox if bbox else None,
            "ocr_success": highlighted_ocr_success
        },
        "full_document_ocr": full_text
    }