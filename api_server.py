import os
import io
import time
import json
import logging
import re
import warnings

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse

import easyocr
import numpy as np
from PIL import Image
import cv2

# Uyarıları kapat
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. AYARLAR
# ---------------------------------------------------------

logger = logging.getLogger("ocr-analiz-api")
logger.setLevel(logging.INFO)

API_SECRET_KEY = os.environ.get("MY_API_SECRET_KEY")
MAX_FILE_SIZE_MB = 10

if not API_SECRET_KEY:
    API_SECRET_KEY = "TEST_SECRET" 

app = FastAPI()

# ---------------------------------------------------------
# 2. OCR MOTORU (DENGELİ MOD)
# ---------------------------------------------------------

reader = None

def get_reader():
    global reader
    if reader is None:
        try:
            # quantize=False yaptık çünkü küçük yazılarda hata yapıyordu.
            # Doğruluk için bunu kapattık.
            _reader = easyocr.Reader(["tr", "en"], gpu=False, verbose=False, quantize=False)
            reader = _reader
            logger.info("EasyOCR Hazır.")
        except Exception as e:
            logger.error(f"Hata: {e}")
            raise RuntimeError(str(e))
    return reader

# ---------------------------------------------------------
# 3. YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------

def correct_tourism_terms(text: str) -> str:
    if not text: return ""
    corrections = {
        "S6L": "SGL", "5GL": "SGL", "SGI": "SGL", "SG1": "SGL",
        "D8L": "DBL", "0BL": "DBL", "OBL": "DBL", "DB1": "DBL",
        "TR1": "TRP", "TRIP": "TRP", "TRPL": "TRP",
        "QUAD": "QUAD", "QUD": "QUAD", "0UAD": "QUAD",
        "PR0MO": "PROMO", "STDA": "STD", "STND": "STD",
        "STE": "SUITE", "SU1TE": "SUITE", "SUI": "SUITE",
        "H8": "HB", "H3": "HB", "88": "BB", "B8": "BB", "8B": "BB",
        "F8": "FB", "FB+": "FBPLUS",
        "A1": "AI", "ALL": "AI", "UA1": "UAI",
        "R0": "RO", "SC": "SC", "R00M": "ROOM",
        "TL": "TRY", "TRL": "TRY", "YTL": "TRY",
        "US0": "USD", "U5D": "USD", 
        "EUR0": "EUR", "EU": "EUR",
        "|": "1", "I": "1", "l": "1", "O": "0", "o": "0",
        "CNL": "CANCEL", "CX": "CANCEL", "IPTAL": "CANCEL",
        "REF": "REFUND", "IADE": "REFUND", "REQ": "REQUEST"
    }
    words = text.split()
    corrected_words = []
    for w in words:
        clean_w = w.upper().replace(".", "").replace(",", "").replace(":", "")
        corrected_words.append(corrections.get(clean_w, w))
    return " ".join(corrected_words)

def extract_key_variables(text: str) -> dict:
    variables = {}
    def safe_extract(pattern):
        try:
            m = re.search(pattern, text, re.IGNORECASE)
            return m.group(1).strip() if (m and m.lastindex) else (m.group(0).strip() if m else None)
        except: return None

    variables["REQ_ID"] = safe_extract(r"(REQ[-\s]?\d+)")
    # E-mail regex iyileştirildi
    variables["EMAIL"] = safe_extract(r"([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)")
    
    domain = {"otel": {}, "fiyat": {}}
    domain["fiyat"]["tutar"] = safe_extract(r"(\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\b)")
    domain["otel"]["pnr"] = safe_extract(r"pnr[:\s]*([A-Za-z0-9\-]+)")
    domain["fiyat"]["para_birimi"] = safe_extract(r"\b(try|tl|usd|eur|gbp)\b")
    
    clean_domain = {k: v for k, v in domain.items() if any(v.values())}
    variables["domain"] = clean_domain
    return variables

def categorize_issue(text: str) -> str:
    t = text.lower()
    if "500" in t or "fatal" in t: return "SERVER_ERROR"
    if "404" in t: return "LINK_ERROR"
    if "fatura" in t: return "INVOICE"
    if "iptal" in t: return "CANCEL"
    return "UNCATEGORIZED"

def find_highlighted_areas(image_np):
    """
    KUTU TESPİTİNDE HASSASİYET ARTTIRILDI.
    Mavi çerçeve gibi ince çizgileri yakalamak için 'dilate' (kalınlaştırma) eklendi.
    """
    # Renk tespiti ORİJİNAL boyutta yapılır (Detay kaybı olmasın diye)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "red": ([0, 60, 40], [10, 255, 255]),
        # Mavi aralığı genişletildi
        "blue": ([80, 40, 40], [140, 255, 255]),
        "green": ([35, 40, 30], [85, 255, 255]),
        "yellow": ([15, 40, 40], [50, 255, 255]),
    }

    regions = []
    
    # Çizgileri kalınlaştırmak için kernel
    dilate_kernel = np.ones((5,5), np.uint8) 

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
        
        # KRİTİK DÜZELTME: İnce mavi çizgileri yakalamak için maskeyi şişiriyoruz (Dilate)
        # Bu işlem kopuk çizgileri birleştirir.
        mask = cv2.dilate(mask, dilate_kernel, iterations=2)
        
        # Gürültü temizleme
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500: continue # Çok küçük lekeleri at
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Padding ekle (yazı sınıra yapışmasın)
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(image_np.shape[1], x + w + pad)
            y2 = min(image_np.shape[0], y + h + pad)
            
            regions.append({
                "color": color,
                "bbox": [x1, y1, x2, y2],
                "crop": image_np[y1:y2, x1:x2]
            })
            
    # Yukarıdan aşağıya sırala
    regions.sort(key=lambda r: r["bbox"][1])
    return regions

# ---------------------------------------------------------
# 4. ENDPOINT
# ---------------------------------------------------------

@app.post("/analiz-et")
async def analiz_et(
    file: UploadFile = File(...),
    x_api_secret: str = Header(None),
):
    t_start = time.perf_counter()

    if x_api_secret != API_SECRET_KEY:
        raise HTTPException(401, "Unauthorized")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    try:
        ocr_reader = get_reader()
    except Exception as e:
        return JSONResponse(500, {"status": "ERROR", "detail": str(e)})

    # 1. ALANLARI BUL (Orijinal Boyutta + Dilate ile)
    regions = find_highlighted_areas(image_np)
    
    highlight_results = []
    
    # 2. KUTUCUKLARI OKU (YÜKSEK KALİTE MODU)
    # Kutular küçük olduğu için burada kaliteyi artırabiliriz, yavaşlatmaz.
    for r in regions:
        crop = r["crop"]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        
        # 2x Büyütme (Okunabilirlik için şart)
        crop_upscaled = cv2.resize(crop_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Standart Mod (beamWidth varsayılan/yüksek) - Eksiksiz okuma için
        res = ocr_reader.readtext(crop_upscaled, detail=0, paragraph=True)
        text = " ".join(res)
        
        highlight_results.append({
            "color": r["color"],
            "text": correct_tourism_terms(text),
            "bbox": r["bbox"]
        })

    # 3. TÜM SAYFA OKUMA (HIZLI MOD - SADECE BAĞLAM İÇİN)
    # Tüm sayfayı yine küçültüyoruz ama bu sefer 1280px ideal (960px çok bozabilir).
    gray_full = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    h, w = gray_full.shape[:2]
    TARGET_H = 1280.0
    
    if h > TARGET_H:
        scale = TARGET_H / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray_full = cv2.resize(gray_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Tüm sayfa için beamWidth=1 kullanabiliriz çünkü sadece anahtar kelime arıyoruz.
    full_res = ocr_reader.readtext(
        gray_full, 
        detail=0, 
        beamWidth=1,      # Hız için düşük
        paragraph=True,   # Satır birleştirme
        batch_size=4
    )
    full_text = " ".join(full_res)
    
    # Veri Çıkar
    full_text_corrected = correct_tourism_terms(full_text)
    cat = categorize_issue(full_text_corrected)
    vars = extract_key_variables(full_text_corrected)
    
    total_ms = int((time.perf_counter() - t_start) * 1000)

    return {
        "status": "SUCCESS",
        "issue_category": cat,
        "extracted_variables": vars,
        "full_text": full_text_corrected, # Hızlı okunan metin
        "highlighted_areas": highlight_results, # Detaylı, eksiksiz okunan alanlar
        "metrics": {
            "total_ms": total_ms,
            "detected_regions": len(highlight_results)
        }
    }