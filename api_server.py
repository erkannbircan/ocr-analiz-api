from fastapi import FastAPI, File, UploadFile
import easyocr
import numpy as np
from PIL import Image
import io
import re
import cv2

app = FastAPI()

# --- MODELLERİ VE FONKSİYONLARI TANIMLA (Eski kodun aynısı) ---
reader = easyocr.Reader(['tr', 'en'], gpu=False)

SORUN_SINYALLERI = {
    "Teknik Hata": ["error", "hata", "exception", "404", "500", "bug", "failed"],
    "Finansal Uyuşmazlık": ["fark", "eksik", "bakiye", "borç", "iade", "tutar"],
    "Operasyonel Sorun": ["iptal", "gelmedi", "onaylanmadı", "reddedildi"],
}

def kirmizi_isaret_bul(image_np):
    # OpenCV işlemleri (Eski koddan alındı)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            if area > max_area:
                max_area = area
                best_cnt = cnt
                
    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        pad = 10
        # Koordinatları taşmayacak şekilde ayarla
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(image_cv.shape[1], x+w+pad), min(image_cv.shape[0], y+h+pad)
        cropped = image_np[y1:y2, x1:x2]
        return cropped, True
    return None, False

def sorun_tahmin_et(full_text):
    text_lower = full_text.lower()
    tespitler = []
    for kategori, kelimeler in SORUN_SINYALLERI.items():
        for kelime in kelimeler:
            if kelime in text_lower:
                tespitler.append(kategori)
                break
    return ", ".join(tespitler) if tespitler else "Normal"

# --- n8n'İN BAĞLANACAĞI KAPI (ENDPOINT) ---
@app.post("/analiz-et")
async def analiz_et(file: UploadFile = File(...)):
    # 1. Gelen dosyayı oku
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # 2. İşaretli Alan Kontrolü
    crop_img, is_highlighted = kirmizi_isaret_bul(image_np)
    highlight_text = ""
    if is_highlighted:
        res = reader.readtext(crop_img, detail=0)
        highlight_text = " ".join(res)

    # 3. Tam Metin Okuma
    raw_res = reader.readtext(image_np, detail=0)
    full_text = " ".join(raw_res)

    # 4. Tahmin
    prediction = sorun_tahmin_et(full_text)
    
    # 5. Veri Ayıklama (Basit Regex)
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', full_text)
    talep_no = re.findall(r'(REQ-\d+|ER-\d+)', full_text)

    # n8n'e geri dönecek JSON cevabı
    return {
        "kullanici_isaretli_alan": highlight_text if highlight_text else None,
        "sistem_tahmini": prediction,
        "talep_no": talep_no[0] if talep_no else None,
        "epostalar": emails,
        "full_metin": full_text
    }