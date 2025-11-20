# api_server.py

import os
import io
import re
import time
import json
import logging

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse

import easyocr
import numpy as np
from PIL import Image
import cv2

# ---------------------------------------------------------
# LOG AYARI
# ---------------------------------------------------------

logger = logging.getLogger("ocr-analiz-api")
logger.setLevel(logging.INFO)
# Cloud Run ortamında stdout'a yazmak yeterli; ekstra handler gerekmez.

# ---------------------------------------------------------
# GÜVENLİK AYARI
# ---------------------------------------------------------

API_SECRET_KEY = os.environ.get("MY_API_SECRET_KEY")

if not API_SECRET_KEY:
    raise ValueError(
        "MY_API_SECRET_KEY ortam değişkeni ayarlanmadı. "
        "Lütfen Cloud Run servisinde Environment Variables kısmına ekleyin."
    )

app = FastAPI()

# ---------------------------------------------------------
# EasyOCR için LAZY LOAD
# ---------------------------------------------------------

reader = None  # Başlangıçta yüklenmiyor


def get_reader():
    """
    EasyOCR modelini ilk çağrıda yükler, sonrasında global reader'ı kullanır.
    Böylece container açılır açılmaz ağır yük bindirmemiş oluruz.
    """
    global reader
    if reader is None:
        try:
            _reader = easyocr.Reader(["tr", "en"], gpu=False)
            reader = _reader
            logger.info("EasyOCR reader initialized (tr, en, gpu=False)")
        except Exception as e:
            logger.error(f"EasyOCR başlatılamadı: {e}")
            raise RuntimeError(f"EasyOCR başlatılamadı: {e}")
    return reader


# ---------------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------------

def create_color_mask(hsv_image, lower_hsv, upper_hsv):
    """Belirtilen HSV aralığı için bir maske oluşturur."""
    return cv2.inRange(hsv_image, lower_hsv, upper_hsv)


def find_highlighted_areas(image_np, min_area: int = 500, max_regions: int = 10):
    """
    Görüntüdeki tüm işaretleme renklerini (Kırmızı, Mavi, Yeşil, Sarı) arar
    ve alanı min_area'dan büyük olan konturların hepsini döndürür.

    Dönüş:
      List[ (cropped_image_np, (x1, y1, x2, y2)) ]
    """
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

    color_ranges = [
        # Kırmızı
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

    for lower, upper in color_ranges:
        lower_hsv = np.array(lower)
        upper_hsv = np.array(upper)
        combined_mask = combined_mask | create_color_mask(hsv, lower_hsv, upper_hsv)

    kernel = np.ones((5, 5), np.uint8)
    processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    if not contours:
        return regions

    # En büyükten küçüğe sırala, ilk max_regions tanesini al
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours_sorted:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        pad = 15

        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image_np.shape[1], x + w + pad), min(image_np.shape[0], y + h + pad)

        cropped = image_np[y1:y2, x1:x2]
        regions.append((cropped, (x1, y1, x2, y2)))

        if len(regions) >= max_regions:
            break

    return regions


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

    req_match = re.search(r"(REQ[-\s]?\d+)", text, re.IGNORECASE)
    if req_match:
        variables["REQ_ID"] = req_match.group(1).replace(" ", "").replace("-", "")

    http_match = re.search(r"(HTTP[\s]?\d{3}|\d{3}\s(Error|Hata))", text, re.IGNORECASE)
    if http_match:
        variables["HTTP_CODE"] = http_match.group(1)

    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    if email_match:
        variables["EMAIL"] = email_match.group(0)

    return variables


# ---------------------------------------------------------
# Sağlık Kontrolü
# ---------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------
# ANA ENDPOINT
# ---------------------------------------------------------

@app.post("/analiz-et")
async def analiz_et(
    file: UploadFile = File(...),
    x_api_secret: str = Header(None),
):
    # Başlangıç zamanını al
    t_start_total = time.perf_counter()

    # Güvenlik kontrolü
    if x_api_secret != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API Secret")

    # EasyOCR modelini hazırla
    try:
        ocr_reader = get_reader()
    except RuntimeError as e:
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "detail": str(e)},
        )

    # 1. Resmi oku
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # 2. Tüm işaretli alanları tespit et
    t0 = time.perf_counter()
    regions = find_highlighted_areas(image_np)
    t_highlight_detect = time.perf_counter() - t0

    highlighted_regions_results = []
    total_highlight_ocr_time = 0.0

    # Her bölge için ayrı OCR
    for idx, (cropped_img_np, bbox) in enumerate(regions):
        t1 = time.perf_counter()
        res = ocr_reader.readtext(cropped_img_np, detail=0)
        dt = time.perf_counter() - t1
        total_highlight_ocr_time += dt

        text = " ".join(res)
        highlighted_regions_results.append(
            {
                "index": idx,
                "text": text if text else "",
                "bbox_pixels": bbox,
                "ocr_success": True if text else False,
                "ocr_ms": int(dt * 1000),
            }
        )

    # 3. Tüm sayfa OCR
    t2 = time.perf_counter()
    raw_res = ocr_reader.readtext(image_np, detail=0)
    t_full_ocr = time.perf_counter() - t2
    full_text = " ".join(raw_res)

    # 4. Analiz için bağlam
    successful_highlights = [
        r["text"] for r in highlighted_regions_results if r["ocr_success"]
    ]
    if successful_highlights:
        analysis_context = " ".join(successful_highlights)
    else:
        analysis_context = full_text

    issue_category = categorize_issue(analysis_context)
    key_vars = extract_key_variables(analysis_context)

    # Toplam süre
    total_elapsed = time.perf_counter() - t_start_total

    # 2 CPU üzerinden kaba CPU saniyesi tahmini
    cpu_seconds_estimate = total_elapsed * 2.0

    # -----------------------------------------------------
    # Performans LOG kaydı (JSON formatında)
    # -----------------------------------------------------
    log_payload = {
        "event": "ocr_request",
        "file_name": file.filename,
        "issue_category": issue_category,
        "highlight_regions_count": len(highlighted_regions_results),
        "timings_ms": {
            "total": int(total_elapsed * 1000),
            "highlight_detect": int(t_highlight_detect * 1000),
            "highlight_ocr_total": int(total_highlight_ocr_time * 1000),
            "full_ocr": int(t_full_ocr * 1000),
        },
        "cpu_seconds_estimate": cpu_seconds_estimate,
        "highlight_regions_meta": [
            {
                "index": r["index"],
                "ocr_ms": r["ocr_ms"],
                "text_len": len(r["text"]),
            }
            for r in highlighted_regions_results
        ],
    }

    logger.info(json.dumps(log_payload, ensure_ascii=False))

    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    # Eski yapıyla geriye dönük uyumluluk: ilk alanı ayrı da döndürelim
    if highlighted_regions_results:
        first_region = highlighted_regions_results[0]
        backward_compat_highlight = {
            "text": first_region["text"] or "İşaretli alan tespit edildi.",
            "bbox_pixels": first_region["bbox_pixels"],
            "ocr_success": first_region["ocr_success"],
        }
    else:
        backward_compat_highlight = {
            "text": "İşaretli alan tespit edilemedi.",
            "bbox_pixels": None,
            "ocr_success": False,
        }

    return {
        "status": "SUCCESS",
        "issue_category": issue_category,
        "extracted_variables": key_vars,
        # Tüm alanlar (yeni)
        "highlighted_areas": highlighted_regions_results,
        # İlk alan (eski kullanım için)
        "highlighted_area": backward_compat_highlight,
        "full_document_ocr": full_text,
        "metrics": {
            "total_ms": int(total_elapsed * 1000),
            "cpu_seconds_estimate": cpu_seconds_estimate,
        },
    }
