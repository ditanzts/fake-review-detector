# ============================================================
# main.py — Pintu utama FastAPI
# Semua "jalan" (endpoint) yang bisa diakses web ada di sini
# ============================================================

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Load file .env agar USE_APIFY dan APIFY_TOKEN terbaca
from dotenv import load_dotenv
load_dotenv()

# Import modul-modul buatan kita (akan dibuat di langkah berikutnya)
from modules.data_source.local_data import get_reviews        # ambil data ulasan
from modules.prediction import predict_reviews                # deteksi fake/genuine
from modules.recommendation import get_recommendations, DB_KAFE  # cari kafe serupa

# ============================================================
# Inisialisasi aplikasi FastAPI
# ============================================================
app = FastAPI(
    title="Deteksi Ulasan Kafe Jember",
    description="API untuk deteksi fake review dan rekomendasi kafe",
    version="1.0.0"
)

# Sambungkan folder static/ agar HTML, CSS, JS bisa diakses
# Artinya: file di folder static/ bisa dibuka lewat browser
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================
# Model data — bentuk data yang dikirim dari web ke backend
# ============================================================

class AnalyzeRequest(BaseModel):
    """Data yang dikirim saat user input URL kafe"""
    url: str                  # URL Google Maps kafe

class RecommendRequest(BaseModel):
    """Data yang dikirim saat user minta rekomendasi"""
    cafe_url: str             # URL kafe yang sedang dilihat


# ============================================================
# ENDPOINT 1 — Halaman utama
# Saat user buka http://localhost:8000 → langsung ke index.html
# ============================================================
@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/results.html")
def results():
    return FileResponse("static/results.html")

@app.get("/recommendations.html")
def recommendations():
    return FileResponse("static/recommendations.html")


# ============================================================
# ENDPOINT BARU — Cek mode (Apify atau Lokal)
# Dipakai index.html untuk memutuskan tampilan mana yang muncul
# ============================================================
@app.get("/mode")
def get_mode():
    use_apify = os.getenv("USE_APIFY", "false").lower() == "true"
    return {"mode": "apify" if use_apify else "local"}


# ============================================================
# ENDPOINT BARU — Daftar semua kafe di database
# Dipakai popup pilih kafe di index.html (mode lokal)
# ============================================================
@app.get("/cafes")
def get_cafes():
    if DB_KAFE is None or DB_KAFE.empty:
        raise HTTPException(status_code=500, detail="Database kafe tidak tersedia")

    cafes = []
    for _, row in DB_KAFE.iterrows():
        cafes.append({
            "name":   row["name"],
            "url":    row["url"],
            "labels": row["labels"] if isinstance(row["labels"], list) else [],
        })

    # Urutkan alfabetis
    cafes.sort(key=lambda x: x["name"].lower())
    return {"cafes": cafes}


# ============================================================
# ENDPOINT 2 — Analisis ulasan kafe
#
# Cara kerja:
# 1. Web kirim URL kafe ke sini
# 2. Kita ambil data ulasan (dari lokal / nanti Apify)
# 3. Kita prediksi mana yang genuine, mana yang fake
# 4. Kita kembalikan hasilnya ke web
#
# Contoh request dari web (JavaScript):
#   fetch("/analyze", {
#     method: "POST",
#     body: JSON.stringify({ url: "https://maps.google.com/..." })
#   })
# ============================================================
@app.post("/analyze")
def analyze(request: AnalyzeRequest):

    # Validasi URL tidak boleh kosong
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL tidak boleh kosong")

    # Langkah 1: Ambil data ulasan mentah
    raw_reviews = get_reviews(request.url)

    if not raw_reviews:
        raise HTTPException(status_code=404, detail="Ulasan tidak ditemukan untuk kafe ini")

    # Langkah 2: Prediksi genuine vs fake
    result = predict_reviews(raw_reviews)

    # Langkah 3: Kembalikan hasil ke web
    return {
        "cafe_name":    result["cafe_name"],
        "total":        result["total"],
        "genuine":      result["genuine_count"],
        "fake":         result["fake_count"],
        "reviews":      result["genuine_reviews"],
        "fake_reviews": result["fake_reviews"],
    }


# ============================================================
# ENDPOINT 3 — Rekomendasi kafe serupa
#
# Cara kerja:
# 1. Web kirim URL kafe yang sedang dilihat
# 2. Kita cari tag kafe tersebut di dbKafe
# 3. Kita cari kafe lain dengan tag serupa + paling dekat
# 4. Kembalikan daftar rekomendasinya
# ============================================================
@app.post("/recommend")
def recommend(request: RecommendRequest):

    if not request.cafe_url.strip():
        raise HTTPException(status_code=400, detail="URL kafe tidak boleh kosong")

    recommendations = get_recommendations(request.cafe_url)

    if not recommendations:
        raise HTTPException(status_code=404, detail="Tidak ada rekomendasi ditemukan")

    return {
        "recommendations": recommendations
    }


# ============================================================
# Jalankan server
# Ketik di terminal: python main.py
# Lalu buka browser: http://localhost:8000
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )