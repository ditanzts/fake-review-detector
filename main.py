from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Load file .env agar USE_APIFY dan APIFY_TOKEN terbaca
from dotenv import load_dotenv
load_dotenv()

# Import modul-modul
from modules.data_source.local_data import get_reviews        # ambil data ulasan
from modules.prediction import predict_reviews                # deteksi fake/genuine
from modules.recommendation import get_recommendations        # cari kafe serupa

# ============================================================
# Inisialisasi aplikasi FastAPI
# ============================================================
app = FastAPI(
    title="Deteksi Ulasan Kafe Jember",
    description="API untuk deteksi fake review dan rekomendasi kafe",
    version="1.0.0"
)

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
# ENDPOINT 2 — Analisis ulasan kafe
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
        "cafe_name": result["cafe_name"],
        "total":     result["total"],
        "genuine":   result["genuine_count"],
        "fake":      result["fake_count"],
        "reviews":   result["genuine_reviews"]   # hanya ulasan genuine yang ditampilkan
    }


# ============================================================
# ENDPOINT 3 — Rekomendasi kafe serupa
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
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )