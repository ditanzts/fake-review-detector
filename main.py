import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# load file .env 
from dotenv import load_dotenv
load_dotenv()

# lmport modul-modul 
from modules.data_source.local_data import get_reviews        
from modules.prediction import predict_reviews                
from modules.recommendation import get_recommendations, DB_KAFE  


# inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Deteksi Ulasan Kafe Jember",
    description="API untuk deteksi fake review dan rekomendasi kafe",
    version="1.0.0"
)

# sambungkan folder static/
app.mount("/static", StaticFiles(directory="static"), name="static")


# model data
class AnalyzeRequest(BaseModel):
    """Data yang dikirim saat user input URL kafe"""
    url: str                  # URL Google Maps kafe

class RecommendRequest(BaseModel):
    """Data yang dikirim saat user minta rekomendasi"""
    cafe_url: str             # URL kafe yang sedang dilihat


# halaman utama
@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/results.html")
def results():
    return FileResponse("static/results.html")

@app.get("/recommendations.html")
def recommendations():
    return FileResponse("static/recommendations.html")


# cek mode apify/lokal
@app.get("/mode")
def get_mode():
    use_apify = os.getenv("USE_APIFY", "false").lower() == "true"
    return {"mode": "apify" if use_apify else "local"}


# popup pilih kafe (mode lokal)
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

    # urutkan alfabetis
    cafes.sort(key=lambda x: x["name"].lower())
    return {"cafes": cafes}


# analisis ulasan kafe
@app.post("/analyze")
def analyze(request: AnalyzeRequest):

    # validasi URL tidak boleh kosong
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL tidak boleh kosong")

    # ambil data ulasan mentah
    raw_reviews = get_reviews(request.url)

    if not raw_reviews:
        raise HTTPException(status_code=404, detail="Ulasan tidak ditemukan untuk kafe ini")

    # prediksi genuine vs fake
    result = predict_reviews(raw_reviews)

    # kembalikan hasil ke web
    return {
        "cafe_name":    result["cafe_name"],
        "total":        result["total"],
        "genuine":      result["genuine_count"],
        "fake":         result["fake_count"],
        "reviews":      result["genuine_reviews"],
        "fake_reviews": result["fake_reviews"],
    }


# rekomendasi kafe serupa
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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )