# ============================================================
# modules/recommendation.py
#
# Mencari rekomendasi kafe serupa berdasarkan:
# 1. Kesamaan label/tag dengan kafe yang diinput user
# 2. Jarak terdekat (menggunakan latitude & longitude)
# ============================================================

import os
import math
import pandas as pd

DB_KAFE_PATH = os.path.join("data", "dbKafe.csv")

# Jumlah maksimal rekomendasi yang ditampilkan
MAX_RECOMMENDATIONS = 5


# ============================================================
# Load & proses dbKafe saat modul diimport
# ============================================================

def _load_db_kafe() -> pd.DataFrame:
    """
    Load dbKafe.csv dan collapse label yang multi-baris
    menjadi satu baris per kafe.

    Struktur asli CSV (tiap kafe punya beberapa baris label):
        placeId | title | placeURL | lat | lng | label
        ABC       Kafe A   http://..   -8.1   113.7   estetis
        NaN       NaN      NaN         NaN    NaN     santai
        NaN       NaN      NaN         NaN    NaN     nyaman

    Setelah diproses jadi:
        placeId | title | placeURL | lat      | lng     | labels
        ABC       Kafe A   http://..   -8.1   113.7   [estetis, santai, nyaman]
    """
    if not os.path.exists(DB_KAFE_PATH):
        print(f"[ERROR] dbKafe tidak ditemukan: {DB_KAFE_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(DB_KAFE_PATH)

    # Hapus kolom Unnamed kalau ada
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Rename kolom biar lebih mudah dipakai
    df = df.rename(columns={
        "location/lat": "lat",
        "location/lng": "lng",
        "label":        "label",
        "title":        "name",
        "placeURL":     "url",
    })

    # ── Collapse multi-baris label jadi satu baris per kafe ──
    df["placeId"] = df["placeId"].ffill()
    df["name"]    = df["name"].ffill()
    df["url"]     = df["url"].ffill()
    df["lat"]     = df["lat"].ffill()
    df["lng"]     = df["lng"].ffill()

    # Group by placeId, kumpulkan semua label jadi list
    df_grouped = df.groupby("placeId", sort=False).agg(
        name   = ("name",  "first"),
        url    = ("url",   "first"),
        lat    = ("lat",   "first"),
        lng    = ("lng",   "first"),
        labels = ("label", lambda x: [str(v).strip().lower() for v in x if pd.notna(v)])
    ).reset_index()

    print(f"[INFO] dbKafe dimuat: {len(df_grouped)} kafe")
    return df_grouped


# Load sekali saat modul diimport
try:
    DB_KAFE = _load_db_kafe()
except Exception as e:
    print(f"[WARNING] Gagal load dbKafe: {e}")
    DB_KAFE = pd.DataFrame()


# ============================================================
# FUNGSI UTAMA
# ============================================================

def get_recommendations(cafe_url: str) -> list[dict]:
    """
    Cari rekomendasi kafe serupa berdasarkan URL kafe yang diinput.

    Langkah:
    1. Cari kafe asal di dbKafe berdasarkan URL
    2. Ambil label-labelnya
    3. Hitung skor kesamaan label dengan semua kafe lain
    4. Urutkan berdasarkan skor kesamaan, lalu jarak terdekat
    5. Kembalikan top N rekomendasi

    Parameter:
        cafe_url : URL Google Maps kafe yang sedang dilihat user

    Mengembalikan list of dict:
    [
        {
            "name":     "Grand Cafe Jember",
            "tags":     ["estetis", "nyaman"],
            "distance": "1.2 km",
            "url":      "https://maps.app.goo.gl/..."
        },
        ...
    ]
    """
    if DB_KAFE.empty:
        print("[WARNING] dbKafe kosong, tidak bisa beri rekomendasi")
        return []

    # ── Langkah 1: cari kafe asal ──
    origin = _find_cafe_by_url(cafe_url)

    if origin is None:
        print(f"[WARNING] Kafe tidak ditemukan di dbKafe untuk URL: {cafe_url}")
        return []

    origin_labels = origin["labels"]
    origin_lat    = origin["lat"]
    origin_lng    = origin["lng"]
    origin_id     = origin["placeId"]

    print(f"[INFO] Kafe asal: {origin['name']} | Labels: {origin_labels}")

    # ── Langkah 2: Hitung skor kesamaan untuk setiap kafe ──
    results = []

    for _, kafe in DB_KAFE.iterrows():
        # Skip kafe yang sama dengan kafe asal (pakai placeId)
        if kafe["placeId"] == origin_id:
            continue

        score       = _jaccard_similarity(origin_labels, kafe["labels"])
        distance_km = _haversine(origin_lat, origin_lng, kafe["lat"], kafe["lng"])

        results.append({
            "name":        kafe["name"],
            "tags":        kafe["labels"],
            "distance_km": distance_km,
            "score":       score,
            "url":         kafe["url"],
            "place_id":    kafe["placeId"],
        })

    if not results:
        return []

    # ── Langkah 3: urutkan — skor tertinggi, jarak terdekat ──
    results.sort(key=lambda x: (-x["score"], x["distance_km"]))

    # Ambil top N
    top = results[:MAX_RECOMMENDATIONS]

    # ── Langkah 4: format untuk frontend ──
    recommendations = []
    for r in top:
        recommendations.append({
            "name":     r["name"],
            "tags":     r["tags"],
            "distance": _format_distance(r["distance_km"]),
            "url":      r["url"],
            "place_id": r["place_id"],
        })

    print(f"[INFO] {len(recommendations)} rekomendasi ditemukan")
    return recommendations


# ============================================================
# Fungsi pembantu
# ============================================================

def _find_cafe_by_url(url: str):
    """Cari kafe di dbKafe berdasarkan URL (exact atau partial match)"""
    # Coba exact match dulu
    match = DB_KAFE[DB_KAFE["url"] == url]

    # Kalau tidak ketemu, coba partial match
    if match.empty:
        match = DB_KAFE[DB_KAFE["url"].str.contains(url[:20], na=False)]

    if match.empty:
        return None

    return match.iloc[0]


def _jaccard_similarity(labels_a: list, labels_b: list) -> float:
    """
    Hitung Jaccard similarity antara dua list label.
    Hasilnya antara 0.0 (tidak ada kesamaan) sampai 1.0 (identik).

    Contoh:
        A = [estetis, nyaman, santai]
        B = [estetis, santai, luas]
        intersection = {estetis, santai} → 2
        union        = {estetis, nyaman, santai, luas} → 4
        Jaccard      = 2/4 = 0.5
    """
    set_a = set(labels_a)
    set_b = set(labels_b)

    if not set_a and not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Hitung jarak antara dua titik koordinat (dalam km).
    Menggunakan Haversine formula — akurat untuk jarak pendek.
    """
    try:
        R = 6371  # radius bumi dalam km

        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

        dlat = lat2 - lat1
        dlng = lng2 - lng1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return round(R * c, 2)
    except Exception:
        return 9999.0  # kalau koordinat tidak valid


def _format_distance(km: float) -> str:
    """Ubah jarak float ke string"""
    if km < 1.0:
        return f"{int(km * 1000)} m"
    return f"{km:.1f} km"


# ============================================================
# Fungsi tambahan — untuk main.py ambil nama kafe dari placeId
# ============================================================

def get_cafe_name_by_place_id(place_id: str) -> str:
    """Ambil nama kafe dari placeId, untuk ditampilkan di results.html"""
    if DB_KAFE.empty:
        return place_id
    match = DB_KAFE[DB_KAFE["placeId"] == place_id]
    if match.empty:
        return place_id
    return match.iloc[0]["name"]