# modules/data_source/apify_scraper.py
# Modul scraping realtime menggunakan Apify API.

import os
import time
import requests
from datetime import datetime, timezone

# ── Konfigurasi ──
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
ACTOR_ID    = "compass~google-maps-reviews-scraper"
BASE_URL    = "https://api.apify.com/v2"

# Maksimal ulasan yang diambil per kafe (limit kredit Apify)
MAX_REVIEWS = 150

# Timeout menunggu hasil scraping (detik)
TIMEOUT_SECONDS = 180


def get_reviews_realtime(cafe_url: str) -> list[dict]:
    """
    Scraping ulasan kafe secara realtime via Apify.

    Parameter:
        cafe_url : URL Google Maps kafe

    Mengembalikan:
        list of dict dengan kolom:
        reviewerId, text, rate, publishedAtDate, isLocalGuide, placeId
    """

    if not APIFY_TOKEN:
        raise ValueError(
            "APIFY_TOKEN belum diisi. "
            "Tambahkan APIFY_TOKEN=token_kamu di file .env"
        )

    print(f"[Apify] Memulai scraping untuk: {cafe_url}")

    # ── Langkah 1: menjalankan actor ──
    run_id = _start_actor(cafe_url)
    if not run_id:
        raise RuntimeError("Gagal menjalankan Apify Actor")

    print(f"[Apify] Actor berjalan, run_id: {run_id}")

    # ── Langkah 2: tunggu ──
    success = _wait_for_completion(run_id)
    if not success:
        raise RuntimeError("Apify Actor timeout atau gagal")

    # ── Langkah 3: ambil hasil ──
    raw_items = _fetch_results(run_id)
    print(f"[Apify] Berhasil mengambil {len(raw_items)} ulasan mentah")

    # ── Langkah 4: transformasi ke format sistem ──
    reviews = _transform(raw_items, cafe_url)
    print(f"[Apify] {len(reviews)} ulasan siap diproses")

    return reviews


# ============================================================
# Fungsi pembantu
# ============================================================

def _start_actor(cafe_url: str) -> str | None:
    """Jalankan Apify Actor dan kembalikan run_id"""
    endpoint = f"{BASE_URL}/acts/{ACTOR_ID}/runs"

    payload = {
        "startUrls": [{"url": cafe_url}],
        "maxReviews": MAX_REVIEWS,
        "language":   "id",          
        "sort":       "newest", 
    }

    try:
        resp = requests.post(
            endpoint,
            json=payload,
            params={"token": APIFY_TOKEN},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["data"]["id"]
    except Exception as e:
        print(f"[Apify ERROR] Gagal start actor: {e}")
        return None


def _wait_for_completion(run_id: str) -> bool:
    """
    Poll status Actor sampai SUCCEEDED atau timeout.
    Cek setiap 5 detik.
    """
    endpoint = f"{BASE_URL}/actor-runs/{run_id}"
    elapsed  = 0

    while elapsed < TIMEOUT_SECONDS:
        try:
            resp = requests.get(
                endpoint,
                params={"token": APIFY_TOKEN},
                timeout=10
            )
            status = resp.json()["data"]["status"]
            print(f"[Apify] Status: {status} ({elapsed}s)")

            if status == "SUCCEEDED":
                return True
            if status in ("FAILED", "ABORTED", "TIMED-OUT"):
                print(f"[Apify ERROR] Actor berhenti dengan status: {status}")
                return False

        except Exception as e:
            print(f"[Apify WARNING] Gagal cek status: {e}")

        time.sleep(5)
        elapsed += 5

    print(f"[Apify ERROR] Timeout setelah {TIMEOUT_SECONDS} detik")
    return False


def _fetch_results(run_id: str) -> list[dict]:
    """Ambil hasil dataset dari run yang sudah selesai"""
    endpoint = f"{BASE_URL}/actor-runs/{run_id}/dataset/items"

    try:
        resp = requests.get(
            endpoint,
            params={"token": APIFY_TOKEN, "limit": MAX_REVIEWS},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[Apify ERROR] Gagal ambil hasil: {e}")
        return []


def _transform(items: list[dict], cafe_url: str) -> list[dict]:
    """
    Ubah format Apify → format sistem kita.

    Nama kolom Apify  →  Nama kolom sistem
    ─────────────────────────────────────
    reviewerId         →  reviewerId
    text               →  text
    stars              →  rate          ← BEDA NAMA!
    publishedAtDate    →  publishedAtDate
    isLocalGuide       →  isLocalGuide
    placeId            →  placeId
    """
    two_years_ago = datetime.now(timezone.utc).replace(
        year=datetime.now().year - 2
    )

    results = []
    for item in items:
        # Skip kalau tidak ada teks
        text = item.get("text", "")
        if not text or not str(text).strip():
            continue

        # Filter 2 tahun terakhir
        try:
            date_str = item.get("publishedAtDate", "")
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt < two_years_ago:
                continue
        except Exception:
            pass  # kalau tanggal tidak valid, tetap include

        results.append({
            "reviewerId":      str(item.get("reviewerId", "anonymous")),
            "text":            str(text),
            "rate":            int(item.get("stars", 0)),   # stars → rate
            "publishedAtDate": item.get("publishedAtDate", ""),
            "isLocalGuide":    bool(item.get("isLocalGuide", False)),
            "placeId":         str(item.get("placeId", "")),
        })

    return results