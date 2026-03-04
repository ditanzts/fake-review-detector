# ============================================================
# modules/data_source/local_data.py
# Modul untuk mengambil data ulasan dari data lokal.
# ============================================================

import pandas as pd
from datetime import datetime, timedelta
import os

# Lokasi file CSV ulasan
CSV_PATH = os.path.join("data", "dsUlasan.csv")

# Kolom yang ada di CSV
REQUIRED_COLUMNS = {"reviewerId", "text", "rate", "publishedAtDate", "isLocalGuide", "placeId"}


def get_reviews(cafe_url: str) -> list[dict]:
    """
    Titik masuk utama — otomatis pilih Apify atau lokal
    berdasarkan USE_APIFY di file .env
    """
    use_apify = os.getenv("USE_APIFY", "false").lower() == "true"

    if use_apify:
        print("[INFO] Mode: Apify (scraping realtime)")
        try:
            from modules.data_source.apify_scraper import get_reviews_realtime
            return get_reviews_realtime(cafe_url)
        except Exception as e:
            print(f"[WARNING] Apify gagal: {e}")
            print("[INFO] Fallback ke data lokal...")

    print("[INFO] Mode: Data lokal (dsUlasan.csv)")
    return _get_reviews_local(cafe_url)


def _get_reviews_local(cafe_url: str) -> list[dict]:
    """
    Ambil ulasan untuk kafe tertentu dari file CSV lokal.
    Alur: URL → cari placeId di dbKafe → ambil ulasan di dsUlasan

    Parameter:
        cafe_url (str): URL Google Maps kafe yang diinput user

    Mengembalikan:
        list of dict — setiap dict adalah satu ulasan mentah
    """

    # ── Langkah 1: baca file CSV ──
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] File CSV tidak ditemukan: {CSV_PATH}")
        return []

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"[ERROR] Gagal membaca CSV: {e}")
        return []

    # ── Langkah 2: validasi kolom ──
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"[ERROR] Kolom berikut tidak ditemukan di CSV: {missing}")
        return []

    # ── Langkah 3: cari placeId via dbKafe ──
    place_id = _get_place_id_from_db(cafe_url)

    if place_id:
        df_filtered = df[df["placeId"] == place_id]
        if df_filtered.empty:
            print(f"[INFO] placeId {place_id} tidak ditemukan di dsUlasan")
            return []
    else:
        print(f"[WARNING] URL tidak ditemukan di dbKafe: {cafe_url}")
        return []

    # ── Langkah 4: filter ulasan 2 tahun terakhir ──
    df_filtered = _filter_two_years(df_filtered.copy())

    if df_filtered.empty:
        print("[INFO] Tidak ada ulasan dalam 2 tahun terakhir")
        return []

    # ── Langkah 5: bersihkan data ──
    df_filtered = _clean_data(df_filtered)

    reviews = df_filtered.to_dict(orient="records")
    print(f"[INFO] Berhasil memuat {len(reviews)} ulasan dari CSV")
    return reviews


# ============================================================
# Fungsi-fungsi pembantu (private, hanya dipakai di file ini)
# ============================================================

def _get_place_id_from_db(url: str) -> str | None:
    """
    Cari placeId berdasarkan URL dengan mencocokkan ke dbKafe.
    Mendukung short URL (maps.app.goo.gl) maupun URL panjang.
    """
    db_kafe_path = os.path.join("data", "dbKafe.csv")
    if not os.path.exists(db_kafe_path):
        print(f"[ERROR] dbKafe tidak ditemukan: {db_kafe_path}")
        return None

    try:
        df = pd.read_csv(db_kafe_path)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df["placeId"]  = df["placeId"].ffill()
        df["placeURL"] = df["placeURL"].ffill()

        # nyoba exact match dulu
        match = df[df["placeURL"] == url]

        # kalau gk ketemu, coba partial match (ambil 30 karakter pertama)
        if match.empty:
            url_short = url[:30]
            match = df[df["placeURL"].str.startswith(url_short, na=False)]

        if not match.empty:
            place_id = match.iloc[0]["placeId"]
            print(f"[INFO] placeId ditemukan: {place_id}")
            return place_id

    except Exception as e:
        print(f"[ERROR] Gagal lookup dbKafe: {e}")

    return None


def _filter_two_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter ulasan hanya yang dalam 2 tahun terakhir
    berdasarkan kolom publishedAtDate.
    """
    try:
        df["publishedAtDate"] = pd.to_datetime(df["publishedAtDate"], utc=True, errors="coerce")
        df = df.dropna(subset=["publishedAtDate"])
        from datetime import datetime, timezone
        two_years_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=730)
        df = df[df["publishedAtDate"] >= two_years_ago]
    except Exception as e:
        print(f"[WARNING] Gagal filter tanggal: {e}")
    return df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan data:
    - Hapus ulasan yang teksnya kosong
    - Pastikan tipe data sudah benar
    - Isi nilai kosong dengan default
    """
    # hapus ulasan tanpa teks
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]

    # validasi kolom rate berisi angka (1-5)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df[df["rate"].between(1, 5)]

    # memastikan isLocalGuide boolean
    # (di CSV mungkin tersimpan sebagai True/False atau 1/0)
    df["isLocalGuide"] = df["isLocalGuide"].astype(str).str.lower().isin(["true", "1", "yes"])

    # isi reviewerId yang kosong
    df["reviewerId"] = df["reviewerId"].fillna("anonymous")

    # reset index
    df = df.reset_index(drop=True)

    return df