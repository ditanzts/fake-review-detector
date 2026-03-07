# ============================================================
# modules/prediction.py
#
# Load 3 model .pkl (Hybrid Stacking Ensemble) dan prediksi
# ulasan fake vs genuine.
#
# Arsitektur:
#   Level 0 → SVM (TF-IDF)  : prediksi dari teks mentah
#   Level 0 → XGBoost       : prediksi dari 17 fitur numerik
#   Level 1 → XGBoost Meta  : gabungkan output level 0 → hasil akhir
# ============================================================

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.feature_extraction import extract_features, FEATURE_COLUMNS

# Lokasi file model
MODEL_SVM_PATH   = os.path.join("models", "model_svm_level0.pkl")
MODEL_XGB_PATH   = os.path.join("models", "model_xgb_level0.pkl")
MODEL_META_PATH  = os.path.join("models", "model_xgb_meta_level1.pkl")
TFIDF_PATH       = os.path.join("models", "tfidf_vectorizer.pkl")  # ← ganti nama jika beda
DB_PERILAKU_PATH = os.path.join("data", "dbPerilaku.csv")


# ============================================================
# Load semua model & dbPerilaku saat aplikasi pertama jalan
# ============================================================

def _load_model(path: str, nama: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {nama} tidak ditemukan: {path}")
    try:
        import joblib
        model = joblib.load(path)
        print(f"[INFO] {nama} berhasil dimuat (joblib)")
        return model
    except Exception:
        # fallback ke pickle kalau joblib gagal
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[INFO] {nama} berhasil dimuat (pickle)")
        return model


def _load_db_perilaku():
    if not os.path.exists(DB_PERILAKU_PATH):
        raise FileNotFoundError(f"dbPerilaku tidak ditemukan: {DB_PERILAKU_PATH}")
    df = pd.read_csv(DB_PERILAKU_PATH)
    print(f"[INFO] dbPerilaku dimuat: {len(df)} baris")
    return df


try:
    MODEL_SVM   = _load_model(MODEL_SVM_PATH,  "SVM Level-0")
    MODEL_XGB   = _load_model(MODEL_XGB_PATH,  "XGBoost Level-0")
    MODEL_META  = _load_model(MODEL_META_PATH, "XGBoost Meta Level-1")
    TFIDF       = _load_model(TFIDF_PATH,      "TF-IDF Vectorizer")
    DB_PERILAKU = _load_db_perilaku()
    MODELS_READY = True
except FileNotFoundError as e:
    print(f"[WARNING] {e}")
    MODEL_SVM = MODEL_XGB = MODEL_META = TFIDF = DB_PERILAKU = None
    MODELS_READY = False


# ============================================================
# FUNGSI UTAMA
# ============================================================

def predict_reviews(raw_reviews: list[dict]) -> dict:
    """
    Prediksi fake/genuine menggunakan Hybrid Stacking Ensemble.

    Alur:
    1. Ekstrak 17 fitur numerik → input XGBoost Level-0
    2. Ambil teks mentah       → input SVM Level-0 (TF-IDF)
    3. Gabungkan probabilitas output Level-0 → input Meta Level-1
    4. Meta model → prediksi akhir (0=fake, 1=genuine)
    """
    if not raw_reviews:
        return _empty_result()

    if not MODELS_READY:
        print("[WARNING] Model belum dimuat, pakai mode dummy")
        return _dummy_result(raw_reviews)

    # ── Langkah 1: Ekstrak 17 fitur numerik (untuk XGBoost) ──
    print(f"[INFO] Mengekstrak fitur untuk {len(raw_reviews)} ulasan...")
    features_list = []
    texts = []

    for review in raw_reviews:
        fitur = extract_features(review, DB_PERILAKU)
        features_list.append(fitur)
        texts.append(str(review.get("text", "")))

    df_features = pd.DataFrame(features_list, columns=FEATURE_COLUMNS)
    df_features = df_features.fillna(0)

    # ── Langkah 2: Prediksi Level-0 ──────────────────────────
    print("[INFO] Menjalankan prediksi Level-0...")

    # XGBoost Level-0 → probabilitas dari fitur numerik
    prob_xgb = MODEL_XGB.predict_proba(df_features)[:, 1]

    # SVM Level-0 → transform teks dulu pakai TF-IDF, baru predict
    X_tfidf  = TFIDF.transform(texts)        # list teks → sparse matrix
    prob_svm = MODEL_SVM.predict_proba(X_tfidf)[:, 1]

    # ── Langkah 3: Gabungkan output Level-0 → input Meta ─────
    # Stack jadi matrix [prob_xgb, prob_svm] per ulasan
    meta_input = np.column_stack([prob_xgb, prob_svm])

    # ── Langkah 4: Prediksi akhir oleh Meta Level-1 ───────────
    print("[INFO] Menjalankan prediksi Meta Level-1...")
    predictions = MODEL_META.predict(meta_input)
    # predictions: array 0 (fake) atau 1 (genuine)

    # ── Langkah 5: Pisahkan hasil ─────────────────────────────
    genuine_reviews = []
    fake_reviews    = []
    for review, pred in zip(raw_reviews, predictions):
        if pred == 0:  # 0 = genuine
            genuine_reviews.append(_format_review(review))
        else:          # 1 = fake
            fake_reviews.append(_format_review(review))

    genuine_count = int(np.sum(predictions == 0))
    fake_count    = int(np.sum(predictions == 1))

    print(f"[INFO] Hasil: {genuine_count} genuine, {fake_count} fake")

    return {
        "cafe_name":       _get_cafe_name(raw_reviews),
        "total":           len(raw_reviews),
        "genuine_count":   genuine_count,
        "fake_count":      fake_count,
        "genuine_reviews": genuine_reviews,
        "fake_reviews":    fake_reviews,
    }


# ============================================================
# Fungsi pembantu
# ============================================================

def _format_review(review: dict) -> dict:
    return {
        "stars":        int(review.get("rate", 0)),
        "text":         str(review.get("text", "")),
        "date":         _format_date(str(review.get("publishedAtDate", ""))),
        "isLocalGuide": bool(review.get("isLocalGuide", False)),
    }


def _format_date(date_str: str) -> str:
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days = (now - dt).days
        if days == 0:   return "Hari ini"
        if days == 1:   return "1 hari lalu"
        if days < 7:    return f"{days} hari lalu"
        if days < 14:   return "1 minggu lalu"
        if days < 30:   return f"{days // 7} minggu lalu"
        if days < 60:   return "1 bulan lalu"
        if days < 365:  return f"{days // 30} bulan lalu"
        return f"{days // 365} tahun lalu"
    except Exception:
        return date_str


def _get_cafe_name(reviews: list[dict]) -> str:
    if reviews:
        place_id = reviews[0].get("placeId", "")
        if place_id:
            from modules.recommendation import get_cafe_name_by_place_id
            return get_cafe_name_by_place_id(place_id)
    return "Kafe"


def _empty_result() -> dict:
    return {
        "cafe_name": "Kafe", "total": 0,
        "genuine_count": 0, "fake_count": 0,
        "genuine_reviews": [], "fake_reviews": []
    }


def _dummy_result(raw_reviews: list[dict]) -> dict:
    print("[INFO] Mode dummy — semua dianggap genuine")
    genuine = [_format_review(r) for r in raw_reviews if str(r.get("text", "")).strip()]
    return {
        "cafe_name":       _get_cafe_name(raw_reviews),
        "total":           len(raw_reviews),
        "genuine_count":   len(genuine),
        "fake_count":      0,
        "genuine_reviews": genuine,
        "fake_reviews":    [],
    }