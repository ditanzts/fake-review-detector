# ============================================================
# modules/feature_extraction.py
# Mengubah ulasan mentah (raw) menjadi 17 fitur numerik untuk xgb base learner
# (tfidfVector ditangani terpisah oleh SVM)
# ============================================================

import re
import pandas as pd
import numpy as np
from collections import Counter

# ── Daftar kata promotional / spam ──
PROMOTIONAL_KEYWORDS = [
    "terbaik", "terpercaya", "recommended", "rekomendasi", "wajib coba",
    "paling enak", "luar biasa", "mantap", "keren banget", "sempurna",
    "lezat", "amazing", "top", "bagus banget", "murah meriah", "worth it",
    "must visit", "favorit", "nomor satu", "terfavorit", "terenak"
]


# ============================================================
# FITUR TEKSTUAL (12 fitur, dihitung per ulasan)
# ============================================================

def exclamation_ratio(text: str) -> float:
    """Rasio tanda seru terhadap total karakter"""
    if not text:
        return 0.0
    return text.count("!") / len(text)


def uppercase_ratio(text: str) -> float:
    """Rasio huruf kapital terhadap semua huruf"""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def emoji_ratio(text: str) -> float:
    """Rasio karakter emoji terhadap total karakter"""
    if not text:
        return 0.0
    # Deteksi emoji berdasarkan range unicode
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"   # symbols & pictographs
        "\U0001F680-\U0001F6FF"   # transport & map
        "\U0001F1E0-\U0001F1FF"   # flags
        "\U00002700-\U000027BF"   # dingbats
        "\U0001F900-\U0001F9FF]", # supplemental symbols
        flags=re.UNICODE
    )
    emoji_count = len(emoji_pattern.findall(text))
    return emoji_count / len(text)


def repeated_char_ratio(text: str) -> float:
    """Rasio karakter yang muncul berulang ≥3 kali berturut-turut"""
    if not text:
        return 0.0
    # Contoh: "wkwkwk", "hahaha", "enak!!!"
    matches = re.findall(r'(.)\1{2,}', text)
    repeated_chars = sum(len(m) + 2 for m in matches)
    return min(repeated_chars / len(text), 1.0)


def special_char_ratio(text: str) -> float:
    """Rasio karakter spesial (@, #, $, %, &, *) terhadap total karakter"""
    if not text:
        return 0.0
    special = re.findall(r'[@#$%&*]', text)
    return len(special) / len(text)


def punctuation_density(text: str) -> float:
    """Rasio total tanda baca terhadap total karakter"""
    if not text:
        return 0.0
    puncts = re.findall(r'[^\w\s]', text)
    return len(puncts) / len(text)


def capital_word_ratio(text: str) -> float:
    """Rasio kata yang seluruh hurufnya kapital"""
    words = text.split()
    if not words:
        return 0.0
    capital_words = [w for w in words if w.isalpha() and w.isupper() and len(w) > 1]
    return len(capital_words) / len(words)


def short_word_ratio(text: str) -> float:
    """Rasio kata pendek (≤2 huruf) terhadap total kata"""
    words = text.split()
    if not words:
        return 0.0
    short = [w for w in words if len(w) <= 2]
    return len(short) / len(words)


def avg_word_length(text: str) -> float:
    """Rata-rata panjang kata dalam ulasan"""
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def review_length(text: str) -> int:
    """Jumlah kata dalam ulasan"""
    return len(text.split())


def unique_word_ratio_per_review(text: str) -> float:
    """Rasio kata unik terhadap total kata dalam satu ulasan"""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def keyword_presence(text: str) -> int:
    """Jumlah kemunculan kata promotional dalam ulasan"""
    text_lower = text.lower()
    count = sum(1 for kw in PROMOTIONAL_KEYWORDS if kw in text_lower)
    return count


# ============================================================
# FITUR PERILAKU (5 fitur numerik, dihitung per reviewerId)
# isLocalGuide sudah ada langsung dari data ulasan
# ============================================================

def compute_behavioral_features(reviewer_id: str, db_perilaku: pd.DataFrame) -> dict:
    """
    Hitung fitur perilaku berdasarkan riwayat ulasan reviewer
    di dbPerilaku.

    Parameter:
        reviewer_id  : ID reviewer yang ingin dihitung fiturnya
        db_perilaku  : DataFrame dbPerilaku.csv

    Mengembalikan dict berisi:
        avgRating, stdRating, reviewFreq, reviewLengthAvg, uniqueWordRatio
    """
    # Ambil semua ulasan milik reviewer ini dari dbPerilaku
    riwayat = db_perilaku[db_perilaku["reviewerId"] == reviewer_id]

    # Kalau reviewer tidak ada di dbPerilaku, pakai nilai default
    if riwayat.empty:
        return {
            "avgRating":       3.0,   # nilai tengah
            "stdRating":       0.0,
            "reviewFreq":      1,
            "reviewLengthAvg": 0.0,
            "uniqueWordRatio": 0.0,
        }

    # avgRating: rata-rata rating semua ulasan reviewer ini
    avg_rating = riwayat["rate"].mean()

    # stdRating: standar deviasi rating (seberapa konsisten ratingnya)
    std_rating = riwayat["rate"].std() if len(riwayat) > 1 else 0.0

    # reviewFreq: total jumlah ulasan reviewer ini
    review_freq = len(riwayat)

    # reviewLengthAvg: rata-rata panjang teks ulasan (dalam kata)
    lengths = riwayat["text"].dropna().apply(lambda t: len(str(t).split()))
    review_length_avg = lengths.mean() if not lengths.empty else 0.0

    # uniqueWordRatio: rata-rata rasio kata unik per akun
    def _uwr(t):
        words = str(t).lower().split()
        return len(set(words)) / len(words) if words else 0.0

    uwr_values = riwayat["text"].dropna().apply(_uwr)
    unique_word_ratio = uwr_values.mean() if not uwr_values.empty else 0.0

    return {
        "avgRating":       round(avg_rating, 4),
        "stdRating":       round(std_rating if not np.isnan(std_rating) else 0.0, 4),
        "reviewFreq":      review_freq,
        "reviewLengthAvg": round(review_length_avg, 4),
        "uniqueWordRatio": round(unique_word_ratio, 4),
    }


# ============================================================
# FUNGSI UTAMA — Ekstraksi semua fitur untuk satu ulasan
# ============================================================

def extract_features(review: dict, db_perilaku: pd.DataFrame) -> dict:
    """
    Ekstrak semua 17 fitur dari satu ulasan.

    Parameter:
        review      : dict satu ulasan (dari local_data.py)
        db_perilaku : DataFrame dbPerilaku.csv

    Mengembalikan dict 17 fitur siap masuk model XGBoost.
    """
    text = str(review.get("text", ""))
    reviewer_id = str(review.get("reviewerId", ""))
    is_local_guide = bool(review.get("isLocalGuide", False))

    # ── Fitur Tekstual ──
    tekstual = {
        "f01_exclamationRatio":         exclamation_ratio(text),
        "f02_uppercaseRatio":           uppercase_ratio(text),
        "f03_emojiRatio":               emoji_ratio(text),
        "f04_repeatedCharRatio":        repeated_char_ratio(text),
        "f05_specialCharRatio":         special_char_ratio(text),
        "f06_punctuationDensity":       punctuation_density(text),
        "f07_capitalWordRatio":         capital_word_ratio(text),
        "f08_shortWordRatio":           short_word_ratio(text),
        "f09_avgWordLength":            avg_word_length(text),
        "f10_reviewLength":             review_length(text),
        "f11_uniqueWordRatioPerReview": unique_word_ratio_per_review(text),
        "f12_keywordPresence":          keyword_presence(text),
    }

    # ── Fitur Perilaku ──
    perilaku_raw = compute_behavioral_features(reviewer_id, db_perilaku)

    perilaku = {
        "f13_avgRating":       perilaku_raw["avgRating"],
        "f14_stdRating":       perilaku_raw["stdRating"],
        "f15_reviewFreq":      perilaku_raw["reviewFreq"],
        "f16_reviewLengthAvg": perilaku_raw["reviewLengthAvg"],
        "f17_uniqueWordRatio": perilaku_raw["uniqueWordRatio"],
        "f18_isLocalGuide":    int(is_local_guide),
    }

    # Gabung semua fitur
    return {**tekstual, **perilaku}

# ============================================================
# Urutan kolom fitur — sdh sama dengan waktu model dilatih
# ============================================================
FEATURE_COLUMNS = [
    "f01_exclamationRatio",
    "f02_uppercaseRatio",
    "f03_emojiRatio",
    "f04_repeatedCharRatio",
    "f05_specialCharRatio",
    "f06_punctuationDensity",
    "f07_capitalWordRatio",
    "f08_shortWordRatio",
    "f09_avgWordLength",
    "f10_reviewLength",
    "f11_uniqueWordRatioPerReview",
    "f12_keywordPresence",
    "f13_avgRating",
    "f14_stdRating",
    "f15_reviewFreq",
    "f16_reviewLengthAvg",
    "f17_uniqueWordRatio",
    "f18_isLocalGuide",
]