import re
import numpy  as np
import pandas as pd

# kata promotional / spam (f07) 
PROMO_KEYWORDS = {
    'rekomendasi', 'rekomen', 'terbaik', 'mantap', 'keren', 'hits',
    'viral', 'instagramable', 'aesthetic', 'wajib', 'harus', 'pasti',
    'sempurna', 'amazing', 'luar biasa', 'worth', 'bagus banget',
    'enak banget', 'ramah banget', 'bersih banget',
}

# kata konjungsi (f08) 
CONJUNCTION_WORDS = {
    'dan', 'atau', 'tetapi', 'tapi', 'namun', 'melainkan', 'sedangkan',
    'karena', 'sebab', 'sehingga', 'supaya', 'agar', 'meskipun', 'walaupun',
    'ketika', 'saat', 'setelah', 'sebelum', 'jika', 'kalau', 'apabila',
    'bahwa', 'yang', 'lalu', 'kemudian', 'selain', 'bahkan', 'apalagi',
    'and', 'or', 'but', 'however', 'although', 'though', 'because',
    'since', 'so', 'yet', 'while', 'when', 'after', 'before', 'if',
    'unless', 'until', 'that', 'which', 'then', 'also', 'moreover',
}

# emoji pattern 
_EMOJI_PATTERN = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF"
    "\u2700-\u27BF]+",
    flags=re.UNICODE,
)


# FITUR TEKSTUAL  F01–F09  (per ulasan)

def extract_text_features(text: str):
    """
    Persis sama dengan extract_text_features() di Step 2.5 pipeline.
    Mengembalikan tuple (f01, f02, f03, f04, f05, f06, f07, f08, f09).
    """
    if not isinstance(text, str):
        text = ''

    chars   = list(text)
    n_chars = max(len(chars), 1)
    words   = text.split()
    n_words = max(len(words), 1)

    # f01 — punctuationDensity
    f01 = len(re.findall(r'[^\w\s]', text)) / n_chars

    # f02 — uppercaseRatio
    f02 = len([c for c in chars if c.isupper()]) / n_chars

    # f03 — emojiRatio
    f03 = len(_EMOJI_PATTERN.findall(text)) / n_words

    # f04 — repeatedCharRatio
    f04 = len(re.findall(r'(.)\1{2,}', text)) / n_chars

    # f05 — specialCharRatio
    f05 = len(re.findall(r'[@#$%&*]', text)) / n_chars

    # f06 — uniqueWordRatioPerReview
    f06 = len(set(w.lower() for w in words)) / n_words

    # f07 — keywordPresence (proporsi keyword ditemukan)
    text_lower = text.lower()
    f07 = sum(1 for kw in PROMO_KEYWORDS if kw in text_lower) / len(PROMO_KEYWORDS)

    # f08 — hasConjunction (binary 0/1)
    f08 = int(any(w.lower() in CONJUNCTION_WORDS for w in words))

    # f09 — reviewLength (jumlah kata)
    f09 = len(words)

    return f01, f02, f03, f04, f05, f06, f07, f08, f09


# FITUR PERILAKU  F10–F14  (lookup reviewerId di dbPerilaku)

def compute_behavioral_features(reviewer_id: str,
                                  db_perilaku: pd.DataFrame,
                                  current_rate: int = 3,
                                  current_is_local_guide: bool = False) -> dict:
    """
    Lookup fitur perilaku dari dbPerilaku berdasarkan reviewerId.

    Sesuai Step 2.3 pipeline:
      - f10_avgRating       : mean(rate) dari dbPerilaku
      - f11_stdRating       : std(rate) dari dbPerilaku
      - f12_reviewFreq      : count ulasan di dbPerilaku
      - f13_uniqueWordRatio : rata-rata rasio kata unik per reviewer
      - f14_isLocalGuide    : first(isLocalGuide) dari dbPerilaku

    Fallback jika reviewer tidak ditemukan (sesuai fillna di Step 2.5):
      f10 → rate ulasan saat ini
      f11 → 0.0
      f12 → 1
      f13 → 0.5
      f14 → isLocalGuide dari ulasan itu sendiri
    """
    riwayat = db_perilaku[db_perilaku['reviewerId'] == reviewer_id]

    if riwayat.empty:
        return {
            'f10_avgRating':       float(current_rate),
            'f11_stdRating':       0.0,
            'f12_reviewFreq':      1,
            'f13_uniqueWordRatio': 0.5,
            'f14_isLocalGuide':    int(current_is_local_guide),
        }

    # f10 — avgRating
    avg_rating = float(riwayat['rate'].mean())

    # f11 — stdRating
    std_rating = float(riwayat['rate'].std())
    if np.isnan(std_rating):
        std_rating = 0.0

    # f12 — reviewFreq
    review_freq = len(riwayat)

    # f13 — uniqueWordRatio (rata-rata rasio kata unik per reviewer)
    def _uwr(t):
        words = str(t).lower().split() if isinstance(t, str) else []
        return len(set(words)) / max(len(words), 1)

    uwr_vals = riwayat['text'].dropna().apply(_uwr)
    unique_word_ratio = float(uwr_vals.mean()) if not uwr_vals.empty else 0.5

    # f14 — isLocalGuide dari dbPerilaku (first), sesuai Step 2.3
    is_local = riwayat['isLocalGuide'].iloc[0]
    try:
        is_local = int(bool(is_local))
    except Exception:
        is_local = int(current_is_local_guide)

    return {
        'f10_avgRating':       round(avg_rating, 4),
        'f11_stdRating':       round(std_rating, 4),
        'f12_reviewFreq':      int(review_freq),
        'f13_uniqueWordRatio': round(unique_word_ratio, 4),
        'f14_isLocalGuide':    is_local,
    }


# FUNGSI UTAMA — Ekstraksi semua 14 fitur untuk satu ulasan

def extract_features(review: dict, db_perilaku: pd.DataFrame) -> dict:
    """
    Ekstrak 14 fitur dari satu ulasan.

    Parameter:
        review      : dict satu ulasan (text, reviewerId, rate, isLocalGuide, ...)
        db_perilaku : DataFrame dbPerilaku.csv

    Mengembalikan dict 14 fitur dengan urutan sama seperti saat training.
    """
    text        = str(review.get('text', ''))
    reviewer_id = str(review.get('reviewerId', ''))
    is_local    = bool(review.get('isLocalGuide', False))
    rate        = int(review.get('rate', 3))

    # F01–F09 tekstual
    f01, f02, f03, f04, f05, f06, f07, f08, f09 = extract_text_features(text)

    # F10–F14 perilaku (lookup dbPerilaku)
    # f14_isLocalGuide diambil dari dbPerilaku (first), fallback ke isLocalGuide ulasan
    perilaku = compute_behavioral_features(
        reviewer_id,
        db_perilaku,
        current_rate=rate,
        current_is_local_guide=is_local,
    )

    return {
        'f01_punctuationDensity':       f01,
        'f02_uppercaseRatio':           f02,
        'f03_emojiRatio':               f03,
        'f04_repeatedCharRatio':        f04,
        'f05_specialCharRatio':         f05,
        'f06_uniqueWordRatioPerReview': f06,
        'f07_keywordPresence':          f07,
        'f08_hasConjunction':           f08,
        'f09_reviewLength':             f09,
        'f10_avgRating':                perilaku['f10_avgRating'],
        'f11_stdRating':                perilaku['f11_stdRating'],
        'f12_reviewFreq':               perilaku['f12_reviewFreq'],
        'f13_uniqueWordRatio':          perilaku['f13_uniqueWordRatio'],
        'f14_isLocalGuide':             perilaku['f14_isLocalGuide'],
    }


# urutan kolom 
FEATURE_COLUMNS = [
    'f01_punctuationDensity',
    'f02_uppercaseRatio',
    'f03_emojiRatio',
    'f04_repeatedCharRatio',
    'f05_specialCharRatio',
    'f06_uniqueWordRatioPerReview',
    'f07_keywordPresence',
    'f08_hasConjunction',
    'f09_reviewLength',
    'f10_avgRating',
    'f11_stdRating',
    'f12_reviewFreq',
    'f13_uniqueWordRatio',
    'f14_isLocalGuide',
]