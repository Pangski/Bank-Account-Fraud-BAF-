# Tugas ML — Kasus A: Deteksi Penipuan Bank (BAF)

**Nama:** Ethan Panggabean | **NIM:** 24110500012

> Status: Kasus A selesai. Kasus B (sentimen ulasan) belum dikerjakan. File `Video_Games.csv` yang diterima.

## Metode dan alasan
- Data: Bank Account Fraud (BAF) varian Base, month 0–3 (547.975 baris, penipu ±1%).
- Preprocessing di dalam `Pipeline`: nilai `-1` pada 5 kolom dianggap hilang (diimputasi median + kolom indikator), angka di-scale, kategori di-one-hot.
- Class imbalance: `class_weight="balanced"`. Tanpa membuang data dan tanpa data sintetis.
- Model: Logistic Regression (baseline) dan HistGradientBoosting (tree-based). HistGradientBoosting dituning dengan grid 8 kombinasi, dinilai di data validasi.
- Threshold dari FPR 5% di data validasi (bukan 0,5).
- `random_state = 42` di semua tempat. Model final dipilih dari skor validasi saja.

## Pembagian data
Temporal, karena data BAF berurutan per bulan dan model dipakai pada bulan berikutnya.

| Set | Month | Baris | Penipu |
|---|---|---|---|
| Latih | 0–1 | 260.060 | 2.698 |
| Validasi | 2 | 136.979 | 1.198 |
| Uji | 3 | 150.936 | 1.392 |

## Skor data validasi (month 2)
| Model | TPR @ FPR 5% | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0,4925 | 0,8758 | 0,1147 |
| **HistGradientBoosting (tuned)** | **0,5409** | **0,8851** | **0,1316** |

Skor data uji (month 3, dipakai satu kali): TPR 0,6932, FPR 0,0882, ROC-AUC 0,9113, PR-AUC 0,1513.
FPR uji (8,8%) lebih tinggi dari 5% karena distribusi skor bergeser antar bulan. Threshold dari validasi tidak otomatis menjaga FPR 5% di bulan berikutnya.

## Cara menjalankan
1. `pip install -r requirements.txt`
2. Taruh `Base.csv` di folder `data/` (tidak ikut repository).
3. Jalankan `notebook_kasus_a.ipynb` (Restart & Run All). Model tersimpan di `model/model_fraud.joblib` (berisi pipeline dan threshold).

## Sumber
- Dataset: Bank Account Fraud, Feedzai — https://github.com/feedzai/bank-account-fraud
- Tidak ada kode dari sumber luar.
