# Online Shoppers Purchasing Intention

Prediksi purchase intention (kolom `Revenue`) pengunjung e-commerce, menggunakan **Online Shoppers Purchasing Intention Dataset** dari UCI Machine Learning Repository.

- **Sumber dataset:** https://archive.ics.uci.edu/dataset/468/online%20shoppers%20purchasing%20intention%20dataset
- **Kaggle Dataset (mirror publik):** https://www.kaggle.com/datasets/dimaspashaakrilian/online-shoppers-purchasing-intention-dataset
- **Kaggle Notebook - Preprocessing:** https://www.kaggle.com/code/dimaspashaakrilian/online-shoppers-01-preprocessing
- **Kaggle Notebook - Modeling:** https://www.kaggle.com/code/dimaspashaakrilian/online-shoppers-02-modeling-comparison

Kedua notebook dijalankan di Kaggle dengan **accelerator CPU (bukan GPU T4)**, karena seluruh model yang dibandingkan adalah model klasik yang ringan secara komputasi.

## Struktur

```
online-shoppers-purchasing-intention/
├── data/
│   └── online_shoppers_intention.csv        # dataset mentah (12.330 baris x 18 kolom)
├── notebooks/
│   ├── 01-preprocessing/
│   │   ├── 01-preprocessing.ipynb           # EDA, cleaning, encoding, train/test split
│   │   └── kernel-metadata.json             # metadata untuk `kaggle kernels push`
│   └── 02-modeling/
│       ├── 02-modeling.ipynb                # training & perbandingan 5 model
│       └── kernel-metadata.json
└── results/
    └── model_comparison_results.csv         # hasil akhir perbandingan model
```

## 1. Preprocessing (`01-preprocessing.ipynb`)

- EDA: cek missing value (tidak ada), duplikat (125 baris dibuang), distribusi target (`Revenue` imbalanced ~85% False / ~15% True).
- Cleaning: `drop_duplicates`.
- Encoding: `Weekend`/`Revenue` → 0/1, one-hot encoding untuk fitur kategorikal nominal (`Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`).
- Split train/test 80/20 (stratified) → disimpan sebagai `train.csv` / `test.csv` (output notebook, dipakai oleh notebook modeling).

## 2. Modeling (`02-modeling.ipynb`)

Membandingkan **5 model klasifikasi yang murah secara komputasi**, semua dijalankan di CPU:

| Model | Alasan dipilih |
|---|---|
| Logistic Regression | linear, training cepat, mudah diinterpretasi |
| Decision Tree | non-linear, murah, tidak butuh scaling |
| Naive Bayes (Gaussian) | training paling ringan (hanya hitung mean/varians) |
| K-Nearest Neighbors | tanpa fase training eksplisit ("lazy learner") |
| Random Forest | ensemble ringan, sebagai pembanding model yang sedikit lebih mahal |

Perbandingan dilakukan dua sisi:
- **Cost**: `fit_time` & `score_time` dari 5-fold cross-validation, plus waktu training/prediksi penuh di holdout test.
- **Performa**: accuracy, precision, recall, F1-score, ROC-AUC (accuracy saja tidak cukup karena target imbalanced).

### Hasil (holdout test set)

| Model | fit_time (CV, ms) | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| K-Nearest Neighbors | 14.9 | 0.854 | 0.212 | 0.745 |
| Naive Bayes (Gaussian) | 18.5 | 0.245 | 0.289 | 0.581 |
| Decision Tree | 82.3 | 0.855 | 0.615 | 0.830 |
| Logistic Regression | 245.6 | 0.846 | 0.619 | 0.908 |
| Random Forest | 836.3 | 0.877 | 0.670 | 0.925 |

**Ringkasan:**
- **Termurah (waktu training tercepat):** K-Nearest Neighbors (~15 ms/fold saat CV — KNN tidak punya fase training eksplisit, hanya menyimpan data).
- **Performa terbaik:** Random Forest (F1 & ROC-AUC tertinggi), tapi juga yang termahal (~56x lebih lambat training dibanding KNN).
- **Trade-off terbaik:** Logistic Regression — murah, dan ROC-AUC-nya (0.908) mendekati Random Forest (0.925) tanpa biaya komputasi ensemble.
- Naive Bayes berkinerja buruk di dataset ini (asumsi independensi antar fitur & distribusi Gaussian tidak cocok untuk data ini).

Detail lengkap ada di `results/model_comparison_results.csv` dan output masing-masing notebook di Kaggle.

## Cara reproduksi

1. Buka notebook `01-preprocessing` di Kaggle → tambahkan Kaggle Dataset `dimaspashaakrilian/online-shoppers-purchasing-intention-dataset` sebagai data source → Run All (accelerator: **None/CPU**).
2. Buka notebook `02-modeling` → tambahkan output notebook `01-preprocessing` sebagai data source → Run All (accelerator: **None/CPU**).

Atau push langsung via Kaggle API dari folder masing-masing notebook:

```bash
kaggle kernels push -p notebooks/01-preprocessing
kaggle kernels push -p notebooks/02-modeling
```
