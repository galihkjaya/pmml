# Online Shoppers Purchasing Intention

Prediksi purchase intention (kolom `Revenue`) pengunjung e-commerce, menggunakan **Online Shoppers Purchasing Intention Dataset** dari UCI Machine Learning Repository.

- **Sumber dataset:** https://archive.ics.uci.edu/dataset/468/online%20shoppers%20purchasing%20intention%20dataset
- **Kaggle Dataset (mirror publik):** https://www.kaggle.com/datasets/dimaspashaakrilian/online-shoppers-purchasing-intention-dataset
- **Kaggle Notebook:** https://www.kaggle.com/code/dimaspashaakrilian/online-shoppers-decision-tree-optuna

Notebook dijalankan di Kaggle dengan **accelerator CPU (bukan GPU T4)**.

## Struktur

```
online-shoppers-decision-tree-optuna/
├── data/
│   └── online_shoppers_intention.csv                 # dataset mentah (12.330 baris x 18 kolom)
├── notebooks/
│   ├── online-shoppers-decision-tree.ipynb            # notebook tunggal: preprocessing -> 7-model -> Decision Tree
│   └── kernel-metadata.json                           # metadata untuk `kaggle kernels push`
└── results/
    ├── model_comparison_results.csv                   # perbandingan cost & performa 7 model
    ├── final_decision_tree_metrics.json                # metrics lengkap model final
    └── decision_tree_threshold_comparison.csv         # perbandingan threshold default vs optimal
```

## Alur notebook (satu notebook, end-to-end)

1. **EDA & Cleaning** — cek missing value (tidak ada), duplikat (dibuang), distribusi target (`Revenue` imbalanced ~85% False / ~15% True).
2. **Feature Engineering & Encoding** — `Weekend`/`Revenue` → 0/1, one-hot encoding fitur kategorikal nominal (`Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`).
3. **Train-Test Split** — 80/20 stratified.
4. **Perbandingan 7 model klasifikasi murah** (default params): Logistic Regression, Decision Tree, Naive Bayes, KNN, Random Forest, XGBoost, LightGBM — dibandingkan dari sisi cost (`fit_time`/`predict_time`) dan performa (accuracy, precision, recall, F1, ROC-AUC).
5. **Kenapa Decision Tree dipilih sebagai model final** — alasan berdasarkan data di langkah 4 (lihat bagian di bawah).
6. **Hyperparameter Tuning dengan Optuna** (TPE sampler, 50 trials) — objective: memaksimalkan **PR-AUC** (average precision) 5-fold cross-validation. Dipilih dibanding `GridSearchCV` karena search space mencakup parameter kontinu (`ccp_alpha`) yang lebih efisien dieksplorasi lewat Bayesian/TPE search.
7. **Threshold Tuning via PR Curve** — threshold optimal dicari dari prediksi *out-of-fold* di data train (`cross_val_predict`), **bukan** dari test set, supaya tidak *leak* informasi test set ke pemilihan model.
8. **Evaluasi Final** — bandingkan performa di threshold default (0.5) vs threshold optimal pada test set yang belum pernah dilihat model.

## Experiment Table

| Model | Tuning | Metrik | Skor (Test Set) |
|---|---|---|---:|
| Logistic Regression | Default | F1 / ROC-AUC | 0.619 / 0.908 |
| Naive Bayes (Gaussian) | Default | F1 / ROC-AUC | 0.289 / 0.581 |
| K-Nearest Neighbors | Default | F1 / ROC-AUC | 0.212 / 0.745 |
| Random Forest | Default | F1 / ROC-AUC | 0.665 / 0.927 |
| XGBoost | Default | F1 / ROC-AUC | 0.662 / 0.937 |
| LightGBM | Default | F1 / ROC-AUC | 0.664 / 0.931 |
| **Decision Tree** | **Default** | **F1 / ROC-AUC** | **0.611 / 0.826** |
| **Decision Tree** | **Optuna (50 trials)** — threshold 0.5 | **F1 / ROC-AUC** | **0.657 / 0.918** |
| **Decision Tree** | **Optuna (50 trials)** — threshold optimal (0.356) | **F1** | **0.668** |

### Detail Perbandingan 7 Model (default params, holdout test set)

| Model | fit_time (CV, ms) | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| K-Nearest Neighbors | 26.7 | 0.854 | 0.212 | 0.745 |
| Naive Bayes (Gaussian) | 26.8 | 0.245 | 0.289 | 0.581 |
| **Decision Tree** | **66.0** | **0.853** | **0.611** | **0.826** |
| XGBoost | 200.2 | 0.864 | 0.662 | 0.937 |
| LightGBM | 210.6 | 0.873 | 0.664 | 0.931 |
| Logistic Regression | 316.9 | 0.846 | 0.619 | 0.908 |
| Random Forest | 745.6 | 0.868 | 0.665 | 0.927 |

## Kenapa Decision Tree?

- **Dibanding KNN & Naive Bayes (satu-satunya model yang lebih murah):** F1 keduanya jauh di bawah Decision Tree (0.212 dan 0.289 vs **0.611**) — jadi keduanya "terlalu murah" dengan harga performa yang tidak sepadan.
- **Dibanding Random Forest, XGBoost, LightGBM (performa terbaik):** F1 mereka lebih tinggi (0.66–0.665), tapi `fit_time`-nya **3–11x lebih mahal** dari Decision Tree (RF bahkan 11.3x lebih lambat). Untuk proyek yang eksplisit mengutamakan model murah/CPU-only, biaya tambahan ini sulit dijustifikasi hanya untuk kenaikan F1 yang relatif kecil (~0.05).
- **Decision Tree = titik keseimbangan (sweet spot):** satu pohon tunggal → training cepat (peringkat cost #3 dari 7), tidak menyimpan seluruh dataset seperti KNN, dan tetap interpretable (bisa ditelusuri aturan keputusannya) — sesuatu yang tidak dimiliki Random Forest/XGBoost/LightGBM. F1-nya (0.611) jauh lebih dekat ke kelompok model "mahal" (0.62–0.665) dibanding ke kelompok "murah" lainnya (0.21–0.29).
- Gap performa yang tersisa ke Random Forest/boosting realistis dipersempit lewat hyperparameter tuning (Optuna) + threshold optimization (PR curve) — dua langkah yang tidak mengubah *fundamental cost* Decision Tree (tetap satu pohon).

## Hasil Model Final (Decision Tree, tuned dengan Optuna)

**Best hyperparameters** (dari 50 trials Optuna, objective = PR-AUC 5-fold CV):

| Parameter | Nilai |
|---|---|
| `max_depth` | 13 |
| `min_samples_split` | 22 |
| `min_samples_leaf` | 33 |
| `criterion` | entropy |
| `class_weight` | None |
| `ccp_alpha` | 0.00055 |
| Best CV PR-AUC | 0.7135 |

**Evaluasi di test set (holdout, belum pernah dilihat model):**

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Default (0.5) | 0.901 | 0.720 | 0.605 | 0.657 |
| **Optimal (PR curve, OOF) = 0.356** | 0.894 | 0.655 | 0.681 | **0.668** |

- **Test ROC-AUC:** 0.918
- **Test PR-AUC (average precision):** 0.686

Setelah tuning, F1 Decision Tree naik dari **0.611 → 0.668** — melampaui F1 Random Forest/XGBoost/LightGBM versi default (0.66–0.665), sambil tetap jauh lebih murah secara komputasi.

**Interpretasi threshold:** menurunkan threshold dari 0.5 ke 0.356 (hasil optimasi PR curve di data train, bukan test set) menaikkan recall (0.605 → 0.681) dengan sedikit trade-off precision (0.720 → 0.655). Ini relevan karena tujuan bisnis (mendeteksi sesi dengan niat beli) biasanya lebih mementingkan recall — lebih baik menangkap lebih banyak calon pembeli walau ada sedikit false positive.

Detail lengkap ada di `results/model_comparison_results.csv` dan `results/final_decision_tree_metrics.json`, serta output notebook di Kaggle.

## Cara reproduksi

1. Buka notebook di Kaggle → tambahkan Kaggle Dataset `dimaspashaakrilian/online-shoppers-purchasing-intention-dataset` sebagai data source → Run All (accelerator: **None/CPU**).

Atau push langsung via Kaggle API:

```bash
kaggle kernels push -p notebooks
```
