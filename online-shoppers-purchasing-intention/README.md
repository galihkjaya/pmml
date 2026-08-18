# Online Shoppers Purchasing Intention

Prediksi purchase intention (kolom `Revenue`) pengunjung e-commerce, menggunakan **Online Shoppers Purchasing Intention Dataset** dari UCI Machine Learning Repository.

- **Sumber dataset:** https://archive.ics.uci.edu/dataset/468/online%20shoppers%20purchasing%20intention%20dataset
- **Kaggle Dataset (mirror publik):** https://www.kaggle.com/datasets/dimaspashaakrilian/online-shoppers-purchasing-intention-dataset
- **Kaggle Notebook:** https://www.kaggle.com/code/dimaspashaakrilian/online-shoppers-decision-tree-optuna

Notebook dijalankan di Kaggle dengan **accelerator CPU (bukan GPU T4)**.

## Struktur

```
online-shoppers-purchasing-intention/
├── data/
│   └── online_shoppers_intention.csv                 # dataset mentah (12.330 baris x 18 kolom)
├── notebooks/
│   └── online-shoppers-decision-tree/
│       ├── online-shoppers-decision-tree.ipynb        # notebook tunggal: preprocessing -> Decision Tree
│       └── kernel-metadata.json                       # metadata untuk `kaggle kernels push`
└── results/
    ├── final_decision_tree_metrics.json               # metrics lengkap model final
    └── decision_tree_threshold_comparison.csv         # perbandingan threshold default vs optimal
```

## Alur notebook (satu notebook, end-to-end)

1. **EDA & Cleaning** — cek missing value (tidak ada), duplikat (dibuang), distribusi target (`Revenue` imbalanced ~85% False / ~15% True).
2. **Feature Engineering & Encoding** — `Weekend`/`Revenue` → 0/1, one-hot encoding fitur kategorikal nominal (`Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`).
3. **Train-Test Split** — 80/20 stratified.
4. **Baseline Decision Tree** — parameter default, sebagai titik pembanding.
5. **Hyperparameter Tuning dengan Optuna** (TPE sampler, 50 trials) — objective: memaksimalkan **PR-AUC** (average precision) 5-fold cross-validation. Dipilih dibanding `GridSearchCV` karena search space mencakup parameter kontinu (`ccp_alpha`) yang lebih efisien dieksplorasi lewat Bayesian/TPE search.
6. **Threshold Tuning via PR Curve** — threshold optimal dicari dari prediksi *out-of-fold* di data train (`cross_val_predict`), **bukan** dari test set, supaya tidak *leak* informasi test set ke pemilihan model.
7. **Evaluasi Final** — bandingkan performa di threshold default (0.5) vs threshold optimal pada test set yang belum pernah dilihat model.

## Kenapa Decision Tree?

Proyek ini awalnya membandingkan beberapa model klasifikasi yang murah secara komputasi (Logistic Regression, Naive Bayes, KNN, Random Forest). KNN punya `fit_time` tercepat, tapi performanya (F1) jauh di bawah model lain bahkan setelah di-tuning. **Decision Tree dipilih sebagai model final** karena tetap murah (tidak perlu scaling secara konseptual, training cepat, tidak menyimpan seluruh dataset seperti KNN, tidak sebanyak pohon seperti Random Forest) sekaligus performanya jauh lebih baik dan stabil untuk target yang imbalanced.

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

**Interpretasi:** menurunkan threshold dari 0.5 ke 0.356 (hasil optimasi PR curve di data train, bukan test set) menaikkan recall (0.605 → 0.681) dengan sedikit trade-off precision (0.720 → 0.655), menghasilkan F1 yang lebih baik. Ini relevan karena tujuan bisnis (mendeteksi sesi dengan niat beli) biasanya lebih mementingkan recall — lebih baik menangkap lebih banyak calon pembeli walau ada sedikit false positive.

Detail lengkap ada di `results/final_decision_tree_metrics.json` dan output notebook di Kaggle.

## Cara reproduksi

1. Buka notebook di Kaggle → tambahkan Kaggle Dataset `dimaspashaakrilian/online-shoppers-purchasing-intention-dataset` sebagai data source → Run All (accelerator: **None/CPU**).

Atau push langsung via Kaggle API:

```bash
kaggle kernels push -p notebooks/online-shoppers-decision-tree
```
