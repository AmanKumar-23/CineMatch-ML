<div align="center">

# 🎬 CineMatch-ML

### A Multi-Algorithm Movie Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-3.0.2-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-22c55e?style=flat-square)]()

> A comprehensive recommendation system that implements and benchmarks 11 collaborative filtering algorithms alongside content-based and popularity-based approaches — evaluated on MovieLens 100K and 1M datasets.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Datasets](#-datasets)
- [Methodology](#-methodology)
- [Results & Benchmarks](#-results--benchmarks)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)

---

## 🧠 Overview

CineMatch-ML is a research-grade movie recommendation system that explores four distinct paradigms of collaborative and content-based filtering. The system is built on the **MovieLens 100K** and **MovieLens 1M** benchmark datasets and the **TMDB 5000** movie metadata dataset.

The core objective is to systematically compare recommendation algorithms using standard IR metrics — **RMSE**, **MAE**, **Precision@K**, **Recall@K**, **F-measure**, and **NDCG** — and identify the optimal model for real-world deployment.

**Key contributions:**
- End-to-end pipeline from raw data ingestion to evaluated predictions
- Genre-aware user vector construction for content-based scoring
- Benchmarking of 11 surprise-library algorithms under identical conditions
- Weighted popularity scoring using Bayesian-style vote normalization

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CineMatch-ML                         │
├─────────────┬───────────────┬──────────────┬────────────────┤
│  Content-   │  Popularity-  │Collaborative │   KNN Deep     │
│  Based      │  Based        │  Filtering   │   Analysis     │
│  Filtering  │  Model        │  (Surprise)  │  (ML-1M)       │
├─────────────┴───────────────┴──────────────┴────────────────┤
│                    Preprocessing Pipeline                    │
│         (Train/Test Split · Genre Encoding · Vectors)        │
├─────────────────────────────────────────────────────────────┤
│              MovieLens 100K · MovieLens 1M · TMDB            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Datasets

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| MovieLens 100K | GroupLens | 100,836 ratings · 9,742 movies · 610 users | Primary training & evaluation dataset |
| MovieLens 1M | GroupLens | 1,000,209 ratings · 3,883 movies · 6,040 users | KNN deep analysis |
| TMDB 5000 | Kaggle / TMDB | 4,803 movies · 20 features | Popularity & metadata modeling |

**Train/Test Split:** 80/20 stratified split on userId for all collaborative filtering experiments.

---

## 🔬 Methodology

### 1. Content-Based Filtering (`content_based_recommendation.ipynb`)

Constructs a **genre-weighted user vector** for each user by averaging the genre one-hot vectors of all rated movies, weighted by their ratings:

```
user_vector[g] = Σ (rating_i × genre_indicator[g][i]) / count[g]
```

Predicted rating for an unseen movie is computed as the **mean of non-zero** dot product elements between the user vector and movie genre vector.

**Evaluation:** RMSE: `0.9235` · MAE: `0.7128`

---

### 2. Popularity-Based Model (`popularity_model.ipynb`)

Implements a **Bayesian-smoothed weighted rating** formula (similar to IMDb's formula):

```
WR = (v / (v + m)) × R + (m / (m + v)) × C
```

Where:
- `v` = number of votes for the movie
- `m` = minimum votes required (95th percentile = 3,040)
- `R` = average rating of the movie
- `C` = mean rating across all movies = `5.65`

Genre-aware filtering enables per-genre leaderboards across 20 genre categories.

---

### 3. Collaborative Filtering — Surprise Library (`surprise_model_recs.ipynb`)

Benchmarks **11 algorithms** from the Surprise library under identical 5-fold cross-validation conditions:

| Algorithm | Class |
|-----------|-------|
| SVD | Matrix Factorization |
| SVD++ | Matrix Factorization (implicit) |
| NMF | Non-negative Matrix Factorization |
| BaselineOnly | Baseline Estimation |
| KNNBasic | Memory-Based CF |
| KNNWithMeans | Memory-Based CF |
| KNNWithZScore | Memory-Based CF |
| KNNBaseline (cosine) | Memory-Based CF |
| KNNBaseline (pearson_baseline) | Memory-Based CF |
| SlopeOne | Slope One |
| CoClustering | Co-Clustering |

**Evaluation metrics:** RMSE · MAE · Precision@5 · Recall@5 · F-measure · NDCG

---

### 4. KNN Deep Analysis (`knn_analysis.ipynb`)

Extended KNN analysis on the **MovieLens 1M** dataset — significantly larger scale to evaluate algorithm robustness and scalability under high-cardinality user-item matrices.

---

## 📊 Results & Benchmarks

### Collaborative Filtering — Full Benchmark (sorted by RMSE)

| Algorithm | RMSE ↓ | MAE ↓ | Precision@5 ↑ | Recall@5 ↑ | F-measure ↑ | NDCG ↑ |
|-----------|--------|-------|--------------|-----------|------------|--------|
| **KNNBaseline (pearson)** | **0.8511** | **0.6493** | **0.8410** | 0.4140 | 0.5549 | **0.9645** |
| SVDpp | 0.8634 | 0.6619 | 0.8323 | 0.3983 | 0.5388 | 0.9624 |
| BaselineOnly | 0.8702 | 0.6716 | 0.8317 | 0.4095 | 0.5488 | 0.9603 |
| SVD | 0.8743 | 0.6710 | 0.8202 | 0.4021 | 0.5396 | 0.9601 |
| KNNBaseline | 0.8706 | 0.6669 | 0.8060 | 0.4162 | 0.5490 | 0.9593 |
| KNNWithZScore | 0.8902 | 0.6772 | 0.8061 | 0.4025 | 0.5369 | 0.9591 |
| KNNWithMeans | 0.8915 | 0.6833 | 0.8049 | 0.3917 | 0.5270 | 0.9586 |
| SlopeOne | 0.8976 | 0.6877 | 0.8086 | 0.4061 | 0.5407 | 0.9594 |
| NMF | 0.9188 | 0.7029 | 0.7867 | 0.3938 | 0.5249 | 0.9560 |
| KNNBasic | 0.9459 | 0.7262 | 0.7902 | **0.4280** | **0.5552** | 0.9613 |
| CoClustering | 0.9405 | 0.7294 | 0.7922 | 0.3859 | 0.5190 | 0.9550 |

> **Best model:** KNNBaseline with pearson_baseline similarity achieves the lowest RMSE (0.851) and highest NDCG (0.964), making it the recommended production model.

### Content-Based vs Collaborative Filtering

| Approach | RMSE | MAE |
|----------|------|-----|
| Content-Based (genre vectors) | 0.9236 | 0.7128 |
| Best Collaborative (KNNBaseline) | 0.8511 | 0.6493 |
| Improvement | **↓ 7.8%** | **↓ 8.9%** |

---

## 📁 Project Structure

```
CineMatch-ML/
│
├── Code/
│   ├── preprocessing.ipynb                 # Data pipeline & train/test split
│   ├── content_based_recommendation.ipynb  # Genre-vector content filtering
│   ├── popularity_model.ipynb              # Weighted popularity rankings
│   ├── surprise_model_recs.ipynb           # 11-algorithm CF benchmark
│   ├── knn_analysis.ipynb                  # KNN on MovieLens 1M
│   ├── hybrid_model.ipynb                  # Hybrid approach
│   ├── model_hyperparameter_tuning.ipynb   # GridSearchCV tuning
│   ├── evaluating_recs.py                  # Evaluation utilities
│   ├── generating_predictions.py           # Prediction pipeline
│   ├── movies.csv                          # MovieLens movie metadata
│   ├── ratings.csv                         # MovieLens 100K ratings
│   ├── tags.csv                            # MovieLens tags
│   └── movies_tmdb.csv                     # TMDB 5000 metadata
│
└── README.md
```

---

## ⚙️ Installation & Setup

**Requirements:** Python 3.8+, pip

```bash
# Clone the repository
git clone https://github.com/AmanKumar-23/CineMatch-ML.git
cd CineMatch-ML

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise jupyter

# Downgrade numpy for surprise compatibility
pip install "numpy<2"
```

**Datasets required:**
- MovieLens 100K → [grouplens.org/datasets/movielens/latest](https://grouplens.org/datasets/movielens/latest/)
- TMDB 5000 → [kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Place all CSV files inside the `Code/` directory.

---

## ▶️ How to Run

```bash
# Launch Jupyter
jupyter notebook
```

Run notebooks in this exact order:

```
1. preprocessing.ipynb                 → generates training_data.csv & testing_data.csv
2. content_based_recommendation.ipynb  → content-based predictions & evaluation
3. popularity_model.ipynb              → genre-wise popularity rankings
4. surprise_model_recs.ipynb           → full 11-algorithm benchmark
5. knn_analysis.ipynb                  → KNN deep dive on MovieLens 1M
```

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Data Processing | Pandas 3.0, NumPy 1.26 |
| Machine Learning | Scikit-learn 1.8, Scikit-surprise 1.1.4 |
| Visualization | Matplotlib 3.10, Seaborn 0.13 |
| Environment | Jupyter Notebook, Anaconda |
| Datasets | MovieLens 100K, MovieLens 1M, TMDB 5000 |
| Algorithms | SVD, SVD++, NMF, KNN variants, SlopeOne, CoClustering |

---

<div align="center">

**If you found this useful, please consider giving it a ⭐**

</div>
