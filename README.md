<h1 align="center">⚽ FIFA Player Market Value — Prediction & Classification</h1>

<p align="center">
An end-to-end ML project that predicts a FIFA player's <b>market value</b> and classifies their <b>performance tier</b> using regression, classification, and ensemble learning.
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
<img src="https://img.shields.io/badge/XGBoost-2C3E50?style=for-the-badge">
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
</p>

---

## 🎯 What This Project Does

- 📊 Analyzes **19,667 players** (9 features: rating, potential, age, position, value, etc.)
- 🧮 **Regression:** predicts `value per m$` (market value)
- 🏷️ **Classification:** tiers players into `Bronze / Silver / Gold / Elite`
- 🧠 Builds and stacks multiple ML models, then proves the winning model is **stable**, not just lucky

---

## 🧹 Data Prep

- ✅ 0 nulls, 0 duplicates
- 🗑️ Dropped `name`, `country`, `team` — no real effect on value (e.g. Salah is Egyptian but worth 100M+, while avg. Egyptian player is worth 0–5M)
- ⚖️ Scaled numeric features + encoded `position` (dropped `team` to avoid 10K+ One-Hot columns 💀)

---

## 🏆 Best Models

| Task | Winner | Score |
|---|---|---|
| 📈 Regression | **Stacking Regressor** | Test R² = **0.9945** |
| 🏷️ Classification | **XGBoost** | Test Accuracy = **0.9925** |

### 📈 Regression Leaderboard
| Model | Test R² | RMSE |
|---|---|---|
| Ridge / Lasso | 0.828 | 0.311 |
| SVR | 0.971 | 0.128 |
| Decision Tree | 0.993 | 0.064 |
| Random Forest | 0.994 | 0.056 |
| **Stacking ★** | **0.9945** | **0.056** |

### 🏷️ Classification Leaderboard
| Model | Accuracy |
|---|---|
| Logistic Regression | 0.955 |
| KNN | 0.965 |
| SVM (RBF) | 0.985 |
| Random Forest | 0.989 |
| **XGBoost 🥇** | **0.9925** |
| Voting Ensemble | 0.990 |

---

## 🔍 Key Findings

- 🌀 The relationship between features and market value is **non-linear** — Ridge/Lasso underfit (R² ≈ 0.83), tree/kernel models thrive (R² ≈ 0.99+)
- 🥇 **Classification is easier than regression** here — quartile-based tiers are clean & balanced, while value is skewed with extreme outliers (superstar prices)
- 🧊 **GaussianNB is scale-invariant** — same accuracy with/without `StandardScaler`, since it works on distributions, not distances
- 🛡️ **Ridge beats Lasso** — with lots of One-Hot position features, Ridge keeps them all useful instead of zeroing them out

---

## ✅ Stability Proof

Ran K-Fold CV + Coefficient of Variation on the final Stacking model:

> **Mean R² = 0.994 | Std = 0.001 | CV% = 0.18%** → 🟢 **HIGHLY STABLE** (rule: CV% < 5% = stable)

---

## 📌 Conclusion

| Metric | Baseline | Final System |
|---|---|---|
| R² | Lower | **~0.99** |
| RMSE | Higher | **Much lower** |
| Stability | Not tested | **< 1% CV ✅** |

Ensembling (Stacking/Voting) + solid preprocessing turned a decent baseline into a near-perfect, verified-stable scouting system. ⚽📈
