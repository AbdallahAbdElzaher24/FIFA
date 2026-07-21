# ⚽ Unified Football Scouting System

A unified machine learning system that predicts a football player's **market value** and classifies their **performance tier**, using a suite of regression and classification models bundled together inside a single class (`UnifiedFootball`).

---

## 📌 Overview

The project is built around one main class: **`UnifiedFootball`**, which contains:

1. **Input Pipeline** (`InputPipeline`): prepares raw data for prediction.
2. **Regression models**: predict a player's market value in millions of dollars (M$).
3. **Classification models**: classify a player into a performance tier (`low`, `mid`, `high`, `elite`).
4. **Evaluation tools (Stats)**: compare the performance of all models against each other using metrics and charts.

---

## 🗂️ Class Structure

```
UnifiedFootball
├── InputPipeline           (Nested) — feature preprocessing
├── RegressionStats         (Nested) — evaluates regression models
├── ClassificationStats     (Nested) — evaluates classification models
├── predict_value()         — predicts market value
├── predict_tier()          — predicts performance tier
├── predict()               — unified prediction (value + tier)
├── predict_raw()           — predicts directly from a raw DataFrame
├── evaluate_regression()   — evaluates & compares regression models
└── evaluate_classification()— evaluates & compares classification models
```

---

## ⚙️ 1. Input Pipeline

`InputPipeline` prepares the raw data before feeding it into the models:

- **`age`** → `log1p` is applied to reduce skewness.
- **Standard scaling** on a set of numeric columns (`standard_cols`).
- **One-Hot Encoding** on the `position` column.
- Builds two separate feature matrices:
  - `X_reg`: includes `overall_rating` (after its own scaling) + numeric columns + one-hot columns, used for regression models.
  - `X_cls`: the same columns but without `overall_rating`, used for classification models.

It also includes `_classify_rating()`: a function that converts the numeric `overall_rating` into a tier based on these thresholds:

| Threshold | Tier |
|---|---|
| < 65 | `low` |
| 65 – 74 | `mid` |
| 75 – 84 | `high` |
| ≥ 85 | `elite` |

---

## 📊 2. Regression Models (Market Value Prediction)

Goal: predict a player's market value (in log scale, then converted back to millions of dollars via `expm1`).

Available models (`_reg_models`):

- Polynomial Ridge
- Ridge
- Lasso
- SVR
- Decision Tree
- Random Forest
- KNN Regressor
- Voting Regressor
- **Stacking ★** (the flagship model)

### Evaluation metrics (`RegressionStats`)
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (coefficient of determination)

Plus a bar chart comparing R² and RMSE across all models.

---

## 🏅 3. Classification Models (Performance Tier Prediction)

Goal: classify a player into one of four tiers: `low` / `mid` / `high` / `elite`.

Available models (`_cls_models`):

- KNN
- Logistic Regression
- SVM (Linear / RBF / Poly)
- Decision Tree (Gini / Entropy)
- Random Forest
- XGBoost
- Voting Classifier

### Evaluation metrics (`ClassificationStats`)
- **Accuracy**
- **F1 Score** (weighted)
- **Classification Report**
- **Confusion Matrix** (optional, as a heatmap)

Plus a bar chart comparing Accuracy and F1 across all models.

---

## 🚀 Usage

### Predicting from already-preprocessed data
```python
ufc = UnifiedFootball()

# Predict market value only
ufc.predict_value(X_reg)

# Predict performance tier only
ufc.predict_tier(X_cls)

# Unified prediction (value + tier)
ufc.predict(X_reg, X_cls)
```

### Predicting from raw data
```python
ufc.predict_raw(raw_dataframe)
```

### Evaluating and comparing models
```python
# Compare regression models
ufc.evaluate_regression(y_test_reg, X_test_all)

# Compare classification models
ufc.evaluate_classification(y_test, X_test)

# Include confusion matrix for each model
ufc.evaluate_classification(y_test, X_test, plot_cm=True)
```

---

## 📦 Requirements (Dependencies)

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
```

> ⚠️ **Note**: The class assumes several pre-trained variables already exist in scope, such as:
> `onehot`, `std_scaler`, `rating_scaler`, `ohe_cols`, `standard_cols`,
> the trained regression and classification models (e.g. `ridge_best`, `rf_best`, `gs_knn`, `voting_clf`, etc.),
> as well as `poly_best` and `scaler_poly` used by the `Polynomial Ridge` model.
> These must be trained/loaded **before** creating a `UnifiedFootball` instance.

---

## 🧠 Quick Summary

| Method | Input | Output |
|---|---|---|
| `predict_value` | `X_reg` | market value (M$) per model |
| `predict_tier` | `X_cls` | performance tier (`low/mid/high/elite`) per model |
| `predict` | `X_reg`, `X_cls` | dict combining value and tier |
| `predict_raw` | raw data | full prediction after automatic preprocessing |
| `evaluate_regression` | `y_true`, `X_reg` | comparison table + chart |
| `evaluate_classification` | `y_true`, `X_cls` | comparison table + chart |
