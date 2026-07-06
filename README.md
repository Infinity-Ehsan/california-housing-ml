# 🏠 California Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine-Learning-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Model-green)
![Status](https://img.shields.io/badge/Project-Complete-success)

---

## 📌 Project Overview

This project is an end-to-end **machine learning regression pipeline** built on the California Housing dataset.

The goal is to predict house prices and compare multiple ML models using:

- Baseline training
- Hyperparameter tuning (GridSearchCV)
- Cross-validation (KFold)
- Model evaluation (R², MSE)
- Model interpretability (SHAP)

---

## 📊 Dataset

- Source: `sklearn.datasets.fetch_california_housing`
- Samples: ~20,640
- Features: 8 numerical features
- Target: Median House Value

---

## ⚙️ Workflow

### 1. Data Preparation
- Load dataset
- Convert to DataFrame
- Define features (X) and target (y)
- Train/Test split (80/20)

---

### 2. Models Implemented

- Ridge Regression (Linear baseline)
- Random Forest Regressor
- Gradient Boosting Regressor

---

### 3. Model Training Strategy

For each model:

- Baseline model training
- Cross-validation (KFold = 5)
- Hyperparameter tuning (GridSearchCV)
- Best model selection

---

### 4. Evaluation Metrics

- R² Score (Primary metric)
- Mean Squared Error (MSE)
- Cross-validation Mean & Standard Deviation (Variance)

---

## 🏆 Final Results (Leaderboard)

| Rank | Model | CV R² | Test R² |
|------|------|------|--------|
| 🥇 | Gradient Boosting (Best) | 0.842 | 0.848 |
| 🥈 | Random Forest (Best) | 0.819 | 0.818 |
| 🥉 | Random Forest (Base) | 0.810 | 0.805 |
| 4 | Gradient Boosting (Base) | 0.788 | 0.776 |
| 5 | Ridge (Best) | 0.611 | 0.576 |
| 6 | Ridge (Base) | 0.601 | 0.576 |

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive Explanations) was used to interpret model predictions.

### 📌 Global Feature Importance (Across Models)

| Feature | Ridge | Random Forest | Gradient Boosting |
|--------|------|--------------|------------------|
| Latitude | High | Medium | High |
| Longitude | High | Medium | High |
| MedInc | Medium | High | Medium |
| Population | High | Low | Very Low |
| HouseAge | Medium | Low | Low |
| AveRooms | Low | Low | Low |
| AveOccup | Low | Medium | Medium |
| AveBedrms | Low | Low | Low |

---

#### 📊 Key SHAP Insight

Most important drivers of house prices:
1. Location (Latitude / Longitude)
2. Income (MedInc)
3. Demographics (Population, Occupancy)


##### 💾 Model Persistence (Production Ready)

To ensure reproducibility and deployment readiness, the final trained model is serialized using `joblib`.

This enables:
- Fast loading without retraining
- Consistent inference results
- Easy integration into APIs or applications

###### 🧠 Key Insights

- Location (Latitude / Longitude) is the strongest predictor of housing prices
- Income (MedInc) is highly correlated with house value
- Tree-based models outperform linear models significantly


###### 🏁 Conclusion

Gradient Boosting achieved the best performance with the highest R² score, making it the most suitable model for this dataset.


###### 🚀 How to Run

```bash
python main.py