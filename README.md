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


## 📂 Project Structure
     california-housing-ml/
     │
     ├── .gitignore
     ├── README.md
     ├── main.py
     ├── requirements.txt
     │
     ├── models/
     │   └── gb_best_model.pkl
     │
     └── notebooks/
         ├── 01_housing_eda.ipynb
         └── 02_modeling.ipynb

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
- Cross-validation Mean R²
- Cross-validation Standard Deviation (model stability)
---

## 🏆 Final Results (Leaderboard)

| Rank | Model                    | CV R²| Test R²|
|------|--------------------------|------|--------|
| 🥇  | Gradient Boosting (Best) | 0.842 | 0.848 |
| 🥈  | Random Forest (Best)     | 0.819 | 0.818 |
| 🥉  | Random Forest (Base)     | 0.805 | 0.805 |
| 4    | Gradient Boosting (Base) | 0.788 | 0.776 |
| 5    | Ridge (Best)             | 0.611 | 0.576 |
| 6    | Ridge (Base)             | 0.611 | 0.576 |

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive Explanations) was used to interpret model predictions.

### 📌 Global Feature Importance (Across Models)

| Feature  | Ridge  | Random Forest | Gradient Boosting |
|----------|--------|---------------|-------------------|
|Latitude  |High    |Medium         |High               |
|Longitude |High    |Medium         |High               |
|MedInc    |High    |High           |Medium             |
|AveOccup  |Very Low|Medium         |Medium             |
|AveRooms  |Low     |Low            |Low                |
|HouseAge  |Low     |Low            |Low                |
|AveBedrms |Low     |Very Low       |Very Low           |
|Population|Very Low|Very Low       |Very Low           |

---

#### 📊 Key SHAP Insight

SHAP analysis shows that:
1. **Location (Latitude and Longitude)** are consistently among the strongest predictors of house prices across all models.
2. **Median Income (MedInc)** is one of the most influential features, especially for Ridge and Random Forest.
3. **Average Occupancy (AveOccup)** has low impact in the linear model but becomes more important in tree-based models.
4. Features such as **Population and AveBedrms** show consistently low contribution to model predictions.

##### 💾 Model Persistence (Production Ready)

To ensure reproducibility and deployment readiness, the final trained model is serialized using `joblib`.

This enables:
- Fast loading without retraining
- Consistent inference results
- Easy integration into APIs or applications

Saved model:

`models/gb_best_model.pkl`


###### 🧠 Key Insights

- SHAP analysis shows that geographical features (Latitude and Longitude) are consistently among the most important predictors across models.

- Median Income (MedInc) is also a major contributor, especially in Ridge and Random Forest.

- Tree-based models capture additional nonlinear relationships, making features like AveOccup more influential compared to the linear Ridge model.


###### 🏁 Conclusion

Gradient Boosting achieved the best performance with the highest R² score, making it the most suitable model for this dataset.


###### 🚀 How to Run

```bash
python main.py
```

## 📈 Best Model

The final selected model is:

**Gradient Boosting Regressor**

Best parameters:

```python
{
'learning_rate': 0.1,
'max_depth': 5,
'n_estimators': 1000,
'subsample': 0.8
}
```

###### 🔮 Future Improvements

Possible future improvements:

- Feature engineering for numerical features
- Experiment with XGBoost / LightGBM
- Model deployment using FastAPI
- Automated ML pipeline with CI/CD