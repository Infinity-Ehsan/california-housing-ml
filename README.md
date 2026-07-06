# 🏠 California Housing Price Prediction (End-to-End Machine Learning Project)

A complete end-to-end Machine Learning project for predicting median house prices in California using the **California Housing dataset** from Scikit-learn.

This project demonstrates a full ML pipeline including data exploration, feature engineering, model building, hyperparameter tuning, evaluation, and model interpretability using SHAP.

---

# 📊 Dataset

- Source: `sklearn.datasets.fetch_california_housing`
- Samples: 20,640
- Features: 8 numerical features
- Target: Median house value (continuous regression problem)

---

# 🎯 Objective

The goal is to build and compare multiple regression models to predict housing prices accurately while ensuring interpretability and robustness.

---

# 🧠 Models Used

## 🌱 Linear Regression (Baseline)
- Simple linear model
- Used as a baseline reference

## 🌲 Random Forest Regressor
- Bagging ensemble method
- High stability and robustness
- Handles non-linear relationships well

## 🌿 Gradient Boosting Regressor
- Boosting ensemble method
- High predictive performance
- More sensitive to hyperparameters

---

# ⚙️ ML Pipeline

All models were implemented using **Scikit-learn Pipelines** to ensure:

- No data leakage
- Clean workflow
- Reproducibility
- Scalable structure

---

# 🔍 Hyperparameter Tuning

Hyperparameters were optimized using **GridSearchCV** with cross-validation.

### 🌲 Random Forest:
- n_estimators
- max_depth
- min_samples_split
- max_features

### 🌿 Gradient Boosting:
- n_estimators
- learning_rate
- max_depth
- subsample

---

# 📊 Model Performance Comparison

```python
import pandas as pd

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
    "Mean_CV_R2": [0.62, 0.8125, 0.8387],
    "CV_Std": [0.08, 0.00047, 0.0046]
})

results.sort_values(by="Mean_CV_R2", ascending=False)


🏆 Key Results

🌿 Gradient Boosting achieved the highest predictive performance (R² ≈ 0.84)
🌲 Random Forest showed the highest stability (lowest variance)
🌱 Linear Regression served as a simple baseline model


🔍 Model Interpretability (SHAP)

SHAP (SHapley Additive exPlanations) was used to interpret model predictions.

Key Insights:

📌 Median Income (MedInc) is the strongest predictor of house value
📌 Geographic location (Latitude & Longitude) strongly influences prices
📌 Random Forest distributes feature importance more evenly
📌 Gradient Boosting focuses more on key predictive features


🚀 How to Run

git clone https://github.com/your-username/california-housing-ml.git
cd california-housing-ml

pip install -r requirements.txt


Run notebooks in order:

📊 EDA
🧠 Modeling
⚙️ Hyperparameter Tuning
🔍 SHAP Analysis


📦 Requirements

numpy
pandas
scikit-learn
matplotlib
seaborn
shap


🧠 Skills Demonstrated

- End-to-end Machine Learning pipeline design
- Regression modeling
- Feature engineering & preprocessing
- Cross-validation (K-Fold)
- Hyperparameter tuning (GridSearchCV)
- Model comparison & evaluation
- Feature importance analysis
- SHAP explainability
- Production-style ML workflow


📌 Future Improvements

🚀 Experiment with XGBoost / LightGBM
🌐 Deploy model using FastAPI
📊 Build interactive dashboard (Streamlit)
💾 Add model persistence (joblib / pickle)
🐳 Dockerize the project


⭐ Conclusion

This project demonstrates a complete machine learning workflow from raw data to interpretable and optimized models, focusing on both predictive performance and model explainability.
It highlights the trade-off between:
🌿 Higher accuracy (Gradient Boosting)
🌲 Higher stability (Random Forest)


📬 Author

Built as a personal ML learning project focused on real-world workflow, model comparison, and explainability.