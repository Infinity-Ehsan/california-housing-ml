import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------
# 1. Load Data
# -----------------------
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]

# -----------------------
# 2. Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# 3. BEST MODEL (from GridSearchCV)
# -----------------------
best_gb_model = Pipeline([
    ("scaler", StandardScaler()),
    ("gb", GradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=1000,
        subsample=0.8,
        random_state=42
    ))
])

# -----------------------
# 4. Train
# -----------------------
best_gb_model.fit(X_train, y_train)

# -----------------------
# 5. Predict
# -----------------------
y_pred = best_gb_model.predict(X_test)

# -----------------------
# 6. Evaluate
# -----------------------
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("="*40)
print("🏆 BEST Gradient Boosting Model")
print("="*40)
print(f"R2 Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print("="*40)