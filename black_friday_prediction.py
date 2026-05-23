"""
Task 5 — Black Friday Sales Prediction
Predicts how much a customer will spend using ML models.
Dataset: 550,068 purchase records with customer & product info.
"""

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("  Task 5 — Black Friday Sales Prediction")
print("=" * 60)

# ── 1. Load dataset ────────────────────────────────────────────
print("\n📂 Step 1: Loading dataset...")
data = pd.read_csv("Data/BlackFridaySales.csv")
print(f"Shape: {data.shape[0]:,} rows × {data.shape[1]} columns")
print(data.head(3))

# ── 2. Inspect ─────────────────────────────────────────────────
print("\n📊 Step 2: Dataset info:")
print(data.info())

# ── 3. Check null values ───────────────────────────────────────
print("\n🔍 Step 3: Null values:")
nulls = data.isnull().sum()
null_pct = (nulls / data.shape[0] * 100).round(1)
for col in data.columns:
    if nulls[col] > 0:
        print(f"  {col}: {nulls[col]:,} nulls ({null_pct[col]}%)")
print("→ Product_Category_2 (31%) and Product_Category_3 (69%) have nulls — will fill with 0")

# ── 4. Feature Engineering ─────────────────────────────────────
print("\n🔧 Step 4: Feature engineering & encoding...")
df = data.copy()

# One-hot encode Stay_In_Current_City_Years (has '4+' value)
df = pd.get_dummies(df, columns=['Stay_In_Current_City_Years'])

# Label encode categorical columns
le = LabelEncoder()
df['Gender']        = le.fit_transform(df['Gender'])         # F=0, M=1
df['Age']           = le.fit_transform(df['Age'])            # age bins → numbers
df['City_Category'] = le.fit_transform(df['City_Category'])  # A=0, B=1, C=2

# Fill nulls in product categories with 0
df['Product_Category_2'] = df['Product_Category_2'].fillna(0).astype('int64')
df['Product_Category_3'] = df['Product_Category_3'].fillna(0).astype('int64')

# Drop ID columns (not useful for prediction)
df = df.drop(['User_ID', 'Product_ID'], axis=1)

print("Columns after encoding:")
print(list(df.columns))
print(f"No null values remaining: {df.isnull().sum().sum() == 0} ✅")

# ── 5. Split into features and target ─────────────────────────
print("\n✂️  Step 5: Splitting data...")
X = df.drop('Purchase', axis=1)
y = df['Purchase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)
print(f"Training samples : {len(X_train):,}")
print(f"Testing  samples : {len(X_test):,}")

# ── Helper function ────────────────────────────────────────────
def evaluate(name, y_test, y_pred):
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"\n  {'─'*40}")
    print(f"  {name}")
    print(f"  {'─'*40}")
    print(f"  MAE  : {mae:,.2f}  (avg error in rupees)")
    print(f"  RMSE : {rmse:,.2f}  (lower is better)")
    print(f"  R²   : {r2:.4f}   (1.0 = perfect)")
    return rmse, r2

results = {}

# ── 6. Model 1: Linear Regression ─────────────────────────────
print("\n🤖 Step 6: Training Model 1 — Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse, r2 = evaluate("Linear Regression", y_test, y_pred_lr)
results['Linear Regression'] = (rmse, r2)

# ── 7. Model 2: Decision Tree ──────────────────────────────────
print("\n🌳 Step 7: Training Model 2 — Decision Tree Regressor...")
dt = DecisionTreeRegressor(random_state=0)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
rmse, r2 = evaluate("Decision Tree Regressor", y_test, y_pred_dt)
results['Decision Tree'] = (rmse, r2)

# ── 8. Model 3: Random Forest ──────────────────────────────────
print("\n🌲 Step 8: Training Model 3 — Random Forest Regressor...")
print("  (this may take 2-3 minutes on 550k rows...)")
rf = RandomForestRegressor(random_state=0, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse, r2 = evaluate("Random Forest Regressor", y_test, y_pred_rf)
results['Random Forest'] = (rmse, r2)

# ── 9. Model 4: XGBoost ────────────────────────────────────────
print("\n⚡ Step 9: Training Model 4 — XGBoost Regressor...")
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40,
                       seed=0, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    rmse, r2 = evaluate("XGBoost Regressor", y_test, y_pred_xgb)
    results['XGBoost'] = (rmse, r2)
except ImportError:
    print("  XGBoost not installed. Run: pip install xgboost")
    print("  Skipping XGBoost...")

# ── 10. Final Comparison ───────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL MODEL COMPARISON")
print("=" * 60)
print(f"\n  {'Model':<25} {'RMSE':>10} {'R²':>10}")
print(f"  {'─'*45}")
best_model = min(results, key=lambda k: results[k][0])
for name, (rmse, r2) in sorted(results.items(), key=lambda x: x[1][0]):
    star = " ⭐ BEST" if name == best_model else ""
    print(f"  {name:<25} {rmse:>10,.2f} {r2:>10.4f}{star}")

print(f"""
{'='*60}
  CONCLUSION
{'='*60}

Dataset  : 550,068 Black Friday purchase records
Target   : Predict Purchase Amount (in rupees)
Best Model: {best_model} with RMSE = {results[best_model][0]:,.2f}

Lower RMSE = better predictions.
The model helps retailers personalise offers for customers
by predicting how much each person is likely to spend
on different product categories during Black Friday.
""")
