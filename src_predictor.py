from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "age_years", "parking"]
CATEGORICAL = ["location", "furnishing"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "price_lakhs"

def train_model(csv_path):
    data = pd.read_csv(csv_path)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    data = data[required].dropna()
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    prep = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    ])
    model = Pipeline([
        ("preprocessor", prep),
        ("regressor", RandomForestRegressor(
            n_estimators=250, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred))
    }
    return model, metrics

def load_or_train(csv_path, model_path):
    csv_path, model_path = Path(csv_path), Path(model_path)
    # Retrain on startup so changed CSV data is picked up.
    model, metrics = train_model(csv_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics}, model_path)
    return model, metrics

def predict_price(model, property_details):
    row = pd.DataFrame([property_details], columns=FEATURES)
    return float(model.predict(row)[0])
