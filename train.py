from pathlib import Path
from src.predictor import train_model
import joblib

DATA = Path("data/house_data.csv")
OUTPUT = Path("models/house_price_pipeline.joblib")

model, metrics = train_model(DATA)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": model,
    "metrics": metrics
}, OUTPUT)

print("Training completed!")
print("MAE:", metrics["mae"])
print("R2:", metrics["r2"])