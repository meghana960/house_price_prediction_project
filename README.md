# House Price Prediction — Machine Learning Project

A beginner-friendly end-to-end regression project with a Streamlit web interface.

## Features
- Explore the included housing dataset
- Train a Random Forest regression pipeline
- Encode categorical features automatically
- Evaluate using Mean Absolute Error (MAE) and R²
- Predict an estimated price from user-entered property details
- Save the trained pipeline with Joblib

## Project structure
```text
house_price_prediction_project/
├── app.py
├── train.py
├── src_predictor.py
├── requirements.txt
├── README.md
├── data/
│   └── house_data.csv
└── models/
    └── (trained model is created here)
```

## Requirements
- Python 3.9 or newer recommended
- Internet connection for installing Python packages

## Run on Windows (PowerShell)
```powershell
cd house_price_prediction_project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Run on macOS/Linux
```bash
cd house_price_prediction_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

## Dataset columns
| Column | Meaning |
|---|---|
| area_sqft | Built-up area in square feet |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| age_years | Approximate property age |
| location | Prime, Good, متوسط, or Developing |
| parking | Number of parking spaces |
| furnishing | Furnished, Semi-Furnished, or Unfurnished |
| price_lakhs | Target house price in lakh rupees |

## Important dataset note
`data/house_data.csv` is a **synthetic sample dataset generated for demonstration and practice**. It is not fetched from a real estate website and must not be used for real market valuation. For a credible academic or real-world model, replace it with a properly licensed, verified dataset. Keep the same column names and target column, or update `FEATURES` and `TARGET` in `src_predictor.py`.

## Model
A scikit-learn Pipeline combines:
1. ColumnTransformer — passes numeric fields through and one-hot encodes categorical fields.
2. RandomForestRegressor — learns nonlinear relationships between property features and price.

The data is split into 80% training and 20% test sets using a fixed random seed for reproducibility.

## Troubleshooting
- If `streamlit` is not recognized, run `python -m streamlit run app.py`.
- If package installation fails, upgrade pip: `python -m pip install --upgrade pip`.
- Run commands from the project root directory.
