import streamlit as st
import pandas as pd
from pathlib import Path
from src.predictor import load_or_train, predict_price

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 House Price Prediction")
st.caption("Estimate a property's value using a machine-learning regression model.")

MODEL_PATH = Path("models/house_price_pipeline.joblib")
DATA_PATH = Path("data/house_data.csv")

with st.sidebar:
    st.header("Property details")
    area = st.number_input("Area (sq. ft.)", min_value=200, max_value=10000, value=1200, step=50)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6], index=1)
    age = st.slider("Property age (years)", 0, 60, 5)
    location = st.selectbox("Location category", ["Prime", "Good", " متوسط", "Developing"])
    parking = st.selectbox("Parking spaces", [0, 1, 2, 3])
    furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
    submitted = st.button("Predict price", type="primary", use_container_width=True)

st.info("This starter project includes a synthetic demonstration dataset. Predictions are for learning only, not real property valuation.")
pipeline, metrics = load_or_train(DATA_PATH, MODEL_PATH)

left, right = st.columns([1, 1])
with left:
    st.subheader("Model performance")
    st.metric("Test R²", f"{metrics['r2']:.3f}")
    st.metric("Test MAE", f"₹ {metrics['mae']:.2f} lakhs")
    st.write("The model is trained on the included sample dataset. Replace it with verified local property data for meaningful estimates.")
with right:
    st.subheader("Dataset preview")
    st.dataframe(pd.read_csv(DATA_PATH).head(8), use_container_width=True)

if submitted:
    row = {
        "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age_years": age, "location": location, "parking": parking,
        "furnishing": furnishing
    }
    estimate = predict_price(pipeline, row)
    st.success(f"Estimated house price: ₹ {estimate:,.2f} lakhs")
    st.caption("Actual selling prices may differ substantially.")
