from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

app = FastAPI()

# ===============================
# LOAD MODELS
# ===============================
model = joblib.load("model.pkl")
forecast_model = joblib.load("forecast_model.pkl")
training_data = joblib.load("training_data.pkl")

# ===============================
# INPUT SCHEMA
# ===============================
class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float


# ===============================
# HOME
# ===============================
@app.get("/")
def home():
    return {"message": "AI Business Platform is running 🚀"}


# ===============================
# CHURN PREDICTION (FINAL)
# ===============================
@app.post("/predict")
def predict(data: CustomerData):
    try:
        df = training_data.copy()

        # -----------------------------
        # 🔥 Find closest real customer
        # -----------------------------
        df["distance"] = (
            abs(df["tenure_months"] - data.tenure) +
            abs(df["monthly_charges"] - data.MonthlyCharges) +
            abs(df["total_charges"] - data.TotalCharges)
        )

        nearest = df.loc[df["distance"].idxmin()].drop("distance")

        input_dict = nearest.to_dict()

        # Override user inputs
        input_dict["tenure_months"] = data.tenure
        input_dict["monthly_charges"] = data.MonthlyCharges
        input_dict["total_charges"] = data.TotalCharges

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[model.feature_names_in_]

        # -----------------------------
        # 🔥 Prediction
        # -----------------------------
        prediction = model.predict(input_df)[0]

        proba = model.predict_proba(input_df)[0]
        probability = proba[1] if len(proba) > 1 else proba[0]

        result = "Customer will leave ❌" if prediction == 1 else "Customer will stay ✅"

        # -----------------------------
        # 🔥 Explainability (CLEAN)
        # -----------------------------
        importances = model.feature_importances_
        feature_names = model.feature_names_in_

        feature_importance = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

        # ✅ ONLY BUSINESS-RELEVANT FEATURES
        allowed_features = [
            "csat_score",
            "num_support_tickets",
            "num_tech_tickets",
            "monthly_charges",
            "tenure_months",
            "network_outages_6mo",
            "avg_download_speed_mbps",
            "packet_loss_pct",
            "late_payments",
            "contract_type"
        ]

        filtered = [f for f in feature_importance if f[0] in allowed_features]

        top_features = [f[0] for f in filtered[:5]]

        # -----------------------------
        # 🔥 Human-friendly mapping
        # -----------------------------
        feature_map = {
            "csat_score": "Customer Satisfaction",
            "num_support_tickets": "Support Issues",
            "num_tech_tickets": "Technical Issues",
            "monthly_charges": "Monthly Cost",
            "tenure_months": "Customer Loyalty",
            "network_outages_6mo": "Network Reliability",
            "avg_download_speed_mbps": "Internet Speed",
            "packet_loss_pct": "Connection Quality",
            "late_payments": "Payment Delays",
            "contract_type": "Contract Type"
        }

        top_features_clean = [feature_map.get(f, f) for f in top_features]

        return {
            "prediction": result,
            "churn_probability": min(round(probability * 100, 2), 95.0),
            "top_factors": top_features_clean
        }

    except Exception as e:
        return {"error": str(e)}


# ===============================
# FORECASTING
# ===============================
@app.get("/forecast")
def forecast(days: int = 10):

    future_days = pd.DataFrame({
        "day": range(100, 100 + days)
    })

    predictions = forecast_model.predict(future_days)

    return {"forecast": predictions.tolist()}


# ===============================
# NLP INSIGHTS
# ===============================
@app.post("/nlp-insights")
def nlp_insights():

    reviews = [
        "Delivery was very slow",
        "Payment failed multiple times",
        "Customer support was not helpful",
        "App crashes frequently",
        "Delivery delayed again",
        "Bad customer service experience",
        "Payment issues again",
        "App is very slow"
    ]

    df = pd.DataFrame({"review": reviews})

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["review"])

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    terms = vectorizer.get_feature_names_out()

    insights = {}

    for i in range(3):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-5:]
        keywords = [terms[index] for index in top_indices]
        insights[f"cluster_{i}"] = keywords

    return {"insights": insights}