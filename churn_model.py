import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ===============================
# STEP 1: Load dataset
# ===============================
df = pd.read_csv("data/telco_churn_100k.csv")

# ===============================
# STEP 2: Clean data
# ===============================
df = df.dropna()

# Remove non-actionable demographic features
drop_cols = [
    "customer_id",
    "churn_date",
    "churn_reason",
    "join_date",

    # 👇 ADD THESE (IMPORTANT)
    "gender",
    "age",
    "marital_status",
    "dependents",
    "education_level",
    "income_bracket",
    "state"
]

df = df.drop(columns=drop_cols, errors="ignore")




# ===============================
# STEP 3: Encode categorical columns
# ===============================
label_encoders = {}

for column in df.columns:
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# ===============================
# STEP 4: Split features & target
# ===============================
X = df.drop(columns=["churn"])
y = df["churn"]

# ===============================
# STEP 5: Save training data (IMPORTANT)
# ===============================
joblib.dump(X, "training_data.pkl")

# ===============================
# STEP 6: Train model
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===============================
# STEP 7: Save supporting artifacts
# ===============================

# Mean values (optional fallback)
default_values = X.mean().to_dict()
joblib.dump(default_values, "defaults.pkl")

# Real baseline row
baseline = X.sample(1, random_state=42).to_dict(orient="records")[0]
joblib.dump(baseline, "baseline.pkl")

# Save encoders (for future use)
joblib.dump(label_encoders, "encoders.pkl")

# Save model
joblib.dump(model, "model.pkl")

# ===============================
# DONE
# ===============================
print("Model + training data + baseline + defaults saved ✅")