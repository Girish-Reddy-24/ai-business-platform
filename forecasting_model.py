import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Create sample time-series data
dates = pd.date_range(start="2023-01-01", periods=100)
sales = np.random.randint(100, 200, size=100)

df = pd.DataFrame({
    "date": dates,
    "sales": sales
})

# Step 2: Convert date to number
df["day"] = np.arange(len(df))

# Step 3: Train model
X = df[["day"]]
y = df["sales"]

model = LinearRegression()
model.fit(X, y)

# Step 4: Predict next 10 days
future_days = np.arange(len(df), len(df) + 10).reshape(-1, 1)
predictions = model.predict(future_days)

# Step 5: Print predictions
print("Future Predictions:")
print(predictions)

# Plot
import matplotlib.pyplot as plt

plt.plot(df["date"], df["sales"], label="Actual")
future_dates = pd.date_range(start=df["date"].iloc[-1], periods=10)
plt.plot(future_dates, predictions, label="Forecast", color="red")

plt.legend()
plt.show()

import joblib

joblib.dump(model, "forecast_model.pkl")
print("Forecast model saved ✅")