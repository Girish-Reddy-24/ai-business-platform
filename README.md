 AI Business Decision Platform

An end-to-end AI-powered SaaS platform that helps businesses make smarter decisions using:
	•	🔴 Customer Churn Prediction (with explainability)
	•	📈 Demand Forecasting
	•	💬 Customer Feedback Analysis (NLP)
	•	🌐 Fully deployed (FastAPI + Streamlit on Render)



📊 Live Demo
	• https://ai-business-intelligence-platform2.onrender.com/




🧠 What This Project Does

This platform allows businesses to:

🔴 1. Predict Customer Churn
	•	Input customer details
	•	Get:
	•	Risk prediction (leave/stay)
	•	Probability score
	•	Key business drivers (explainable AI)

⸻

📈 2. Forecast Demand
	•	Predict future trends using ML
	•	Visualized as a time-series chart

⸻

💬 3. Analyze Customer Feedback
	•	Uses NLP (TF-IDF + Clustering)
	•	Identifies common customer issues automatically


🏗️ Architecture
Frontend (Streamlit - Render)
        ↓
Backend API (FastAPI - Render)
        ↓
Machine Learning Models


⚙️ Tech Stack
	•	Backend: FastAPI, Uvicorn
	•	Frontend: Streamlit
	•	ML: Scikit-learn (RandomForest, Linear Regression)
	•	Data: Pandas, NumPy
	•	NLP: TF-IDF + KMeans
	•	Deployment: Render


📂 Project Structure
ai-business-platform/
│
├── backend/
│   └── app/
│       └── main.py        # FastAPI backend
│
├── app.py                 # Streamlit frontend
├── churn_model.py         # Model training
├── forecasting_model.py   # Forecasting model
├── nlp_model.py           # NLP processing
│
├── model.pkl
├── training_data.pkl
├── forecast_model.pkl
│
├── requirements.txt
└── README.md


💡 Key Features
	•	✅ Explainable AI (not black box)
	•	✅ Real-world dataset (100K+ records)
	•	✅ Multi-module system (ML + NLP + Forecasting)
	•	✅ Production deployment
	•	✅ Business-friendly insights


🤝 Contributing
Feel free to fork, improve, and submit PRs!


⭐ If you like this project
Give it a ⭐ on GitHub!
:::