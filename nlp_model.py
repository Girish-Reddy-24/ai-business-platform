import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Step 1: Sample reviews (we will replace with real later)
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

# Step 2: Convert text → numbers (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["review"])

# Step 3: Cluster reviews into groups
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

df["cluster"] = kmeans.labels_

# Step 4: Show results
print(df)

# Get top keywords per cluster
terms = vectorizer.get_feature_names_out()

for i in range(3):
    print(f"\nCluster {i} keywords:")
    center = kmeans.cluster_centers_[i]
    top_indices = center.argsort()[-5:]
    keywords = [terms[index] for index in top_indices]
    print(keywords)
