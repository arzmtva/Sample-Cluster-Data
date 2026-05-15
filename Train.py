import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("🚀 Запуск обучения...\n")

df = pd.read_csv("Sample_Cluster_Data.csv")

df = df[["X", "Y"]].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df["Cluster"] = kmeans.labels_

os.makedirs("static", exist_ok=True)

with open("model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

df.to_csv("final_dataset.csv", index=False)

plt.figure(figsize=(8, 6))
plt.scatter(df["X"], df["Y"], c=df["Cluster"])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Кластеризация данных")
plt.savefig("static/cluster_plot.png")
plt.close()

print("✅ Обучение завершено!")