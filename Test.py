import pandas as pd
import pickle

# Загрузка
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

print("Введите данные:\n")

user_data = {}

for feature in feature_names:
    while True:
        try:
            value = float(input(f"{feature}: "))
            user_data[feature] = value
            break
        except:
            print("Введите число!")

df_input = pd.DataFrame([user_data])
scaled = scaler.transform(df_input)

cluster = model.predict(scaled)[0]

print("\nРезультат:")
print(f"Кластер: {cluster}")