from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv("final_dataset.csv")
sample_data = df.head(5).to_dict(orient="records")

@app.route("/")
def index():
    return render_template(
        "index.html",
        sample_data=sample_data
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        x = float(request.form.get("X"))
        y = float(request.form.get("Y"))

        df_input = pd.DataFrame([[x, y]], columns=["X", "Y"])

        scaled = scaler.transform(df_input)
        cluster = model.predict(scaled)[0]

        return jsonify({
            "success": True,
            "cluster": int(cluster)
        })

    except:
        return jsonify({
            "success": False,
            "error": "Ошибка ввода"
        })

if __name__ == "__main__":
    print("🚀 Сервер запущен: http://127.0.0.1:5000")
    app.run(debug=True)