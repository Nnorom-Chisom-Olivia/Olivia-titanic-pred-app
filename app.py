from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and columns
model = joblib.load("model/titanic_survival_model.pkl")
model_columns = joblib.load("model/titanic_columns.joblib")

@app.route("/")
def home():
    return render_template("index.html") 

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data from HTML
        data = {
            "Pclass": int(request.form["Pclass"]),
            "Sex": 1 if request.form["Sex"].lower() == "male" else 0, # Match .map({'male': 1, 'female': 0})
            "Age": float(request.form["Age"]),
            "SibSp": int(request.form["SibSp"]),
            "Fare": float(request.form["Fare"])
        }

        # Create DataFrame and reindex to match training columns
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]
        result = "Survived" if prediction == 1 else "Did not Survive"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)