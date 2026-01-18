from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and columns
model = joblib.load("titanic_model.pkl")
model_columns = joblib.load("titanic_columns.joblib")

@app.route("/")
def home():
    return render_template("index.html")  # Your HTML frontend

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data from HTML
        data = {
            "Pclass": int(request.form["Pclass"]),
            "Age": float(request.form["Age"]),
            "SibSp": int(request.form["SibSp"]),
            "Parch": int(request.form["Parch"]),
            "Fare": float(request.form["Fare"]),
            "Sex_male": 1 if request.form["Sex"].lower() == "male" else 0,
            "Embarked_Q": 1 if request.form["Embarked"].upper() == "Q" else 0,
            "Embarked_S": 1 if request.form["Embarked"].upper() == "S" else 0
        }

        # Create DataFrame and reindex to match training columns
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]
        result = "Survived" if prediction == 1 else "Not Survived"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)