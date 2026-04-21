import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("linear_regression_model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form['age'])
        gender = request.form['gender']
        blood = request.form['blood']
        condition = request.form['condition']

        data = [[age, gender, blood, condition]]
        transformed_data = transformer.transform(data)

        prediction = model.predict(transformed_data)

        return render_template("index.html",
                               prediction_text=f"Predicted Billing Amount: {prediction[0]:.2f}")

    except Exception as e:
        print("ERROR:", e)  # muncul di logs
        return render_template("index.html",
                               prediction_text=f"ERROR: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
