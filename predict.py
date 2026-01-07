import numpy as np
import pickle
import os
import json
from flask import Flask, request, jsonify

# ------------------------------------------------
# Load the trained pickle model
# ------------------------------------------------
model_path = "loan_default_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully!")

# ------------------------------------------------
# Example: Single Prediction
# ------------------------------------------------
# Provide new values in SAME order as the training dataset:
# Age, Salary, CreditScore, LoanAmount, Tenure

# sample_data = {
#     "Age": [45],
#     "Salary": [702000],
#     "CreditScore": [750],
#     "LoanAmount": [200000],
#     "Tenure": [3]
# }

# df = pd.DataFrame(sample_data)

# prediction = model.predict(df)[0]
# probability = model.predict_proba(df)[0][1]

# print("\nInput Data:")
# print(df)

# print("\nPrediction:", prediction)
# print("Probability of Default:", probability)

# Initializing Flask app
app = Flask(__name__)

@app.route("/index",methods=["GET"])
def index():
    return "Welcome to the Loan Default API!"

@app.route("/predict", methods=["POST"])
def predict():
    # JSON data as a string
    user_input = request.json

    # Parse JSON into a dictionary
    Age = int(user_input.get("Age",0))
    Salary = int(user_input.get("Salary",0))
    CreditScore = int(user_input.get("CreditScore",0))
    LoanAmount = int(user_input.get("LoanAmount",0))
    Tenure = int(user_input.get("Tenure",0))

    user_input_predictions = np.array([[Age, Salary, CreditScore, LoanAmount, Tenure]])

    prediction = model.predict(user_input_predictions)
    return f"Loan Default Prediction: {prediction[0]}"

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)


# {
#     "room": 2,
#     "area": 2000
# }