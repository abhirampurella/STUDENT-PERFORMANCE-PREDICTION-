from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("student_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from the form
    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_marks = float(request.form["previous_marks"])
    assignment_score = float(request.form["assignment_score"])
    sleep_hours = float(request.form["sleep_hours"])

    # Create input data
    student_data = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_marks,
            assignment_score,
            sleep_hours
        ]],
        columns=[
            "study_hours",
            "attendance",
            "previous_marks",
            "assignment_score",
            "sleep_hours"
        ]
    )

    # Make prediction
    prediction = model.predict(student_data)[0]

    # Convert prediction number back to Pass/Fail
    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"

    # Prediction confidence
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(student_data)[0]
        confidence = round(max(probabilities) * 100, 2)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)