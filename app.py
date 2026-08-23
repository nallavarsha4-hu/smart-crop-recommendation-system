
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("crop_model.pkl")
feature_names = joblib.load("feature_names.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Read input values from the website
            values = [
                float(request.form[name])
                for name in feature_names
            ]

            # Convert inputs into a DataFrame
            input_data = pd.DataFrame(
                [values],
                columns=feature_names
            )

            # Predict the recommended crop
            prediction = model.predict(input_data)[0]

        except (ValueError, KeyError):
            error = "Please enter valid numeric values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
